import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_add_pool, global_mean_pool
from torch_geometric.utils import softmax


FEATURE_CLAMP = 1e4
LOGIT_CLAMP = 50.0


def finite_tensor(value, clamp_value=FEATURE_CLAMP):
    return torch.nan_to_num(
        value,
        nan=0.0,
        posinf=clamp_value,
        neginf=-clamp_value,
    ).clamp(-clamp_value, clamp_value)


def finite_logits(value):
    return finite_tensor(value, clamp_value=LOGIT_CLAMP)


def transform_descriptor_features(prop):
    x = finite_tensor(prop.float()).clone()
    if x.dim() == 1:
        x = x.unsqueeze(0)
    return finite_tensor(x)


class FingerprintEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout):
        super().__init__()
        self.input_dim = int(input_dim)
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, x):
        return finite_tensor(self.net(finite_tensor(x.float())))


class MultiFingerprintEncoder(nn.Module):
    def __init__(
        self,
        fingerprint_specs,
        hidden_dim,
        dropout,
        gate_temperature,
        gate_init,
    ):
        super().__init__()
        self.specs = [dict(spec) for spec in fingerprint_specs]
        self.names = [str(spec["name"]) for spec in self.specs]
        self.dims = [int(spec["dim"]) for spec in self.specs]
        self.input_dim = int(sum(self.dims))
        self.hidden_dim = int(hidden_dim)
        self.gate_temperature = max(float(gate_temperature), 0.1)
        self.encoders = nn.ModuleDict(
            {
                name: FingerprintEncoder(dim, self.hidden_dim, dropout)
                for name, dim in zip(self.names, self.dims)
            }
        )
        gate_input_dim = self.hidden_dim * len(self.names)
        self.gate = nn.Sequential(
            nn.LayerNorm(gate_input_dim),
            nn.Linear(gate_input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, len(self.names)),
        )
        nn.init.zeros_(self.gate[-1].weight)
        initial = torch.tensor(
            [
                max(float(dict(gate_init or {}).get(name, 1.0)), 1e-4)
                for name in self.names
            ],
            dtype=torch.float32,
        )
        initial = initial / initial.sum().clamp(min=1e-6)
        self.register_buffer("initial_prior", initial.clone())
        with torch.no_grad():
            self.gate[-1].bias.copy_(
                torch.log(initial) * self.gate_temperature
            )

    def forward(self, x):
        x = finite_tensor(x.float())
        if x.size(-1) != self.input_dim:
            raise ValueError(
                f"Expected fingerprint dim {self.input_dim}, got {x.size(-1)}"
            )
        chunks = torch.split(x, self.dims, dim=-1)
        embeddings = {
            name: self.encoders[name](chunk)
            for name, chunk in zip(self.names, chunks)
        }
        concat = finite_tensor(
            torch.cat([embeddings[name] for name in self.names], dim=-1)
        )
        gate_scores = finite_logits(self.gate(concat))
        weights = torch.softmax(
            gate_scores / self.gate_temperature,
            dim=-1,
        )
        stack = finite_tensor(
            torch.stack([embeddings[name] for name in self.names], dim=1)
        )
        pooled = finite_tensor(
            (stack * weights.unsqueeze(-1)).sum(dim=1)
        )
        return pooled, weights, embeddings


class GINEEncoder(nn.Module):
    """Full-C GINE: virtual node, JK-concat and attention pooling."""

    def __init__(
        self,
        hidden_dim,
        atom_vocab_size,
        chirality_vocab_size,
        bond_vocab_size,
        bond_direction_vocab_size,
        num_layers,
        dropout,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)

        self.atom_embedding = nn.Embedding(atom_vocab_size, hidden_dim)
        self.chirality_embedding = nn.Embedding(
            chirality_vocab_size,
            hidden_dim,
        )
        self.degree_embedding = nn.Embedding(8, hidden_dim)
        self.formal_charge_embedding = nn.Embedding(5, hidden_dim)
        self.hybridization_embedding = nn.Embedding(6, hidden_dim)
        self.atom_aromatic_embedding = nn.Embedding(2, hidden_dim)
        self.num_h_embedding = nn.Embedding(6, hidden_dim)
        self.atom_ring_embedding = nn.Embedding(2, hidden_dim)
        self.atom_feature_proj = nn.Sequential(
            nn.Linear(hidden_dim * 8, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        self.bond_embedding = nn.Embedding(bond_vocab_size, hidden_dim)
        self.bond_direction_embedding = nn.Embedding(
            bond_direction_vocab_size,
            hidden_dim,
        )
        self.bond_stereo_embedding = nn.Embedding(6, hidden_dim)
        self.bond_conjugated_embedding = nn.Embedding(2, hidden_dim)
        self.bond_aromatic_embedding = nn.Embedding(2, hidden_dim)
        self.bond_ring_embedding = nn.Embedding(2, hidden_dim)
        self.bond_feature_proj = nn.Sequential(
            nn.Linear(hidden_dim * 6, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(self.num_layers):
            self.convs.append(
                GINEConv(
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim * 2),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(hidden_dim * 2, hidden_dim),
                    ),
                    edge_dim=hidden_dim,
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim))
        self.dropout = nn.Dropout(dropout)

        self.virtualnode_embedding = nn.Embedding(1, hidden_dim)
        self.virtualnode_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                )
                for _ in range(max(self.num_layers - 1, 1))
            ]
        )
        self.jk_proj = nn.Sequential(
            nn.Linear(hidden_dim * (self.num_layers + 1), hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        pool_hidden = hidden_dim // 2 if hidden_dim >= 2 else 1
        self.pool_gate = nn.Sequential(
            nn.Linear(hidden_dim, pool_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(pool_hidden, 1),
        )
        self.pool_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    @staticmethod
    def _pad_columns(x, width):
        if x.size(1) >= width:
            return x[:, :width]
        padded = x.new_zeros((x.size(0), width))
        padded[:, : x.size(1)] = x
        return padded

    def _embed_atoms(self, x):
        x = self._pad_columns(x.long(), 8)
        parts = [
            self.atom_embedding(
                x[:, 0].clamp(0, self.atom_embedding.num_embeddings - 1)
            ),
            self.chirality_embedding(
                x[:, 1].clamp(
                    0,
                    self.chirality_embedding.num_embeddings - 1,
                )
            ),
            self.degree_embedding(
                x[:, 2].clamp(0, self.degree_embedding.num_embeddings - 1)
            ),
            self.formal_charge_embedding(
                x[:, 3].clamp(
                    0,
                    self.formal_charge_embedding.num_embeddings - 1,
                )
            ),
            self.hybridization_embedding(
                x[:, 4].clamp(
                    0,
                    self.hybridization_embedding.num_embeddings - 1,
                )
            ),
            self.atom_aromatic_embedding(x[:, 5].clamp(0, 1)),
            self.num_h_embedding(
                x[:, 6].clamp(0, self.num_h_embedding.num_embeddings - 1)
            ),
            self.atom_ring_embedding(x[:, 7].clamp(0, 1)),
        ]
        return finite_tensor(
            self.atom_feature_proj(torch.cat(parts, dim=-1))
        )

    def _embed_bonds(self, edge_attr, hidden_ref):
        if edge_attr.numel() == 0:
            return hidden_ref.new_empty((0, hidden_ref.size(-1)))
        edge_attr = self._pad_columns(edge_attr.long(), 6)
        parts = [
            self.bond_embedding(
                edge_attr[:, 0].clamp(
                    0,
                    self.bond_embedding.num_embeddings - 1,
                )
            ),
            self.bond_direction_embedding(
                edge_attr[:, 1].clamp(
                    0,
                    self.bond_direction_embedding.num_embeddings - 1,
                )
            ),
            self.bond_stereo_embedding(
                edge_attr[:, 2].clamp(
                    0,
                    self.bond_stereo_embedding.num_embeddings - 1,
                )
            ),
            self.bond_conjugated_embedding(edge_attr[:, 3].clamp(0, 1)),
            self.bond_aromatic_embedding(edge_attr[:, 4].clamp(0, 1)),
            self.bond_ring_embedding(edge_attr[:, 5].clamp(0, 1)),
        ]
        return finite_tensor(
            self.bond_feature_proj(torch.cat(parts, dim=-1))
        )

    def forward(self, x, edge_index, edge_attr, batch):
        h = finite_tensor(self._embed_atoms(x))
        edge_emb = finite_tensor(self._embed_bonds(edge_attr, h))
        layer_outputs = [h]
        num_graphs = int(batch.max().item()) + 1 if batch.numel() else 1
        virtualnode_emb = (
            self.virtualnode_embedding.weight[0]
            .unsqueeze(0)
            .repeat(num_graphs, 1)
        )

        for layer_idx, (conv, norm) in enumerate(
            zip(self.convs, self.norms)
        ):
            h = finite_tensor(h + virtualnode_emb[batch])
            h_new = finite_tensor(conv(h, edge_index, edge_emb))
            h = finite_tensor(norm(h + self.dropout(h_new)))
            h = finite_tensor(F.gelu(h))
            layer_outputs.append(h)
            if layer_idx < self.num_layers - 1:
                pooled = global_mean_pool(h, batch)
                virtualnode_emb = finite_tensor(
                    virtualnode_emb
                    + self.virtualnode_mlps[layer_idx](pooled)
                )

        h = finite_tensor(self.jk_proj(torch.cat(layer_outputs, dim=-1)))
        weights = softmax(self.pool_gate(h).view(-1), batch).unsqueeze(-1)
        pooled = global_add_pool(h * weights, batch)
        return finite_tensor(self.pool_proj(pooled))


class DescriptorGroupEncoder(nn.Module):
    def __init__(self, group_dims, hidden_dim, dropout):
        super().__init__()
        self.group_dims = [int(dim) for dim in group_dims]
        self.group_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                )
                for dim in self.group_dims
            ]
        )

    def forward(self, prop):
        chunks = torch.split(
            transform_descriptor_features(prop),
            self.group_dims,
            dim=-1,
        )
        tokens = [
            finite_tensor(mlp(chunk))
            for mlp, chunk in zip(self.group_mlps, chunks)
        ]
        hidden = finite_tensor(torch.stack(tokens, dim=1))
        return finite_tensor(hidden.mean(dim=1))


class MultimodalModel(nn.Module):
    """MAPViS-C: graph, descriptor, fingerprint and joint experts."""

    active_modalities = ("graph", "desc", "fp")
    global_expert_names = ("graph", "desc", "fp", "joint")

    def __init__(self, config):
        super().__init__()
        model_cfg = config["model"]
        self.hidden_dim = int(model_cfg["hidden_dim"])
        self.proj_dim = int(model_cfg["proj_dim"])
        self.fingerprint_hidden_dim = int(
            model_cfg.get("fingerprint_hidden_dim", self.hidden_dim)
        )
        self.fingerprint_dim = int(model_cfg["fingerprint_dim"])
        self.fingerprint_specs = [
            {
                "name": str(spec["name"]),
                "type": str(spec.get("type", "unknown")),
                "dim": int(spec["dim"]),
            }
            for spec in model_cfg["fingerprint_specs"]
        ]
        if sum(spec["dim"] for spec in self.fingerprint_specs) != self.fingerprint_dim:
            raise ValueError(
                "fingerprint_dim must equal fingerprint_specs dimensions"
            )
        self.task_name = str(config["dataset"]["target"])
        self.task_names = [self.task_name]

        dropout = float(model_cfg.get("dropout", 0.2))
        self.modality_dropout = float(
            model_cfg.get("modality_dropout", 0.0)
        )
        self.modality_dropout_rates = {
            "graph": float(
                model_cfg.get(
                    "graph_modality_dropout",
                    self.modality_dropout,
                )
            ),
            "desc": float(
                model_cfg.get(
                    "desc_modality_dropout",
                    self.modality_dropout,
                )
            ),
            "fp": float(
                model_cfg.get(
                    "fp_modality_dropout",
                    self.modality_dropout,
                )
            ),
            "joint": float(model_cfg.get("joint_modality_dropout", 0.0)),
        }
        self.cmf_input_dropout = float(
            model_cfg.get("cmf_input_dropout", 0.0)
        )
        self.gate_entropy_weight = float(
            model_cfg.get("gate_entropy_weight", 0.0)
        )
        self.gate_prior_kl_weight = float(
            model_cfg.get("gate_prior_kl_weight", 0.0)
        )
        self.gate_balance_weight = float(
            model_cfg.get("gate_balance_weight", 0.0)
        )
        self.adaptive_gate_temperature = max(
            float(model_cfg.get("adaptive_gate_temperature", 1.0)),
            0.1,
        )

        self.encoder_2d = GINEEncoder(
            hidden_dim=self.hidden_dim,
            atom_vocab_size=int(model_cfg.get("atom_vocab_size", 119)),
            chirality_vocab_size=int(
                model_cfg.get("chirality_vocab_size", 4)
            ),
            bond_vocab_size=int(model_cfg.get("bond_vocab_size", 5)),
            bond_direction_vocab_size=int(
                model_cfg.get("bond_direction_vocab_size", 3)
            ),
            num_layers=int(model_cfg.get("gnn_layers", 3)),
            dropout=dropout,
        )
        group_dims = list(
            model_cfg.get("desc_group_dims", [10, 14, 7, 8, 2, 5])
        )
        self.desc_encoder = DescriptorGroupEncoder(
            group_dims,
            self.hidden_dim,
            dropout,
        )
        self.cmf_encoder = MultiFingerprintEncoder(
            self.fingerprint_specs,
            self.fingerprint_hidden_dim,
            dropout,
            model_cfg.get("fingerprint_gate_temperature", 1.0),
            model_cfg.get("fingerprint_gate_init", {}),
        )
        self.proj = nn.Linear(self.hidden_dim, self.proj_dim)
        self.fp_proj = (
            nn.Linear(self.fingerprint_hidden_dim, self.proj_dim)
            if self.fingerprint_hidden_dim != self.hidden_dim
            else None
        )
        self.desc_proj = nn.Linear(self.hidden_dim, self.proj_dim)
        self.joint_encoder = nn.Sequential(
            nn.LayerNorm(self.proj_dim * len(self.active_modalities)),
            nn.Linear(
                self.proj_dim * len(self.active_modalities),
                self.proj_dim,
            ),
            nn.GELU(),
            nn.Dropout(model_cfg.get("fusion_dropout", dropout)),
            nn.LayerNorm(self.proj_dim),
        )

        # test6 constructed one unused fused classifier here. Consume the same
        # RNG draws without registering dead parameters so seeds 42/43/44 keep
        # the original Full-C initialization for every active module.
        legacy_rng_head = nn.Sequential(
            nn.Linear(
                self.proj_dim * len(self.active_modalities),
                self.hidden_dim,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, 1),
        )
        del legacy_rng_head

        init_logits = self._initial_logit_weight_tensor(model_cfg)
        self.register_buffer(
            "global_gate_prior",
            torch.softmax(init_logits.detach(), dim=0),
        )
        self.modality_logit_heads = nn.ModuleDict(
            {
                self.task_name: nn.ModuleDict(
                    {
                        name: self._build_class_head()
                        for name in self.global_expert_names
                    }
                )
            }
        )
        self.fingerprint_logit_heads = nn.ModuleDict(
            {
                self.task_name: nn.ModuleDict(
                    {
                        spec["name"]: self._build_class_head()
                        for spec in self.fingerprint_specs
                    }
                )
            }
        )
        self.adaptive_gates = nn.ModuleDict(
            {
                self.task_name: self._build_adaptive_gate(
                    init_logits,
                    model_cfg,
                )
            }
        )

    def _build_class_head(self):
        return nn.Sequential(
            nn.Linear(self.proj_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self._dropout_probability()),
            nn.Linear(self.hidden_dim, 1),
        )

    def _dropout_probability(self):
        for module in self.encoder_2d.modules():
            if isinstance(module, nn.Dropout):
                return float(module.p)
        return 0.2

    def _build_adaptive_gate(self, init_logits, model_cfg):
        gate = nn.Sequential(
            nn.LayerNorm(
                self.proj_dim * len(self.global_expert_names)
            ),
            nn.Linear(
                self.proj_dim * len(self.global_expert_names),
                self.hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(
                model_cfg.get(
                    "adaptive_gate_dropout",
                    self._dropout_probability(),
                )
            ),
            nn.Linear(self.hidden_dim, len(self.global_expert_names)),
        )
        nn.init.zeros_(gate[-1].weight)
        with torch.no_grad():
            gate[-1].bias.copy_(
                init_logits * self.adaptive_gate_temperature
            )
        return gate

    def _initial_logit_weight_tensor(self, model_cfg):
        raw = model_cfg.get(
            "global_gate_prior",
            {
                "graph": 0.30,
                "fingerprint": 0.30,
                "descriptor": 0.15,
                "joint": 0.25,
            },
        )
        aliases = {
            "graph": ("graph", "2d"),
            "desc": ("desc", "descriptor", "3d"),
            "fp": ("fp", "fingerprint", "cmf"),
            "joint": ("joint",),
        }
        weights = []
        for name in self.global_expert_names:
            value = next(
                (
                    raw[alias]
                    for alias in aliases[name]
                    if alias in raw
                ),
                1.0,
            )
            weights.append(max(float(value), 1e-4))
        tensor = torch.tensor(weights, dtype=torch.float32)
        tensor = tensor / tensor.sum().clamp(min=1e-6)
        return torch.log(tensor)

    def _encode_modalities(self, graph, prop, cmf):
        graph_z = finite_tensor(
            self.proj(
                self.encoder_2d(
                    graph.x,
                    graph.edge_index,
                    graph.edge_attr,
                    graph.batch,
                )
            )
        )
        desc_z = finite_tensor(
            self.desc_proj(self.desc_encoder(prop))
        )
        cmf = cmf.to(device=prop.device, dtype=prop.dtype)
        if self.training and self.cmf_input_dropout > 0.0:
            keep = (
                torch.rand_like(cmf) >= self.cmf_input_dropout
            ).to(dtype=cmf.dtype)
            cmf = finite_tensor(cmf * keep)
        fp_pooled, fp_weights, fp_embeddings = self.cmf_encoder(cmf)
        projector = self.fp_proj if self.fp_proj is not None else self.proj
        fp_z = finite_tensor(projector(fp_pooled))
        return (
            {"graph": graph_z, "desc": desc_z, "fp": fp_z},
            {
                "weights": fp_weights,
                "embeddings": {
                    name: finite_tensor(projector(value))
                    for name, value in fp_embeddings.items()
                },
            },
        )

    def _apply_modality_dropout(self, vectors):
        batch_size = next(iter(vectors.values())).size(0)
        device = next(iter(vectors.values())).device
        keep_masks = {
            name: torch.ones(batch_size, 1, device=device)
            for name in self.active_modalities
        }
        if not self.training:
            return vectors, keep_masks
        dropped = {}
        for name in self.active_modalities:
            probability = max(
                0.0,
                min(self.modality_dropout_rates[name], 0.95),
            )
            keep = (
                torch.rand(batch_size, 1, device=device) >= probability
            ).float()
            keep_masks[name] = keep
            dropped[name] = finite_tensor(vectors[name] * keep)
        keep_stack = torch.cat(
            [keep_masks[name] for name in self.active_modalities],
            dim=1,
        )
        all_dropped = keep_stack.sum(dim=1) <= 0
        if all_dropped.any():
            restored = dropped["graph"].clone()
            restored[all_dropped] = vectors["graph"][all_dropped]
            dropped["graph"] = restored
            graph_keep = keep_masks["graph"].clone()
            graph_keep[all_dropped] = 1.0
            keep_masks["graph"] = graph_keep
        return dropped, keep_masks

    def _build_experts(self, base_z, keep_masks):
        experts = dict(base_z)
        joint_input = finite_tensor(
            torch.cat(
                [experts[name] for name in self.active_modalities],
                dim=-1,
            )
        )
        joint = finite_tensor(self.joint_encoder(joint_input))
        joint_keep = torch.ones(
            joint.size(0),
            1,
            device=joint.device,
        )
        if self.training and self.modality_dropout_rates["joint"] > 0.0:
            joint_keep = (
                torch.rand(joint.size(0), 1, device=joint.device)
                >= self.modality_dropout_rates["joint"]
            ).to(dtype=joint.dtype)
            joint = finite_tensor(joint * joint_keep)
        experts["joint"] = joint
        keep_masks = dict(keep_masks)
        keep_masks["joint"] = joint_keep
        return experts, keep_masks

    def _classify(self, experts, fingerprint_context, keep_masks):
        concat = finite_tensor(
            torch.cat(
                [experts[name] for name in self.global_expert_names],
                dim=-1,
            )
        )
        task_heads = self.modality_logit_heads[self.task_name]
        modality_logits = {}
        for name in self.global_expert_names:
            if name == "fp":
                fp_weights = fingerprint_context["weights"]
                fp_heads = self.fingerprint_logit_heads[self.task_name]
                fp_logits = {
                    spec["name"]: finite_logits(
                        fp_heads[spec["name"]](
                            fingerprint_context["embeddings"][
                                spec["name"]
                            ]
                        ).view(-1)
                    )
                    for spec in self.fingerprint_specs
                }
                fp_stack = torch.stack(
                    [
                        fp_logits[spec["name"]]
                        for spec in self.fingerprint_specs
                    ],
                    dim=1,
                )
                modality_logits["fp"] = finite_logits(
                    (fp_stack * fp_weights).sum(dim=1)
                )
                modality_logits.update(
                    {
                        f"fp:{fp_name}": value
                        for fp_name, value in fp_logits.items()
                    }
                )
                continue
            modality_logits[name] = finite_logits(
                task_heads[name](experts[name]).view(-1)
            )

        for name in self.global_expert_names:
            keep = keep_masks[name].view(-1)
            modality_logits[name] = finite_logits(
                modality_logits[name] * keep
            )
            if name == "fp":
                for key in list(modality_logits):
                    if key.startswith("fp:"):
                        modality_logits[key] = finite_logits(
                            modality_logits[key] * keep
                        )

        modal_stack = torch.stack(
            [
                modality_logits[name]
                for name in self.global_expert_names
            ],
            dim=1,
        )
        keep_stack = torch.cat(
            [keep_masks[name] for name in self.global_expert_names],
            dim=1,
        )
        gate_scores = finite_logits(
            self.adaptive_gates[self.task_name](concat)
        )
        gate_scores = gate_scores.masked_fill(
            keep_stack <= 0.0,
            -1e9,
        )
        weights = torch.softmax(
            gate_scores / self.adaptive_gate_temperature,
            dim=-1,
        )
        logits = finite_logits((modal_stack * weights).sum(dim=1))
        return (
            {self.task_name: logits},
            {self.task_name: weights},
            {self.task_name: modality_logits},
        )

    def _distribution_gate_regularization(
        self,
        weights,
        prior,
        keep_mask=None,
    ):
        eps = 1e-8
        weights = weights.clamp(min=0.0, max=1.0)
        terms = []
        entropy = -(
            weights.clamp_min(eps)
            * torch.log(weights.clamp_min(eps))
        ).sum(dim=-1)
        if keep_mask is None:
            available = weights.new_full(
                (weights.size(0),),
                float(weights.size(-1)),
            )
        else:
            available = keep_mask.float().sum(dim=-1).clamp(min=1.0)
        normalized_entropy = entropy / torch.log(
            available.clamp(min=2.0)
        )
        entropy_penalty = torch.where(
            available > 1.0,
            1.0 - normalized_entropy,
            torch.zeros_like(normalized_entropy),
        )
        if self.gate_entropy_weight > 0.0:
            terms.append(
                self.gate_entropy_weight * entropy_penalty.mean()
            )

        mean_gate = weights.mean(dim=0)
        mean_gate = mean_gate / mean_gate.sum().clamp(min=eps)
        prior = prior.to(
            device=weights.device,
            dtype=weights.dtype,
        )
        prior = prior / prior.sum().clamp(min=eps)
        if self.gate_prior_kl_weight > 0.0:
            prior_kl = (
                mean_gate.clamp_min(eps)
                * (
                    torch.log(mean_gate.clamp_min(eps))
                    - torch.log(prior.clamp_min(eps))
                )
            ).sum()
            terms.append(self.gate_prior_kl_weight * prior_kl)
        if self.gate_balance_weight > 0.0:
            uniform = torch.full_like(
                mean_gate,
                1.0 / float(weights.size(-1)),
            )
            terms.append(
                self.gate_balance_weight
                * F.mse_loss(mean_gate, uniform)
            )
        return (
            torch.stack(terms).sum()
            if terms
            else weights.new_tensor(0.0)
        )

    def forward(self, graph, prop, cmf):
        base_z, fingerprint_context = self._encode_modalities(
            graph,
            prop,
            cmf,
        )
        base_z, base_keep_masks = self._apply_modality_dropout(base_z)
        experts, expert_keep_masks = self._build_experts(
            base_z,
            base_keep_masks,
        )
        logits, adaptive_weights, modality_logits = self._classify(
            experts,
            fingerprint_context,
            expert_keep_masks,
        )
        keep_stack = torch.cat(
            [
                expert_keep_masks[name]
                for name in self.global_expert_names
            ],
            dim=1,
        )
        global_gate_reg = self._distribution_gate_regularization(
            adaptive_weights[self.task_name],
            self.global_gate_prior,
            keep_stack,
        )
        fingerprint_gate_reg = self._distribution_gate_regularization(
            fingerprint_context["weights"],
            self.cmf_encoder.initial_prior,
        )
        gate_reg = 0.5 * (
            global_gate_reg + fingerprint_gate_reg
        )
        zero = global_gate_reg.new_tensor(0.0)
        aux = {
            "align": zero,
            "global_align": zero,
            "ecoli_align": zero,
            "gate_reg": gate_reg,
            "global_gate_reg": global_gate_reg,
            "fingerprint_gate_reg": fingerprint_gate_reg,
            "property": zero,
            "modality_logits": modality_logits,
            "modality_keep_masks": expert_keep_masks,
            "adaptive_weights": adaptive_weights,
            "fingerprint_weights": fingerprint_context["weights"],
        }
        return logits, aux

    def gate_modalities(self):
        return list(self.global_expert_names)

    def fingerprint_gate_names(self):
        return [spec["name"] for spec in self.fingerprint_specs]

    def structure_summary(self):
        return {
            "active_modalities": list(self.active_modalities),
            "active_experts": list(self.global_expert_names),
            "joint_input_dimension": self.proj_dim
            * len(self.active_modalities),
            "global_gate_dimension": len(self.global_expert_names),
            "trainable_parameter_count": sum(
                parameter.numel()
                for parameter in self.parameters()
                if parameter.requires_grad
            ),
        }

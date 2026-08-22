import argparse
import copy
import json
import math
import os
import random
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from dataset.dataset import Dataset
from models.multimodal_model import MultimodalModel
from utils import AsymmetricLoss, compute_metrics


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def enable_tf32():
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")


def to_builtin(value):
    if isinstance(value, dict):
        return {key: to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def finite_mean_std(values):
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return float("nan"), float("nan")
    return float(finite.mean()), float(finite.std())


def build_save_dir(config):
    configured = config["training"].get("save_dir")
    if configured:
        return str(configured)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    task = config["task_name"]
    split = config["dataset"]["splitting"]
    seed = config["dataset"]["seed"]
    return os.path.join(
        "experiment_records",
        f"{task}_{split}_seed{seed}_{timestamp}",
    )


class Trainer:
    """Single-GPU, single-task, single-checkpoint MAPViS-C trainer."""

    def __init__(self, dataset, config):
        self.dataset = dataset
        self.config = copy.deepcopy(config)
        self.train_cfg = self.config["training"]
        self.task_name = self.config["dataset"]["target"]
        device_ids = [int(item) for item in self.train_cfg.get("device_ids", [])]
        if torch.cuda.is_available():
            device_id = device_ids[0] if device_ids else 0
            if device_id >= torch.cuda.device_count():
                raise ValueError(f"Unavailable CUDA device: {device_id}")
            self.device = torch.device(f"cuda:{device_id}")
        else:
            self.device = torch.device("cpu")
        self.save_dir = build_save_dir(self.config)
        os.makedirs(self.save_dir, exist_ok=True)
        self.config["training"]["save_dir"] = self.save_dir

        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.current_epoch = 0
        self.best_epoch = 0
        self.best_val_auc = float("nan")
        self.best_val_auprc = -float("inf")
        self.best_checkpoint_metric = -float("inf")
        self.test_metrics = {}
        self.training_time_seconds = 0.0
        self.modality_aux_weights = {
            "graph": float(self.train_cfg.get("lambda_graph_aux", 0.0)),
            "fp": float(self.train_cfg.get("lambda_fp_aux", 0.0)),
            "desc": float(self.train_cfg.get("lambda_desc_aux", 0.0)),
            "joint": float(self.train_cfg.get("lambda_joint_aux", 0.0)),
        }
        self.fingerprint_expert_aux_weight = float(
            self.train_cfg.get("lambda_fingerprint_expert_aux", 0.0)
        )

        with open(
            os.path.join(self.save_dir, "config.yaml"),
            "w",
            encoding="utf-8",
        ) as handle:
            yaml.safe_dump(
                self.config,
                handle,
                allow_unicode=True,
                sort_keys=False,
            )
        with open(
            os.path.join(self.save_dir, "config_runtime.yaml"),
            "w",
            encoding="utf-8",
        ) as handle:
            yaml.safe_dump(
                self.config,
                handle,
                allow_unicode=True,
                sort_keys=False,
            )
        run_info = {
            "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "split_seed": int(self.config["dataset"]["seed"]),
            "model_seed": int(self.train_cfg["model_seed"]),
            "run_idx": int(self.config["run_idx"]),
            "task_name": self.config["task_name"],
            "target": self.task_name,
            "split_method": self.config["dataset"]["splitting"],
            "data_path": self.config["dataset"]["data_path"],
            "save_dir": self.save_dir,
        }
        with open(
            os.path.join(self.save_dir, "run_info.json"),
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(run_info, handle, indent=2, ensure_ascii=False)

    def _move_batch(self, batch):
        if len(batch) == 6:
            _, graph, prop, cmf, label, smiles = batch
        elif len(batch) == 5:
            graph, prop, cmf, label, smiles = batch
        else:
            raise ValueError(f"Unexpected batch with {len(batch)} fields")
        graph = graph.to(self.device)
        prop = prop.to(self.device)
        cmf = cmf.to(self.device)
        if not isinstance(label, dict) or self.task_name not in label:
            raise ValueError("Expected a single-task label dictionary")
        label = {self.task_name: label[self.task_name].to(self.device)}
        return graph, prop, cmf, label, list(smiles)

    def _build_criterion(self):
        if str(self.train_cfg.get("loss_type", "asl")).lower() != "asl":
            raise ValueError("Simplified MAPViS-C supports loss_type=asl only")
        return AsymmetricLoss(
            gamma_pos=self.train_cfg.get("asl_gamma_pos", 0.0),
            gamma_neg=self.train_cfg.get("asl_gamma_neg", 2.0),
            clip=self.train_cfg.get("asl_clip", 0.05),
        )

    def _optimizer_group_name(self, name):
        if name.startswith("encoder_2d."):
            return "graph"
        if name.startswith(("desc_encoder.", "desc_proj.")):
            return "descriptor"
        if name.startswith("joint_encoder."):
            return "joint"
        if name.startswith(("cmf_encoder.encoders.", "fp_proj.")):
            return "fingerprint_encoders"
        if name.startswith("cmf_encoder.gate."):
            return "fingerprint_gate"
        if name.startswith("adaptive_gates."):
            return "global_gate"
        if name.startswith(
            ("modality_logit_heads.", "fingerprint_logit_heads.")
        ):
            return "prediction_heads"
        return "default"

    def _build_optimizer(self):
        learning_rates = self.train_cfg.get("learning_rates", {})
        base_lr = float(self.train_cfg.get("lr", 1e-4))
        weight_decay = self.train_cfg.get("weight_decay", {})
        if not isinstance(weight_decay, dict):
            weight_decay = {"default": float(weight_decay)}
        default_decay = float(weight_decay.get("default", 1e-4))
        grouped = {}
        for name, parameter in self.model.named_parameters():
            grouped.setdefault(
                self._optimizer_group_name(name),
                [],
            ).append(parameter)

        groups = []
        order = (
            "graph",
            "descriptor",
            "joint",
            "prediction_heads",
            "fingerprint_encoders",
            "fingerprint_gate",
            "global_gate",
            "default",
        )
        for name in order:
            parameters = grouped.get(name, [])
            if not parameters:
                continue
            decay_key = (
                "fingerprint"
                if name == "fingerprint_encoders"
                else "gates"
                if name in {"fingerprint_gate", "global_gate"}
                else "default"
            )
            groups.append(
                {
                    "name": name,
                    "params": parameters,
                    "lr": float(learning_rates.get(name, base_lr)),
                    "weight_decay": float(
                        weight_decay.get(decay_key, default_decay)
                    ),
                }
            )
        optimizer = torch.optim.AdamW(groups)
        for group in optimizer.param_groups:
            count = sum(item.numel() for item in group["params"])
            print(
                f"Optimizer | {group['name']} | params={count} | "
                f"lr={group['lr']:.6g} | wd={group['weight_decay']:.6g}",
                flush=True,
            )
        return optimizer

    def _build_scheduler(self):
        if self.train_cfg.get("scheduler", "reduce_on_plateau") != "reduce_on_plateau":
            raise ValueError(
                "Simplified MAPViS-C supports reduce_on_plateau only"
            )
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=float(self.train_cfg.get("scheduler_factor", 0.5)),
            patience=int(self.train_cfg.get("scheduler_patience", 5)),
            min_lr=float(self.train_cfg.get("scheduler_min_lr", 1e-6)),
        )

    def _set_global_gate_trainable(self, epoch):
        enabled = bool(self.train_cfg.get("global_gate_trainable", True))
        warmup = int(self.train_cfg.get("gate_warmup_epochs", 0))
        trainable = enabled and int(epoch) > warmup
        for parameter in self.model.adaptive_gates.parameters():
            parameter.requires_grad_(trainable)

    def _modality_aux_loss(self, aux, labels, valid_mask):
        task_logits = aux["modality_logits"][self.task_name]
        keep_masks = aux["modality_keep_masks"]
        total = labels.new_tensor(0.0)
        for name in ("graph", "fp", "desc", "joint"):
            weight = self.modality_aux_weights[name]
            if weight == 0.0:
                continue
            sample_mask = valid_mask & (keep_masks[name].view(-1) > 0.0)
            if sample_mask.any():
                total = total + weight * self.criterion(
                    task_logits[name].view(-1)[sample_mask],
                    labels[sample_mask],
                )

        if self.fingerprint_expert_aux_weight > 0.0:
            sample_mask = valid_mask & (
                keep_masks["fp"].view(-1) > 0.0
            )
            if sample_mask.any():
                losses = [
                    self.criterion(
                        value.view(-1)[sample_mask],
                        labels[sample_mask],
                    )
                    for key, value in task_logits.items()
                    if key.startswith("fp:")
                ]
                if losses:
                    reduction = self.train_cfg.get(
                        "fingerprint_expert_aux_reduction",
                        "sum",
                    )
                    stacked = torch.stack(losses)
                    reduced = (
                        stacked.sum()
                        if reduction == "sum"
                        else stacked.mean()
                    )
                    total = (
                        total
                        + self.fingerprint_expert_aux_weight * reduced
                    )
        return total

    def _step(self, batch):
        graph, prop, cmf, label_dict, smiles = self._move_batch(batch)
        logits_dict, aux = self.model(graph, prop, cmf)
        logits = logits_dict[self.task_name].view(-1)
        labels = label_dict[self.task_name].float().view(-1)
        valid_mask = labels != -1
        if not valid_mask.any():
            loss = logits.sum() * 0.0
        else:
            loss = self.criterion(
                logits[valid_mask],
                labels[valid_mask],
            )
            loss = loss + self._modality_aux_loss(
                aux,
                labels,
                valid_mask,
            )
            loss = loss + float(
                self.train_cfg.get("lambda_global_gate_reg", 0.0)
            ) * aux["global_gate_reg"]
            loss = loss + float(
                self.train_cfg.get("lambda_fingerprint_gate_reg", 0.0)
            ) * aux["fingerprint_gate_reg"]
        return loss, logits, labels, valid_mask, aux, smiles

    def _gradients_are_finite(self):
        return all(
            parameter.grad is None
            or torch.isfinite(parameter.grad).all().item()
            for parameter in self.model.parameters()
        )

    def _evaluate(self, loader, collect_predictions=False):
        self.model.eval()
        logits_batches = []
        label_batches = []
        expert_batches = {
            name: {"logits": [], "labels": []}
            for name in ("graph", "fp", "desc", "joint")
        }
        global_gate_batches = []
        fingerprint_gate_batches = []
        prediction_rows = []
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in loader:
                loss, logits, labels, valid_mask, aux, smiles = self._step(
                    batch
                )
                valid_logits = logits[valid_mask].detach().cpu()
                valid_labels = labels[valid_mask].detach().cpu()
                logits_batches.append(valid_logits)
                label_batches.append(valid_labels)
                count = int(valid_mask.sum().item())
                total_loss += float(loss.item()) * count
                total_samples += count

                task_logits = aux["modality_logits"][self.task_name]
                for name in expert_batches:
                    expert_batches[name]["logits"].append(
                        task_logits[name]
                        .view(-1)[valid_mask]
                        .detach()
                        .cpu()
                    )
                    expert_batches[name]["labels"].append(valid_labels)
                global_gate_batches.append(
                    aux["adaptive_weights"][self.task_name]
                    .detach()
                    .cpu()
                )
                fingerprint_gate_batches.append(
                    aux["fingerprint_weights"].detach().cpu()
                )

                if collect_predictions:
                    probabilities = torch.sigmoid(logits).detach().cpu()
                    labels_cpu = labels.detach().cpu()
                    valid_cpu = valid_mask.detach().cpu()
                    for index, smile in enumerate(smiles):
                        if not bool(valid_cpu[index]):
                            continue
                        prediction_rows.append(
                            {
                                "SMILES": smile,
                                "label": float(labels_cpu[index]),
                                "logit": float(
                                    logits.detach().cpu()[index]
                                ),
                                "probability": float(
                                    probabilities[index]
                                ),
                            }
                        )

        all_logits = torch.cat(logits_batches)
        all_labels = torch.cat(label_batches)
        metrics = compute_metrics(all_logits, all_labels)
        metrics["loss"] = total_loss / max(total_samples, 1)
        for name, batches in expert_batches.items():
            expert_metrics = compute_metrics(
                torch.cat(batches["logits"]),
                torch.cat(batches["labels"]),
            )
            metrics[f"{name}_auprc"] = expert_metrics["AUPRC"]

        global_weights = torch.cat(global_gate_batches).mean(dim=0)
        fingerprint_weights = torch.cat(
            fingerprint_gate_batches
        ).mean(dim=0)
        gate_payload = {
            "global": {
                name: float(value)
                for name, value in zip(
                    self.model.gate_modalities(),
                    global_weights,
                )
            },
            "fingerprint": {
                name: float(value)
                for name, value in zip(
                    self.model.fingerprint_gate_names(),
                    fingerprint_weights,
                )
            },
        }
        return metrics, prediction_rows, gate_payload

    def train(self):
        split_seed = int(self.config["dataset"]["seed"])
        train_loader, valid_loader, test_loader = (
            self.dataset.get_train_validation_data_loaders(
                seed=split_seed,
                num_workers=int(
                    self.config["dataset"].get("num_workers", 0)
                ),
                pin_memory=self.device.type == "cuda",
                split_report_dir=self.save_dir,
                primary_task=self.task_name,
            )
        )

        model_seed = int(self.train_cfg["model_seed"])
        set_seed(model_seed)
        self.model = MultimodalModel(self.config).to(self.device)
        structure = self.model.structure_summary()
        print(
            "Model structure | "
            f"modalities={structure['active_modalities']} | "
            f"experts={structure['active_experts']} | "
            f"joint_dim={structure['joint_input_dimension']} | "
            f"gate_dim={structure['global_gate_dimension']} | "
            f"trainable_params={structure['trainable_parameter_count']}",
            flush=True,
        )
        with open(
            os.path.join(self.save_dir, "model_structure.yaml"),
            "w",
            encoding="utf-8",
        ) as handle:
            yaml.safe_dump(structure, handle, sort_keys=False)

        self.criterion = self._build_criterion()
        self._set_global_gate_trainable(epoch=1)
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

        epochs = int(self.train_cfg["epochs"])
        patience = int(self.train_cfg["early_stopping_patience"])
        early_start = int(
            self.train_cfg.get("early_stopping_start_epoch", 1)
        )
        grad_clip = float(self.train_cfg.get("grad_clip_norm", 1.0))
        best_path = os.path.join(self.save_dir, "best.pt")
        patience_counter = 0
        history = []
        training_start = time.perf_counter()

        for epoch in range(1, epochs + 1):
            self.current_epoch = epoch
            self._set_global_gate_trainable(epoch)
            self.model.train()
            loss_sum = 0.0
            sample_count = 0
            skipped_steps = 0
            for batch in tqdm(
                train_loader,
                desc=f"Epoch {epoch}/{epochs}",
                leave=False,
            ):
                self.optimizer.zero_grad(set_to_none=True)
                loss, _, labels, valid_mask, _, _ = self._step(batch)
                if not torch.isfinite(loss).item():
                    skipped_steps += 1
                    continue
                loss.backward()
                if not self._gradients_are_finite():
                    skipped_steps += 1
                    self.optimizer.zero_grad(set_to_none=True)
                    continue
                norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    grad_clip,
                )
                if not torch.isfinite(norm).item():
                    skipped_steps += 1
                    self.optimizer.zero_grad(set_to_none=True)
                    continue
                self.optimizer.step()
                count = int(valid_mask.sum().item())
                loss_sum += float(loss.item()) * count
                sample_count += count

            train_loss = loss_sum / max(sample_count, 1)
            valid_metrics, _, valid_gates = self._evaluate(valid_loader)
            current_auprc = float(valid_metrics["AUPRC"])
            self.scheduler.step(current_auprc)
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "skipped_steps": skipped_steps,
                    **{
                        f"val_{key}": value
                        for key, value in valid_metrics.items()
                    },
                }
            )
            print(
                f"Epoch {epoch:3d} | train_loss={train_loss:.4f} | "
                f"val_loss={valid_metrics['loss']:.4f} | "
                f"val_AUC={valid_metrics['AUC']:.4f} | "
                f"val_AUPRC={current_auprc:.4f}",
                flush=True,
            )

            valid_candidate = (
                math.isfinite(current_auprc)
                and math.isfinite(valid_metrics["loss"])
                and skipped_steps == 0
            )
            if valid_candidate and current_auprc > self.best_val_auprc:
                self.best_epoch = epoch
                self.best_val_auprc = current_auprc
                self.best_val_auc = float(valid_metrics["AUC"])
                self.best_checkpoint_metric = current_auprc
                patience_counter = 0
                torch.save(self.model.state_dict(), best_path)
                with open(
                    os.path.join(self.save_dir, "valid_gate_weights.yaml"),
                    "w",
                    encoding="utf-8",
                ) as handle:
                    yaml.safe_dump(valid_gates, handle, sort_keys=False)
            elif epoch >= early_start:
                patience_counter += 1
                if patience_counter >= patience:
                    print(
                        f"Early stopping | epoch={epoch} | "
                        f"best_epoch={self.best_epoch}",
                        flush=True,
                    )
                    break

        self.training_time_seconds = time.perf_counter() - training_start
        pd.DataFrame(history).to_csv(
            os.path.join(self.save_dir, "history.csv"),
            index=False,
        )
        if not os.path.exists(best_path):
            raise RuntimeError("No valid checkpoint was produced")
        self.model.load_state_dict(
            torch.load(best_path, map_location=self.device)
        )
        self.test_metrics, predictions, test_gates = self._evaluate(
            test_loader,
            collect_predictions=True,
        )
        with open(
            os.path.join(self.save_dir, "metrics_test.yaml"),
            "w",
            encoding="utf-8",
        ) as handle:
            yaml.safe_dump(
                to_builtin(self.test_metrics),
                handle,
                sort_keys=False,
            )
        pd.DataFrame(predictions).to_csv(
            os.path.join(self.save_dir, "test_predictions.csv"),
            index=False,
        )
        with open(
            os.path.join(self.save_dir, "test_gate_weights.yaml"),
            "w",
            encoding="utf-8",
        ) as handle:
            yaml.safe_dump(test_gates, handle, sort_keys=False)
        run_summary = {
            "best_epoch": self.best_epoch,
            "best_val_auc": self.best_val_auc,
            "best_val_auprc": self.best_val_auprc,
            "checkpoint_metric": "valid_auprc",
            "checkpoint_selection": "best",
            "training_time_seconds": self.training_time_seconds,
            "trainable_parameter_count": sum(
                parameter.numel()
                for parameter in self.model.parameters()
                if parameter.requires_grad
            ),
            "test": self.test_metrics,
        }
        with open(
            os.path.join(self.save_dir, "run_summary.yaml"),
            "w",
            encoding="utf-8",
        ) as handle:
            yaml.safe_dump(
                to_builtin(run_summary),
                handle,
                sort_keys=False,
            )
        return run_summary


def append_aggregate_row(path, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    frame = pd.DataFrame([row])
    frame.to_csv(
        path,
        mode="a",
        header=not os.path.exists(path),
        index=False,
    )


def run_experiment(config, dataset):
    num_runs = int(config["training"].get("num_runs", 3))
    base_split_seed = int(config["dataset"].get("seed", 42))
    base_model_seed = int(
        config["training"].get("model_seed", base_split_seed)
    )
    configured_save_root = config["training"].get("save_dir")
    records = []
    for run_index in range(num_runs):
        run_config = copy.deepcopy(config)
        split_seed = base_split_seed + run_index
        model_seed = base_model_seed + run_index
        run_config["dataset"]["seed"] = split_seed
        run_config["training"]["model_seed"] = model_seed
        run_config["training"]["save_dir"] = (
            os.path.join(str(configured_save_root), f"seed{split_seed}")
            if configured_save_root and num_runs > 1
            else configured_save_root
        )
        run_config["run_idx"] = run_index + 1
        print(
            f"Run {run_index + 1}/{num_runs} | "
            f"split_seed={split_seed} | model_seed={model_seed}",
            flush=True,
        )
        trainer = Trainer(dataset, run_config)
        summary = trainer.train()
        metrics = summary["test"]
        row = {
            "task_name": config["task_name"],
            "split": config["dataset"]["splitting"],
            "run": run_index + 1,
            "split_seed": split_seed,
            "model_seed": model_seed,
            "best_epoch": summary["best_epoch"],
            "best_val_AUPRC": summary["best_val_auprc"],
            "AUC": metrics["AUC"],
            "AUPRC": metrics["AUPRC"],
            "BEDROC20": metrics["BEDROC20"],
            "EF@1%": metrics["EF@1%"],
            "Precision@50": metrics["Precision@50"],
            "Precision@100": metrics["Precision@100"],
            "save_dir": trainer.save_dir,
        }
        records.append(row)
        print(
            f"Run {run_index + 1}/{num_runs} result | "
            f"AUC={float(metrics['AUC']):.10f} | "
            f"AUPRC={float(metrics['AUPRC']):.10f}",
            flush=True,
        )
        append_aggregate_row(
            "experiment_records/MAPViS_C_runs.csv",
            row,
        )

    run_results = [
        {
            "run": int(record["run"]),
            "split_seed": int(record["split_seed"]),
            "model_seed": int(record["model_seed"]),
            "AUC": float(record["AUC"]),
            "AUPRC": float(record["AUPRC"]),
        }
        for record in records
    ]
    auprc_mean, auprc_std = finite_mean_std(
        [record["AUPRC"] for record in records]
    )
    auc_mean, auc_std = finite_mean_std(
        [record["AUC"] for record in records]
    )
    grouped = {
        "task_name": config["task_name"],
        "split": config["dataset"]["splitting"],
        "runs": num_runs,
        "run_results": run_results,
        "AUPRC_mean": auprc_mean,
        "AUPRC_std": auprc_std,
        "AUC_mean": auc_mean,
        "AUC_std": auc_std,
    }
    append_aggregate_row(
        "experiment_records/MAPViS_C_grouped.csv",
        {
            key: value
            for key, value in grouped.items()
            if key != "run_results"
        },
    )
    return grouped


def validate_config(config):
    if config["task_name"] not in {
        "GNEtolC",
        "HepG2",
        "HSkMC",
        "IMR90",
    }:
        raise ValueError(f"Unsupported task: {config['task_name']}")
    dataset_cfg = config["dataset"]
    target = dataset_cfg["target"]
    dataset_cfg["task"] = "classification"
    dataset_cfg["primary_task"] = target
    if dataset_cfg["splitting"] not in {
        "stratified",
        "stratified_scaffold",
        "fixed_indices",
    }:
        raise ValueError(
            "Only stratified, stratified_scaffold, and fixed_indices are supported"
        )


def main(config):
    validate_config(config)
    enable_tf32()
    print("Pre-building/loading dataset cache...", flush=True)
    dataset = Dataset(config)
    return run_experiment(config, dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--num-runs",
        type=int,
        help="Override training.num_runs from the YAML config.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        help="Override dataset.num_workers from the YAML config.",
    )
    parser.add_argument(
        "--valid-size",
        type=float,
        help="Override dataset.valid_size from the YAML config.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        help="Override dataset.test_size from the YAML config.",
    )
    arguments = parser.parse_args()
    with open(arguments.config, "r", encoding="utf-8") as handle:
        runtime_config = yaml.safe_load(handle)
    if arguments.num_runs is not None:
        if arguments.num_runs < 1:
            parser.error("--num-runs must be at least 1")
        runtime_config.setdefault("training", {})["num_runs"] = arguments.num_runs
    if arguments.num_workers is not None:
        if arguments.num_workers < 0:
            parser.error("--num-workers must be non-negative")
        runtime_config.setdefault("dataset", {})["num_workers"] = arguments.num_workers
    if arguments.valid_size is not None:
        runtime_config.setdefault("dataset", {})["valid_size"] = arguments.valid_size
    if arguments.test_size is not None:
        runtime_config.setdefault("dataset", {})["test_size"] = arguments.test_size
    valid_size = float(runtime_config["dataset"]["valid_size"])
    test_size = float(runtime_config["dataset"]["test_size"])
    if valid_size < 0.0 or test_size < 0.0:
        parser.error("--valid-size and --test-size must be non-negative")
    if valid_size + test_size >= 1.0:
        parser.error("--valid-size + --test-size must be less than 1")
    result = main(runtime_config)
    print(yaml.safe_dump(result, sort_keys=False), flush=True)

from __future__ import annotations

import csv
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.evaluation.metrics import compute_classification_metrics
from src.utils.io import append_csv_row, ensure_dir


def get_amp_settings(training_cfg: Dict, device: torch.device) -> tuple[bool, torch.dtype]:
    use_amp = bool(training_cfg.get("use_amp", False) and device.type == "cuda")
    amp_dtype_name = training_cfg.get("amp_dtype", "bfloat16")
    amp_dtype = torch.bfloat16 if amp_dtype_name == "bfloat16" else torch.float16
    return use_amp, amp_dtype


def create_scheduler(optimizer: torch.optim.Optimizer, training_cfg: Dict):
    scheduler_type = training_cfg.get("scheduler_type", "none")
    if scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=training_cfg["epochs"],
            eta_min=training_cfg.get("scheduler_min_lr", 0.0),
        )
    if scheduler_type == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=training_cfg.get("scheduler_factor", 0.5),
            patience=training_cfg.get("scheduler_patience", 2),
            min_lr=training_cfg.get("scheduler_min_lr", 0.0),
        )
    return None


def run_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    grad_clip: float,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    scaler: torch.amp.GradScaler | None,
) -> Tuple[float, Dict[str, float]]:
    is_train = optimizer is not None
    model.train(is_train)

    running_loss = 0.0
    all_predictions = []
    all_labels = []
    for batch in tqdm(data_loader, leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            autocast_context = (
                torch.autocast(device_type=device.type, dtype=amp_dtype)
                if amp_enabled
                else nullcontext()
            )
            with autocast_context:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs["logits"]
                loss = criterion(logits, labels)
            if is_train:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()

        running_loss += loss.item() * input_ids.size(0)
        predictions = torch.argmax(logits, dim=-1)
        all_predictions.append(predictions.detach().cpu())
        all_labels.append(labels.detach().cpu())

    predictions = torch.cat(all_predictions)
    labels = torch.cat(all_labels)
    metrics = compute_classification_metrics(predictions=predictions, labels=labels)
    avg_loss = running_loss / len(data_loader.dataset)
    return avg_loss, metrics


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict,
    output_dir: str | Path,
    device: torch.device,
) -> Dict:
    training_cfg = config["training"]
    output_dir = Path(output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    log_dir = output_dir / "logs"
    ensure_dir(checkpoint_dir)
    ensure_dir(log_dir)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_cfg["learning_rate"],
        weight_decay=training_cfg["weight_decay"],
    )
    scheduler = create_scheduler(optimizer, training_cfg)
    criterion = nn.CrossEntropyLoss()
    amp_enabled, amp_dtype = get_amp_settings(training_cfg, device)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and amp_dtype == torch.float16)
    history_path = log_dir / f"{config['experiment_name']}_history.csv"
    if not history_path.exists():
        with history_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "epoch",
                    "train_loss",
                    "train_accuracy",
                    "train_failure_rate",
                    "val_loss",
                    "val_accuracy",
                    "val_failure_rate",
                    "learning_rate",
                ],
            )
            writer.writeheader()

    best_val_acc = -1.0
    best_checkpoint_path = checkpoint_dir / f"{config['experiment_name']}_best.pt"

    for epoch in range(1, training_cfg["epochs"] + 1):
        train_loss, train_metrics = run_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            grad_clip=training_cfg["grad_clip"],
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            scaler=scaler,
        )
        val_loss, val_metrics = run_epoch(
            model=model,
            data_loader=val_loader,
            criterion=criterion,
            optimizer=None,
            device=device,
            grad_clip=training_cfg["grad_clip"],
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            scaler=None,
        )

        if scheduler is not None:
            if training_cfg.get("scheduler_type", "none") == "plateau":
                scheduler.step(val_metrics["accuracy"])
            else:
                scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_accuracy": round(train_metrics["accuracy"], 6),
            "train_failure_rate": round(train_metrics["failure_rate"], 6),
            "val_loss": round(val_loss, 6),
            "val_accuracy": round(val_metrics["accuracy"], 6),
            "val_failure_rate": round(val_metrics["failure_rate"], 6),
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        append_csv_row(history_path, row)

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "val_metrics": val_metrics,
                    "epoch": epoch,
                },
                best_checkpoint_path,
            )

    return {
        "best_checkpoint": str(best_checkpoint_path),
        "best_val_accuracy": best_val_acc,
        "history_path": str(history_path),
    }

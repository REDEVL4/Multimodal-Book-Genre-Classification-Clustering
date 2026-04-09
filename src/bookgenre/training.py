from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


@dataclass
class TrainingConfig:
    epochs: int = 5
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    device: str = "cpu"
    label_smoothing: float = 0.0
    use_amp: bool = False
    early_stopping_patience: int = 2
    early_stopping_min_delta: float = 0.0
    lr_scheduler_patience: int = 1
    lr_scheduler_factor: float = 0.5
    lr_scheduler_min_lr: float = 1e-6
    resume_if_possible: bool = True


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    config: TrainingConfig,
    output_dir: str | Path | None = None,
    run_signature: dict[str, object] | None = None,
) -> dict[str, object]:
    device = torch.device(config.device)
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.lr_scheduler_factor,
        patience=config.lr_scheduler_patience,
        min_lr=config.lr_scheduler_min_lr,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=config.use_amp and device.type == "cuda")

    output_path = Path(output_dir) if output_dir is not None else None
    state_path = output_path / "training_state.pt" if output_path is not None else None
    summary_path = output_path / "training_summary.json" if output_path is not None else None
    best_model_path = output_path / "best_model.pt" if output_path is not None else None

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "validation_loss": [],
        "validation_accuracy": [],
        "learning_rate": [],
    }
    best_state: dict[str, torch.Tensor] | None = None
    best_validation_loss = float("inf")
    best_validation_accuracy: float | None = None
    best_epoch = 0
    start_epoch = 0
    resumed_from_epoch = 0
    completed_epochs = 0
    no_improvement_epochs = 0
    stopped_early = False
    resumed = False
    training_start_time = time.perf_counter()

    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)

    if config.resume_if_possible and state_path is not None and state_path.exists():
        state = torch.load(state_path, map_location=device)
        if state.get("run_signature") == run_signature:
            resumed = True
            start_epoch = int(state.get("completed_epochs", 0))
            resumed_from_epoch = start_epoch
            completed_epochs = start_epoch
            history = state.get("history", history)
            best_validation_loss = float(state.get("best_validation_loss", best_validation_loss))
            best_validation_accuracy = state.get("best_validation_accuracy", best_validation_accuracy)
            best_epoch = int(state.get("best_epoch", 0))
            no_improvement_epochs = int(state.get("no_improvement_epochs", 0))

            model.load_state_dict(state["model_state_dict"])
            optimizer.load_state_dict(state["optimizer_state_dict"])
            scheduler.load_state_dict(state["scheduler_state_dict"])
            scaler_state = state.get("scaler_state_dict")
            if scaler_state:
                scaler.load_state_dict(scaler_state)

            best_state = state.get("best_model_state_dict")
            if start_epoch >= config.epochs:
                if best_state is not None:
                    model.load_state_dict(best_state)
                summary = _build_summary(
                    history=history,
                    best_validation_loss=best_validation_loss,
                    best_validation_accuracy=best_validation_accuracy,
                    best_epoch=best_epoch,
                    completed_epochs=completed_epochs,
                    requested_epochs=config.epochs,
                    resumed=resumed,
                    resumed_from_epoch=resumed_from_epoch,
                    stopped_early=state.get("stopped_early", False),
                    total_elapsed_seconds=state.get("total_elapsed_seconds", 0.0),
                    config=config,
                )
                if summary_path is not None:
                    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
                print(
                    f"Requested {config.epochs} epochs already completed in {output_path}. "
                    f"Loaded best model from epoch {best_epoch}."
                )
                return summary

            print(f"Resuming training from epoch {start_epoch + 1}/{config.epochs}")
        else:
            print("Existing training state found, but the training signature changed. Starting a fresh run.")

    for epoch in range(start_epoch, config.epochs):
        epoch_start_time = time.perf_counter()
        train_metrics = _run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scaler,
            training=True,
            use_amp=config.use_amp,
            epoch_index=epoch,
            total_epochs=config.epochs,
        )
        validation_metrics = _run_epoch(
            model,
            validation_loader,
            criterion,
            optimizer,
            device,
            scaler,
            training=False,
            use_amp=config.use_amp,
            epoch_index=epoch,
            total_epochs=config.epochs,
        )
        scheduler.step(validation_metrics["loss"])

        current_lr = float(optimizer.param_groups[0]["lr"])
        epoch_elapsed = time.perf_counter() - epoch_start_time
        epoch_samples = train_metrics["examples"] + validation_metrics["examples"]
        epoch_samples_per_second = epoch_samples / epoch_elapsed if epoch_elapsed > 0 else 0.0

        history["train_loss"].append(train_metrics["loss"])
        history["train_accuracy"].append(train_metrics["accuracy"])
        history["validation_loss"].append(validation_metrics["loss"])
        history["validation_accuracy"].append(validation_metrics["accuracy"])
        history["learning_rate"].append(current_lr)

        completed_epochs = epoch + 1

        improved = validation_metrics["loss"] < (best_validation_loss - config.early_stopping_min_delta)
        if improved:
            best_validation_loss = validation_metrics["loss"]
            best_validation_accuracy = validation_metrics["accuracy"]
            best_epoch = completed_epochs
            no_improvement_epochs = 0
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            if best_model_path is not None:
                torch.save(best_state, best_model_path)
        else:
            no_improvement_epochs += 1

        print(
            f"Epoch {completed_epochs}/{config.epochs} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"val_loss={validation_metrics['loss']:.4f} "
            f"val_acc={validation_metrics['accuracy']:.4f} "
            f"best_epoch={best_epoch} "
            f"lr={current_lr:.6f} "
            f"elapsed={epoch_elapsed:.1f}s "
            f"speed={epoch_samples_per_second:.1f} samples/s"
        )

        if state_path is not None:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_model_state_dict": best_state,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "history": history,
                    "completed_epochs": completed_epochs,
                    "best_validation_loss": best_validation_loss,
                    "best_validation_accuracy": best_validation_accuracy,
                    "best_epoch": best_epoch,
                    "no_improvement_epochs": no_improvement_epochs,
                    "stopped_early": stopped_early,
                    "total_elapsed_seconds": time.perf_counter() - training_start_time,
                    "training_config": asdict(config),
                    "run_signature": run_signature,
                },
                state_path,
            )

        summary = _build_summary(
            history=history,
            best_validation_loss=best_validation_loss,
            best_validation_accuracy=best_validation_accuracy,
            best_epoch=best_epoch,
            completed_epochs=completed_epochs,
            requested_epochs=config.epochs,
            resumed=resumed,
            resumed_from_epoch=resumed_from_epoch,
            stopped_early=stopped_early,
            total_elapsed_seconds=time.perf_counter() - training_start_time,
            config=config,
        )
        if summary_path is not None:
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        if no_improvement_epochs >= config.early_stopping_patience:
            stopped_early = True
            print(
                f"Early stopping triggered after epoch {completed_epochs}. "
                f"Best epoch was {best_epoch} with val_loss={best_validation_loss:.4f}."
            )
            break

    if best_model_path is not None and best_model_path.exists():
        best_state = torch.load(best_model_path, map_location=device)
    if best_state is not None:
        model.load_state_dict(best_state)

    summary = _build_summary(
        history=history,
        best_validation_loss=best_validation_loss,
        best_validation_accuracy=best_validation_accuracy,
        best_epoch=best_epoch,
        completed_epochs=completed_epochs,
        requested_epochs=config.epochs,
        resumed=resumed,
        resumed_from_epoch=resumed_from_epoch,
        stopped_early=stopped_early,
        total_elapsed_seconds=time.perf_counter() - training_start_time,
        config=config,
    )

    if summary_path is not None:
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if state_path is not None and state_path.exists():
        final_state = torch.load(state_path, map_location="cpu")
        final_state["stopped_early"] = stopped_early
        final_state["total_elapsed_seconds"] = summary["total_elapsed_seconds"]
        torch.save(final_state, state_path)

    return summary


def _build_summary(
    history: dict[str, list[float]],
    best_validation_loss: float,
    best_validation_accuracy: float | None,
    best_epoch: int,
    completed_epochs: int,
    requested_epochs: int,
    resumed: bool,
    resumed_from_epoch: int,
    stopped_early: bool,
    total_elapsed_seconds: float,
    config: TrainingConfig,
) -> dict[str, object]:
    return {
        "history": history,
        "best_validation_loss": best_validation_loss,
        "best_validation_accuracy": best_validation_accuracy,
        "best_epoch": best_epoch,
        "final_validation_accuracy": history["validation_accuracy"][-1] if history["validation_accuracy"] else None,
        "completed_epochs": completed_epochs,
        "requested_epochs": requested_epochs,
        "resumed": resumed,
        "resumed_from_epoch": resumed_from_epoch,
        "stopped_early": stopped_early,
        "total_elapsed_seconds": total_elapsed_seconds,
        "training_config": asdict(config),
    }


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    training: bool,
    use_amp: bool,
    epoch_index: int,
    total_epochs: int,
) -> dict[str, float]:
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    phase = "train" if training else "val"
    progress = tqdm(
        loader,
        total=len(loader),
        desc=f"Epoch {epoch_index + 1}/{total_epochs} [{phase}]",
        unit="batch",
        leave=True,
        dynamic_ncols=True,
        mininterval=0.5,
        smoothing=0.1,
        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
    )
    epoch_progress_start = time.perf_counter()

    for batch_index, batch in enumerate(progress, start=1):
        images = batch["image"].to(device)
        text_ids = batch["text_ids"].to(device)
        labels = batch["label"].to(device)

        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp and device.type == "cuda"):
                outputs = model(images=images, text_ids=text_ids)
                loss = criterion(outputs["logits"], labels)
            if training:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        predictions = outputs["logits"].argmax(dim=1)
        total_loss += loss.item() * labels.size(0)
        total_correct += (predictions == labels).sum().item()
        total_examples += labels.size(0)

        if total_examples > 0:
            elapsed = time.perf_counter() - epoch_progress_start
            samples_per_second = total_examples / elapsed if elapsed > 0 else 0.0
            batches_per_second = batch_index / elapsed if elapsed > 0 else 0.0
            progress.set_postfix(
                loss=f"{total_loss / total_examples:.4f}",
                acc=f"{total_correct / total_examples:.4f}",
                samples_s=f"{samples_per_second:.1f}",
                batches_s=f"{batches_per_second:.2f}",
            )

    if total_examples == 0:
        return {"loss": 0.0, "accuracy": 0.0, "examples": 0}

    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
        "examples": total_examples,
    }

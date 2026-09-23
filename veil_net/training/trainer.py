from __future__ import annotations

import csv
import hashlib
import random
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from veil_net.datasets.audit_filter import filter_paths
from veil_net.losses import CompletionLoss
from veil_net.models import CompletionConfig, CompletionNetwork
from veil_net.utils.experiment import git_commit, seed_everything
from veil_net.utils.io import ensure_dir, load_yaml_like, read_json, write_json


class NpzTorchDataset(Dataset):
    def __init__(
        self,
        pairs_dir: Path,
        max_samples: int | None = None,
        alignment_report: Path | None = None,
        sample_seed: int | None = None,
        sample_ids: list[str] | None = None,
    ) -> None:
        self.paths = sorted(pairs_dir.rglob("*.npz"))
        if sample_ids is not None:
            indexed = {p.stem: p for p in self.paths}
            if len(indexed) != len(self.paths):
                raise ValueError("Duplicate training sample IDs")
            if not sample_ids or len(sample_ids) != len(set(sample_ids)):
                raise ValueError("Training sample IDs must be nonempty and unique")
            missing = set(sample_ids) - indexed.keys()
            if missing:
                raise ValueError(
                    f"Missing {len(missing)} training pairs, e.g. {sorted(missing)[0]}"
                )
            self.paths = [indexed[sid] for sid in sorted(sample_ids)]
        self.audit_summary = None
        if alignment_report is not None:
            self.paths, self.audit_summary = filter_paths(self.paths, alignment_report, "train")
            print(
                f"Train audit: {self.audit_summary['kept']}/{self.audit_summary['total']} retained; {len(self.audit_summary['excluded'])} excluded",
                flush=True,
            )
        if max_samples is not None:
            self.paths = (
                self.paths[:max_samples]
                if sample_seed is None
                else sorted(
                    random.Random(sample_seed).sample(self.paths, min(max_samples, len(self.paths)))
                )
            )
        if not self.paths:
            raise FileNotFoundError(f"No .npz completion pairs found in {pairs_dir}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        with np.load(self.paths[index], allow_pickle=False) as data:
            if "complete" not in data.files:
                raise ValueError(f"{self.paths[index]} lacks complete target")
            return {
                "partial": torch.from_numpy(data["partial"].astype(np.float32)),
                "complete": torch.from_numpy(data["complete"].astype(np.float32)),
                "risk_target": torch.from_numpy(data["risk_target"].astype(np.float32))
                if "risk_target" in data.files
                else torch.full((data["partial"].shape[0],), float("nan"), dtype=torch.float32),
            }


def check_training_cohort(sample_ids: list[str], validation_ids: list[str]) -> None:
    if not validation_ids or len(validation_ids) != len(set(validation_ids)):
        raise ValueError("Validation sample IDs must be nonempty and unique")
    if not sample_ids or len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Training sample IDs must be nonempty and unique")
    overlap = set(sample_ids) & set(validation_ids)
    if overlap:
        raise ValueError(
            f"Training/validation overlap: {len(overlap)} IDs, e.g. {sorted(overlap)[0]}"
        )


def create_model(model_name: str, output_points: int = 16384) -> torch.nn.Module:
    if model_name == "veil_net":
        return CompletionNetwork(CompletionConfig(output_points=output_points))
    raise RuntimeError(f"unknown model: {model_name}")


def config_for_variant(variant: str | None, output_points: int = 16384) -> CompletionConfig:
    key = (variant or "full").lower()
    switches = {
        "full": {},
        "a0": {
            "use_risk": False,
            "use_router": False,
            "use_preservation": False,
            "use_uncertainty": False,
        },
        "a1": {"use_risk": False},
        "a2": {"use_router": False},
        "a3": {"use_risk": False, "use_router": False},
        "a4": {"use_preservation": False},
        "a5": {"use_uncertainty": False},
        "a6": {"uniform_routing": True},
        "a7": {"oracle_risk": True},
    }
    if key not in switches:
        raise ValueError(f"unknown VEIL-Net ablation variant: {variant}")
    return CompletionConfig(output_points=output_points, **switches[key])


def create_completion_variant(variant: str | None, output_points: int = 16384) -> CompletionNetwork:
    return CompletionNetwork(config_for_variant(variant, output_points=output_points))


def train_completion(
    project_root: Path,
    model_name: str,
    device: str,
    seed: int,
    epochs: int,
    batch_size: int,
    max_samples: int | None,
    confirm: bool,
    resume: bool,
    output_points: int = 16384,
    run_name: str | None = None,
    variant: str | None = None,
    config_path: Path | None = None,
    num_workers: int = 0,
    pin_memory: bool = True,
    pairs_dir: Path | None = None,
    on_epoch_end: Callable[[Path, int], bool] | None = None,
) -> dict[str, Any]:
    if not confirm:
        return {
            "started_training": False,
            "reason": "confirmation flag was not supplied",
            "required_flag": "--confirm-full or --confirm-train",
        }
    seed_everything(seed)
    train_config = (
        load_yaml_like(config_path) if config_path is not None and config_path.exists() else {}
    )
    if train_config.get("require_training_manifest") and not train_config.get(
        "training_sample_ids"
    ):
        raise ValueError("This recipe requires a training manifest in training_sample_ids")
    pairs_dir = (
        pairs_dir
        if pairs_dir is not None
        else project_root / "cache" / "completion_pairs" / "train"
    )
    audit_path = (
        project_root / train_config["alignment_report"]
        if train_config.get("alignment_report")
        else None
    )
    dataset = NpzTorchDataset(
        pairs_dir,
        max_samples=max_samples,
        alignment_report=audit_path,
        sample_seed=train_config.get("sample_seed"),
        sample_ids=train_config.get("training_sample_ids"),
    )
    validation_ids = None
    if train_config.get("validation_protocol"):
        validation_ids = read_json(project_root / train_config["validation_protocol"])["sample_ids"]
        check_training_cohort([p.stem for p in dataset.paths], validation_ids)
        if train_config.get("training_sample_ids"):
            check_training_cohort(train_config["training_sample_ids"], validation_ids)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=max(0, int(num_workers)),
        pin_memory=pin_memory and device.startswith("cuda") and torch.cuda.is_available(),
        persistent_workers=bool(num_workers and num_workers > 0),
    )
    if model_name == "veil_net" and variant is not None:
        model = create_completion_variant(variant, output_points=output_points).to(device)
    else:
        model = create_model(model_name, output_points=output_points).to(device)
    if isinstance(model, CompletionNetwork):
        overrides = {
            key: train_config[key]
            for key in ("use_risk", "use_router", "use_preservation", "use_uncertainty")
            if key in train_config
        }
        if any(not isinstance(value, bool) for value in overrides.values()):
            raise ValueError("Module switches must be YAML booleans")
        if overrides:
            model.config = replace(model.config, **overrides)
    if isinstance(model, CompletionNetwork) and "bbox_margin" in train_config:
        model.config = replace(model.config, bbox_margin=float(train_config["bbox_margin"]))
    if isinstance(model, CompletionNetwork) and "uncertainty_moves_points" in train_config:
        model.config = replace(
            model.config, uncertainty_moves_points=train_config["uncertainty_moves_points"]
        )
    if isinstance(model, CompletionNetwork) and train_config.get("use_surface_refiner", False):
        model = CompletionNetwork(replace(model.config, use_surface_refiner=True)).to(device)
    if isinstance(model, CompletionNetwork) and train_config.get("use_spatial_transformer", False):
        model = CompletionNetwork(
            replace(
                model.config,
                use_spatial_transformer=True,
                input_frame_normalization=bool(
                    train_config.get("input_frame_normalization", False)
                ),
                two_stage_folding=bool(train_config.get("two_stage_folding", False)),
                robust_input_geometry=bool(train_config.get("robust_input_geometry", False)),
                patch_grid_residual=bool(train_config.get("patch_grid_residual", False)),
                geometry_queries=bool(train_config.get("geometry_queries", False)),
                query_position_conditioning=bool(
                    train_config.get("query_position_conditioning", True)
                ),
                spatial_tokens=int(train_config.get("spatial_tokens", 128)),
                spatial_patches=int(train_config.get("spatial_patches", 256)),
                sparse_input_guard=bool(train_config.get("sparse_input_guard", False)),
                sparse_unique_tokens=bool(train_config.get("sparse_unique_tokens", False)),
            )
        ).to(device)
    loss_config = (
        train_config.get("loss", {}) if isinstance(train_config.get("loss", {}), dict) else {}
    )
    loss_args = {
        k: float(v)
        for k, v in loss_config.items()
        if k.startswith("lambda_")
        or k
        in {
            "missing_threshold",
            "missing_on_completed",
            "relative_geometry_loss",
            "coverage_points",
            "coverage_threshold",
            "eval_fscore_points",
        }
    }
    loss_fn = CompletionLoss(**loss_args)
    learning_rate = float(train_config.get("learning_rate", 1e-4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=bool(train_config.get("amp", True))
        and device.startswith("cuda")
        and torch.cuda.is_available(),
    )
    storage_name = run_name or train_config.get("run_name") or model_name
    ckpt_dir = ensure_dir(project_root / "checkpoints" / storage_name / f"seed_{seed}")
    last_ckpt = ckpt_dir / "last.pt"
    if train_config.get("require_fresh_run") and not resume and any(ckpt_dir.glob("*.pt")):
        raise ValueError(
            f"{ckpt_dir} already contains checkpoints. Use --resume to continue; refusing to overwrite a run from epoch zero."
        )
    start_epoch = 0
    best = float("inf")
    init_checkpoint = train_config.get("init_checkpoint")
    if resume and not last_ckpt.is_file():
        raise FileNotFoundError(f"No checkpoint to resume: {last_ckpt}")
    source_ckpt = (
        last_ckpt
        if resume and last_ckpt.exists()
        else (project_root / init_checkpoint if init_checkpoint else None)
    )
    source_sha256 = None
    if source_ckpt is not None:
        if source_ckpt != last_ckpt and train_config.get("init_checkpoint_sha256"):
            source_sha256 = hashlib.sha256(source_ckpt.read_bytes()).hexdigest()
            if source_sha256 != train_config["init_checkpoint_sha256"]:
                raise ValueError(
                    "Warm-start checkpoint changed; refusing to train an unverified source"
                )
        state = torch.load(source_ckpt, map_location=device, weights_only=True)
        if validation_ids is not None:
            check_training_cohort(state.get("training_sample_ids", []), validation_ids)
        if isinstance(model, CompletionNetwork) and "bbox_margin" not in train_config:
            model.config = replace(
                model.config,
                bbox_margin=float(
                    state.get("model_config", {}).get("bbox_margin", model.config.bbox_margin)
                ),
            )
        warm_start = source_ckpt != last_ckpt
        if (
            warm_start
            and "init_checkpoint_epoch" in train_config
            and state["epoch"] != int(train_config["init_checkpoint_epoch"])
        ):
            raise ValueError("Unexpected warm-start epoch")
        if warm_start:
            report = model.load_state_dict(state["model"], strict=False)
            if report.unexpected_keys or any(
                not k.startswith("surface_refiner.") for k in report.missing_keys
            ):
                raise ValueError(f"Incompatible warm-start checkpoint: {report}")
            print(
                f"Warm start from {source_ckpt}; new parameters: {report.missing_keys}", flush=True
            )
        else:
            model.load_state_dict(state["model"])
        if isinstance(model, CompletionNetwork) and "uncertainty_moves_points" not in train_config:
            model.config = replace(
                model.config,
                uncertainty_moves_points=state.get("model_config", {}).get(
                    "uncertainty_moves_points", True
                ),
            )
        if not warm_start:
            optimizer.load_state_dict(state["optimizer"])
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        if not warm_start:
            scaler.load_state_dict(state["scaler"])
        start_epoch = int(state["epoch"]) + 1
        best = float(state["best"])
        if warm_start or state.get("loss_config") != loss_config:
            best = float("inf")
    if epochs <= start_epoch:
        raise ValueError(f"epochs={epochs} must exceed the starting epoch {start_epoch}")
    checkpoint_metadata = {
        "training_sample_ids": [p.stem for p in dataset.paths],
        "model_config": asdict(model.config) if isinstance(model, CompletionNetwork) else {},
        "loss_config": loss_config,
        "training_audit": dataset.audit_summary,
        "initialization": state.get("initialization")
        if source_ckpt == last_ckpt
        else {
            "path": str(source_ckpt) if source_ckpt else None,
            "sha256": source_sha256,
        },
    }
    print(
        f"Training epochs {start_epoch}..{epochs - 1}; lr={learning_rate}; model_config={checkpoint_metadata['model_config']}",
        flush=True,
    )
    history_path = (
        ensure_dir(project_root / "results" / "full" / "training" / storage_name / f"seed_{seed}")
        / "history.csv"
    )
    if dataset.audit_summary is not None:
        write_json(history_path.with_name("training_audit.json"), dataset.audit_summary)
    config_snapshot = {
        "model": model_name,
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "device": device,
        "variant": variant or "",
        "config_path": str(config_path or ""),
        "learning_rate": learning_rate,
        "amp": scaler.is_enabled(),
        "gradient_clip": train_config.get("gradient_clip"),
        "loss": loss_config,
        "training_config": train_config,
        "git_commit": git_commit(project_root),
    }
    write_json(history_path.with_suffix(".config.json"), config_snapshot)
    with history_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["epoch", "train_loss", "epoch_time_s", "peak_vram_mb"]
        )
        if f.tell() == 0:
            writer.writeheader()
        try:
            for epoch in range(start_epoch, epochs):
                model.train()
                started = time.time()
                total = 0.0
                component_totals: dict[str, float] = {}
                sample_count = 0
                optimizer_updates = skipped_updates = 0
                for step, batch in enumerate(loader, 1):
                    partial = batch["partial"].to(device)
                    complete = batch["complete"].to(device)
                    risk_target = batch["risk_target"].to(device)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                        raw_outputs = (
                            model(partial, risk_target)
                            if model_name == "veil_net"
                            else model(partial)
                        )
                        outputs = (
                            raw_outputs
                            if isinstance(raw_outputs, dict)
                            else {"completed_points": raw_outputs}
                        )
                        loss, components = loss_fn(
                            outputs,
                            {"partial": partial, "complete": complete, "risk_target": risk_target},
                        )
                    if not torch.isfinite(loss):
                        raise FloatingPointError("NaN/Inf loss detected")
                    scaler.scale(loss).backward()
                    if train_config.get("gradient_clip") is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            float(train_config["gradient_clip"]),
                            error_if_nonfinite=not scaler.is_enabled(),
                        )
                    previous_scale = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    if scaler.is_enabled() and scaler.get_scale() < previous_scale:
                        skipped_updates += 1
                    else:
                        optimizer_updates += 1
                    total += float(loss.detach().cpu())
                    sample_count += partial.shape[0]
                    for key, value in components.items():
                        component_totals[key] = (
                            component_totals.get(key, 0.0) + value * partial.shape[0]
                        )
                    log_every = int(train_config.get("log_every_steps", 0))
                    if log_every > 0 and (step % log_every == 0 or step == len(loader)):
                        print(
                            f"epoch={epoch} step={step}/{len(loader)} loss={float(loss.detach().cpu()):.6f}",
                            flush=True,
                        )
                mean_loss = total / max(len(loader), 1)
                best = min(best, mean_loss)
                peak = (
                    torch.cuda.max_memory_allocated() / (1024**2)
                    if device.startswith("cuda") and torch.cuda.is_available()
                    else 0.0
                )
                writer.writerow(
                    {
                        "epoch": epoch,
                        "train_loss": mean_loss,
                        "epoch_time_s": time.time() - started,
                        "peak_vram_mb": peak,
                    }
                )
                f.flush()
                # Keep the existing history schema; detailed terms have their own file.
                component_path = history_path.with_name("loss_components.csv")
                with component_path.open("a", newline="", encoding="utf-8") as detail:
                    component_writer = csv.DictWriter(
                        detail, fieldnames=["epoch", "term", "weighted_mean", "raw_mean", "weight"]
                    )
                    if detail.tell() == 0:
                        component_writer.writeheader()
                    for term, weight in loss_fn.weights.items():
                        if term in component_totals:
                            component_writer.writerow(
                                {
                                    "epoch": epoch,
                                    "term": term,
                                    "weighted_mean": component_totals[term] / sample_count,
                                    "raw_mean": component_totals[f"raw_{term}"] / sample_count,
                                    "weight": weight,
                                }
                            )
                raw_cd = component_totals.get("raw_cd", float("nan")) / sample_count
                raw_missing = component_totals.get("raw_missing", float("nan")) / sample_count
                distance_prefix = "relative" if loss_fn.relative_geometry_loss else "raw"
                print(
                    f"epoch={epoch} train_loss={mean_loss:.6f} {distance_prefix}_cd={raw_cd:.6f} {distance_prefix}_missing={raw_missing:.6f} optimizer_updates={optimizer_updates} skipped_updates={skipped_updates}",
                    flush=True,
                )
                with history_path.with_name("optimizer_updates.csv").open(
                    "a", newline=""
                ) as update_file:
                    update_writer = csv.DictWriter(
                        update_file,
                        fieldnames=["epoch", "optimizer_updates", "skipped_updates", "amp_scale"],
                    )
                    if update_file.tell() == 0:
                        update_writer.writeheader()
                    update_writer.writerow(
                        dict(
                            epoch=epoch,
                            optimizer_updates=optimizer_updates,
                            skipped_updates=skipped_updates,
                            amp_scale=scaler.get_scale(),
                        )
                    )
                torch.save(
                    {
                        **checkpoint_metadata,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scaler": scaler.state_dict(),
                        "epoch": epoch,
                        "best": best,
                    },
                    last_ckpt,
                )
                if mean_loss <= best:
                    torch.save(
                        {
                            **checkpoint_metadata,
                            "model": model.state_dict(),
                            "epoch": epoch,
                            "best": best,
                        },
                        ckpt_dir / "best.pt",
                    )
                if on_epoch_end is not None:
                    # Validation constructs a separate model; preserve training's CPU RNG.
                    with torch.random.fork_rng(devices=[]):
                        stop = on_epoch_end(last_ckpt, epoch)
                    if stop:
                        break
        except KeyboardInterrupt:
            torch.save(
                {
                    **checkpoint_metadata,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "epoch": epoch,
                    "best": best,
                },
                ckpt_dir / "interrupted.pt",
            )
            raise
    return {
        "started_training": True,
        "epochs_completed": epoch + 1 - start_epoch,
        "best_loss": best,
        "checkpoint": str(last_ckpt),
    }

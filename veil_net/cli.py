"""Public commands: smoke, infer, evaluate, train, and export."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from dataclasses import asdict
from importlib.resources import files
from pathlib import Path

import numpy as np
import torch

from veil_net import VEILNet, VEILNetConfig, __version__
from veil_net.evaluation import evaluate_pairs, select_pairs
from veil_net.inference import load_model, predict, read_cloud
from veil_net.model import sha256_file


def _positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def smoke(args: argparse.Namespace) -> dict:
    torch.manual_seed(0)
    config = VEILNetConfig(
        input_points=32, output_points=128, hidden_dim=32, spatial_tokens=8, spatial_patches=16
    )
    model = VEILNet(config).to(args.device).eval()
    partial = torch.randn(2, 32, 3, device=args.device)
    started = time.perf_counter()
    output = model(partial)
    if output["completed_points"].shape != (2, 128, 3):
        raise RuntimeError("Unexpected completion shape")
    if not all(torch.isfinite(value).all() for value in output.values()):
        raise RuntimeError("Nonfinite model output")
    output["generated_points"].square().mean().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not grads or not all(torch.isfinite(g).all() for g in grads):
        raise RuntimeError("Missing or nonfinite gradients")
    with tempfile.TemporaryDirectory(prefix="veil-smoke-") as tmp:
        model.save_pretrained(tmp)
        restored = VEILNet.from_pretrained(Path(tmp), device=args.device)
        with torch.inference_mode():
            torch.testing.assert_close(
                restored(partial)["completed_points"], output["completed_points"]
            )
    return {
        "status": "passed",
        "checks": ["forward", "backward", "safetensors_roundtrip"],
        "device": args.device,
        "seconds": round(time.perf_counter() - started, 3),
        "scope": "Reduced-size random-weight software check; not a quality benchmark.",
    }


def infer(args: argparse.Namespace) -> dict:
    if args.output.suffix.lower() != ".npz":
        raise ValueError("Inference output must end in .npz")
    if args.output.exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    model = load_model(
        args.model, args.device, revision=args.revision, local_files_only=args.local_files_only
    )
    result = predict(model, read_cloud(args.input, args.key), seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **result)
    return {
        "output": str(args.output),
        "input_points": len(result["input_partial"]),
        "output_points": len(result["prediction_complete"]),
    }


def export(args: argparse.Namespace) -> dict:
    model = VEILNet.from_checkpoint(args.checkpoint)
    if model.config.oracle_risk:
        raise ValueError("Oracle-risk ablations cannot be exported as an inference release")
    directory = model.save_pretrained(args.output)
    # Verify tensor equality without needing the original datasets.
    restored = VEILNet.from_pretrained(directory)
    for key, tensor in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], tensor, rtol=0, atol=0)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    metadata = {
        "package_version": __version__,
        "source_checkpoint": args.checkpoint.name,
        "source_sha256": sha256_file(args.checkpoint),
        "epoch_index": state.get("epoch"),
        "weights_sha256": sha256_file(directory / "model.safetensors"),
        "config_sha256": sha256_file(directory / "config.json"),
        "model_config": asdict(model.config),
        "tensor_roundtrip": "exact",
    }
    (directory / "provenance.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    card = files("veil_net").joinpath("model_card.md").read_text(encoding="utf-8")
    card += "\n## Export Provenance\n\n```json\n" + json.dumps(metadata, indent=2) + "\n```\n"
    (directory / "README.md").write_text(card, encoding="utf-8")
    license_path = Path(__file__).resolve().parents[1] / "LICENSE"
    if license_path.is_file():
        shutil.copy2(license_path, directory / "LICENSE")
    return {
        "output": str(directory),
        "tensor_roundtrip": "exact",
        "source_sha256": metadata["source_sha256"],
    }


def evaluate(args: argparse.Namespace) -> dict:
    protocol = json.loads(args.protocol.read_text(encoding="utf-8")) if args.protocol else {}
    if args.protocol and "sample_ids" not in protocol:
        raise ValueError("Protocol must contain sample_ids")
    seed = args.seed if args.seed is not None else protocol.get("seed", 0)
    paths = select_pairs(
        args.pairs, sample_ids=protocol.get("sample_ids"), limit=args.max_samples, seed=seed
    )
    points = args.eval_points or protocol.get("eval_points", 1024)
    model = load_model(
        args.model, args.device, revision=args.revision, local_files_only=args.local_files_only
    )
    provenance = {
        "model": args.model,
        "model_config": asdict(model.config),
        **model.artifact_metadata,
    }
    if args.protocol:
        provenance["protocol_sha256"] = sha256_file(args.protocol)
    report = evaluate_pairs(
        model,
        paths,
        args.output,
        device=args.device,
        eval_points=points,
        seed=seed,
        missing_threshold=protocol.get("missing_threshold", 0.01),
        provenance=provenance,
    )
    return {"output": str(args.output), "samples": report["samples"], "metrics": report["metrics"]}


def train(args: argparse.Namespace) -> dict:
    from rapc_net.training.trainer import train_completion
    from rapc_net.utils.io import load_yaml_like

    config = load_yaml_like(args.config)
    return train_completion(
        project_root=args.workspace.resolve(),
        model_name="rapc_net",
        device=args.device,
        seed=args.seed,
        epochs=args.epochs or int(config.get("epochs", 50)),
        batch_size=args.batch_size or int(config.get("batch_size", 2)),
        max_samples=args.max_samples or None,
        confirm=True,
        resume=args.resume,
        output_points=int(config.get("output_points", 16384)),
        run_name=config.get("run_name", "veil_net"),
        config_path=args.config.resolve(),
        pairs_dir=args.pairs.resolve(),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="veil", description="VEIL-Net surface completion")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text, handler in [
        ("smoke", "Check forward/backward and portable weights without datasets", smoke),
        ("infer", "Complete one .npy/.npz partial cloud", infer),
        ("evaluate", "Evaluate prepared partial/complete .npz pairs", evaluate),
        ("train", "Train using an explicit configuration and pair directory", train),
        ("export", "Convert a checkpoint into a Hugging Face model folder", export),
    ]:
        command = commands.add_parser(name, help=help_text)
        command.set_defaults(handler=handler)
        if name != "export":
            command.add_argument("--device", default="cpu")
            command.add_argument("--threads", type=_positive, default=2)
        if name in {"infer", "evaluate", "train"}:
            command.add_argument("--seed", type=int, default=None if name == "evaluate" else 0)
        if name in {"infer", "evaluate"}:
            command.add_argument("--revision", help="Pin a Hugging Face model commit or tag")
            command.add_argument(
                "--local-files-only", action="store_true", help="Use cached Hub files only"
            )
            command.add_argument(
                "--model", required=True, help="Checkpoint, local export, or Hub model ID"
            )
        if name in {"infer", "evaluate", "export"}:
            command.add_argument("--output", type=Path, required=True)
        if name in {"evaluate", "train"}:
            command.add_argument("--pairs", type=Path, required=True)
            command.add_argument(
                "--max-samples",
                type=int,
                default=16 if name == "evaluate" else 0,
                help="0 = all; evaluation defaults to a seeded 16-sample subset",
            )
        if name == "infer":
            command.add_argument("--input", type=Path, required=True)
            command.add_argument("--key", default="partial")
        elif name == "evaluate":
            command.add_argument("--protocol", type=Path, help="JSON with frozen sample_ids")
            command.add_argument(
                "--eval-points", type=_positive, help="Default: protocol value or 1024"
            )
        elif name == "export":
            command.add_argument("--checkpoint", type=Path, required=True)
        elif name == "train":
            command.add_argument("--config", type=Path, required=True)
            command.add_argument("--workspace", type=Path, default=Path("."))
            command.add_argument("--epochs", type=_positive)
            command.add_argument("--batch-size", type=_positive)
            command.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "max_samples", 0) < 0:
        parser.error("--max-samples must be zero (all) or positive")
    if hasattr(args, "threads"):
        torch.set_num_threads(args.threads)
    if getattr(args, "device", "cpu").startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu or install a CUDA-enabled PyTorch")
    try:
        result = args.handler(args)
    except (ValueError, OSError, ImportError, RuntimeError) as exc:
        parser.exit(1, f"veil: {exc}\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0

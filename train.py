"""Train the LitePT Sonata-style SSL model from a JSON configuration."""

import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--steps", type=int, help="Total schedule horizon; keep fixed on resume")
    parser.add_argument("--stop-after", type=int, help="Run only this many additional batches")
    parser.add_argument("--dataroot")
    parser.add_argument("--version")
    parser.add_argument("--split")
    parser.add_argument("--device")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-points", type=int, help="Crop cap, useful for a smoke test")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--amp", choices=("none", "float16", "bfloat16"))
    parser.add_argument("--output-dir")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"))
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-name")
    args = parser.parse_args(argv)
    # Keep --help available without importing the CUDA model dependencies.
    import torch
    from builder import build_trainer

    if args.resume and args.config is None:
        config = torch.load(args.resume, map_location="cpu", weights_only=False)["config"]
    else:
        path = args.config or Path(__file__).parent / "configs/nuscenes_ssl_litept_s.json"
        config = json.loads(path.read_text())
    for arg, key in (("steps", "total_steps"), ("device", "device"),
                     ("workers", "num_workers"), ("batch_size", "batch_size"),
                     ("amp", "amp"), ("output_dir", "output_dir")):
        value = getattr(args, arg)
        if value is not None:
            config[key] = value
    for key in ("dataroot", "version", "split", "max_samples"):
        value = getattr(args, key)
        if value is not None:
            config["dataset"][key] = value
    if args.max_points is not None:
        config["dataset"]["augmentation"]["max_size"] = args.max_points
    schedule = config["dataset"]["augmentation"].get("mask_schedule")
    if schedule is not None:
        schedule["total_steps"] = config["total_steps"]
    wandb_options = {
        key: getattr(args, "wandb_" + key)
        for key in ("mode", "project", "entity", "name")
        if getattr(args, "wandb_" + key) is not None
    }
    if args.wandb or wandb_options:
        logging = dict(config.get("wandb", {}))
        logging.update(wandb_options)
        if args.wandb and "mode" not in wandb_options and logging.get("mode") == "disabled":
            logging["mode"] = "online"
        logging["enabled"] = logging.get("mode", "online") != "disabled"
        config["wandb"] = logging
    trainer = build_trainer({"type": "SSLTrainer", "config": config})
    if args.resume:
        trainer.load_checkpoint(args.resume)
    trainer.run(stop_after=args.stop_after)


if __name__ == "__main__":
    main()

"""CPU-only stochastic toy fixture; never an official model or research run."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from marine_3model_experiment import (
    capture_rng_state, checkpoint_identity, config_sha256, restore_rng_state,
    set_seed, sha256_file, write_session_status,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    out = Path(cfg["out_dir"])
    resume = cfg["training"]["resume"]
    if not resume and out.exists():
        raise RuntimeError("TEST_CLEAN_OUTPUT_MUST_NOT_EXIST")
    if resume and not (out / "session_status.json").exists():
        raise RuntimeError("TEST_RESUME_REQUIRES_PROCESS_A")
    directory = out / "runs/seed_42/torchvision/frcnn"
    directory.mkdir(parents=True, exist_ok=resume)
    set_seed(42)
    torch.set_num_threads(1)
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Dropout(0.25),
                                torch.nn.Linear(4, 1))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3)
    history = []
    first = 1
    if resume:
        state = torch.load(cfg["session_control"]["resume_checkpoint"],
                           map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        history = state["training_history"]
        first = state["next_epoch"]
        restore_rng_state(state)
        print("RESUME_NEXT_EPOCH=" + str(first), flush=True)
    last_epoch = 2 if cfg["session_control"]["stop_after_stage2_epoch"] == 1 else 3
    identity = checkpoint_identity(cfg, "frcnn")
    for epoch in range(first, last_epoch + 1):
        data = torch.randn(8, 4)
        target = torch.randn(8, 1)
        generator = torch.Generator().manual_seed(42 + epoch)
        order = torch.randperm(8, generator=generator)
        loss = (model(data[order]) - target[order]).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        metric = epoch / 10  # Fixture sentinel, not a measured detector metric.
        history.append({"epoch": epoch, "stage": "head" if epoch == 1 else "all",
                        "loss": loss.item(), "map": metric,
                        "lr": optimizer.param_groups[0]["lr"]})
        state = {
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "scaler": {},
            "epoch": epoch, "completed_epoch": epoch, "next_epoch": epoch + 1,
            "stage": "head" if epoch == 1 else "all", "stage_epoch": max(epoch - 1, 1),
            "best_epoch": epoch, "best_map": metric, "val_metrics": {"map": metric},
            "architecture": "frcnn", "seed": 42, "cfg": cfg,
            "training_config_sha256": config_sha256(cfg), "checkpoint_identity": identity,
            "dataset_sha256": identity["dataset_sha256"],
            "split_sha256": identity["split_manifest_sha256"],
            "git_commit": identity["git_commit"], "experiment_id": identity["experiment_id"],
            "training_history": history, "dataloader_generator_state": generator.get_state(),
            "sampler_state": {"seed": 42 + epoch, "next_global_epoch": epoch + 1},
            **capture_rng_state(),
        }
        torch.save(state, directory / "last.pt")
        torch.save(state, directory / "best.pt")
        with (directory / "history.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
    last = directory / "last.pt"
    write_session_status(out / "session_status.json", {
        "status": "SESSION_A_COMPLETE_READY_FOR_RESUME" if last_epoch == 2 else "TRAINING_COMPLETE",
        "training_complete": last_epoch == 3, "stage": "stage2",
        "completed_stage2_epoch": last_epoch - 1, "next_stage2_epoch": last_epoch,
        "next_epoch": last_epoch + 1, "last_checkpoint": str(last.resolve()),
        "last_checkpoint_sha256": sha256_file(last),
    })
    print("FIXTURE_FINISHED", flush=True)
    print("FIXTURE_STDERR_SAVED", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()

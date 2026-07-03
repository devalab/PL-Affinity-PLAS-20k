import os
import csv
import json
import math
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch, Data
from scipy import stats
from tqdm import tqdm

from GIGN import GIGN


# ── Dataset (wraps GIGN's GraphDataset without re-creating graphs) ────────────

class PLASGraphDataset(Dataset):
    def __init__(self, data_dir, data_df, dis_threshold=5, graph_type='Graph_GIGN'):
        self.graph_paths = []
        self.plc_ids     = []
        self.frame_ids   = []
        self.labels      = []
        self.groups      = {}   # plc_id -> group int

        for _, row in data_df.iterrows():
            cid   = row["pdbid"]                # e.g. "3cy2_frame64"
            label = float(row["-logKd/Ki"])
            # Extract original pdbid and frame from the compound cid
            parts = cid.rsplit("_frame", 1)
            plc_id   = parts[0] if len(parts) == 2 else cid
            frame_id = parts[1] if len(parts) == 2 else "0"

            graph_path = os.path.join(
                data_dir, cid,
                f"{graph_type}-{cid}_{dis_threshold}A.pyg"
            )
            if not os.path.exists(graph_path):
                print(f"  Warning: missing graph {graph_path}")
                continue

            if plc_id not in self.groups:
                self.groups[plc_id] = len(self.groups)

            self.graph_paths.append(graph_path)
            self.plc_ids.append(plc_id)
            self.frame_ids.append(frame_id)
            self.labels.append(label)

    def __len__(self):
        return len(self.graph_paths)

    def __getitem__(self, idx):
        data = torch.load(self.graph_paths[idx], weights_only=False)
        data.plc_id   = self.plc_ids[idx]
        data.frame_id = self.frame_ids[idx]
        data.group    = self.groups[self.plc_ids[idx]]
        data.row_idx  = idx
        return data

    def collate_fn(self, batch):
        return Batch.from_data_list(batch)


class PLASDataLoader(DataLoader):
    def __init__(self, dataset, **kwargs):
        super().__init__(dataset, collate_fn=dataset.collate_fn, **kwargs)


# ── Metrics ───────────────────────────────────────────────────────────────────

def grouped_metrics(y_true, y_pred, groups):
    y_true  = np.asarray(y_true,  dtype=float)
    y_pred  = np.asarray(y_pred,  dtype=float)
    groups  = np.asarray(groups)
    frame_mse = float(np.mean((y_pred - y_true) ** 2))
    agg_pred, agg_true = [], []
    for g in sorted(set(groups.tolist())):
        idx = groups == g
        agg_pred.append(float(np.mean(y_pred[idx])))
        agg_true.append(float(np.mean(y_true[idx])))
    agg_pred = np.asarray(agg_pred)
    agg_true = np.asarray(agg_true)
    group_mse = float(np.mean((agg_pred - agg_true) ** 2))
    if len(agg_true) >= 2 and np.std(agg_true) > 0 and np.std(agg_pred) > 0:
        pearson  = float(stats.pearsonr(agg_pred, agg_true)[0])
        spearman = float(stats.spearmanr(agg_pred, agg_true)[0])
    else:
        pearson = spearman = float("nan")
    return {
        "frame_mse":  frame_mse,
        "group_mse":  group_mse,
        "group_rmse": math.sqrt(max(group_mse, 0)),
        "pearson":    pearson,
        "spearman":   spearman,
        "n_frames":   int(len(y_true)),
        "n_plcs":     int(len(agg_true)),
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, trues, groups, plc_ids, frame_ids, row_idxs = [], [], [], [], [], []
    for batch in loader:
        batch = batch.to(device)
        pred  = model(batch).cpu().numpy()
        preds.append(pred)
        trues.append(batch.y.cpu().numpy())
        groups.extend(batch.group.tolist() if hasattr(batch.group, "tolist") else [batch.group])
        plc_ids.extend(batch.plc_id   if isinstance(batch.plc_id,   list) else [batch.plc_id])
        frame_ids.extend(batch.frame_id if isinstance(batch.frame_id, list) else [batch.frame_id])
        row_idxs.extend(batch.row_idx.tolist() if hasattr(batch.row_idx, "tolist") else [batch.row_idx])
    model.train()
    return (
        np.concatenate(preds),
        np.concatenate(trues),
        np.array(groups),
        plc_ids,
        frame_ids,
        row_idxs,
    )


def write_predictions_csv(path, split_name, y_pred, y_true, groups, plc_ids, frame_ids):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "split", "plc_id", "frame_id", "group", "y_true", "y_pred", "row_index"
        ])
        writer.writeheader()
        for i, (pred, true, g, pid, fid) in enumerate(
                zip(y_pred, y_true, groups, plc_ids, frame_ids)):
            writer.writerow({
                "split":     split_name,
                "plc_id":    pid,
                "frame_id":  fid,
                "group":     int(g),
                "y_true":    float(true),
                "y_pred":    float(pred),
                "row_index": i,
            })


# ── Training ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp",        required=True)
    parser.add_argument("--data_root",  default="./data")
    parser.add_argument("--save_dir",   default="./runs_gign_plas")
    parser.add_argument("--distance",   type=int,   default=5)
    parser.add_argument("--epochs",     type=int,   default=600)
    parser.add_argument("--patience",   type=int,   default=100)
    parser.add_argument("--bs",         type=int,   default=128)
    parser.add_argument("--eval_bs",    type=int,   default=256)
    parser.add_argument("--lr",         type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--grad_clip",  type=float, default=5.0)
    parser.add_argument("--save_every", type=int,   default=5)
    parser.add_argument("--num_workers",type=int,   default=4)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--resume",     action="store_true")
    parser.add_argument("--graph_type", default="Graph_GIGN")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_dir  = os.path.join(args.data_root, args.exp)
    run_dir  = Path(args.save_dir) / args.exp
    run_dir.mkdir(parents=True, exist_ok=True)
    resume_path = run_dir / "resume.pt"

    print(f"Device    : {device}")
    print(f"Experiment: {args.exp}")

    # ── Load datasets ─────────────────────────────────────────────────────────
    def load_split(split):
        csv_path  = os.path.join(exp_dir, f"{split}.csv")
        split_dir = os.path.join(exp_dir, split)
        if not os.path.exists(csv_path):
            return None
        df = pd.read_csv(csv_path)
        return PLASGraphDataset(split_dir, df, dis_threshold=args.distance,
                                graph_type=args.graph_type)

    train_ds = load_split("train")
    val_ds   = load_split("val")
    test_ds  = load_split("test")

    train_loader = PLASDataLoader(train_ds, batch_size=args.bs,
                                  shuffle=True,  num_workers=args.num_workers)
    val_loader   = PLASDataLoader(val_ds,   batch_size=args.eval_bs,
                                  shuffle=False, num_workers=args.num_workers)
    test_loader  = PLASDataLoader(test_ds,  batch_size=args.eval_bs,
                                  shuffle=False, num_workers=args.num_workers)

    print(f"Train: {len(train_ds)} frames | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model     = GIGN(35, 256).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.6, patience=20, min_lr=1e-6)
    criterion = nn.MSELoss()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch  = 1
    best_val_mse = float("inf")
    best_epoch   = 0
    best_state   = None
    stale        = 0

    if args.resume and resume_path.exists():
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch  = ckpt["epoch"] + 1
        best_val_mse = ckpt["best_val_mse"]
        best_epoch   = ckpt["best_epoch"]
        best_state   = ckpt["best_state"]
        stale        = ckpt["stale"]
        print(f"Resumed from epoch {start_epoch}, best_val_mse={best_val_mse:.4f}")
    elif args.resume:
        print("--resume passed but no checkpoint found. Starting fresh.")

    with open(run_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total_loss, seen = 0.0, 0
        for batch in train_loader:
            batch = batch.to(device)
            pred  = model(batch)
            loss  = criterion(pred, batch.y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            loss_val = float(loss.item())
            if not math.isnan(loss_val):
                total_loss += loss_val * batch.y.size(0)
                seen += batch.y.size(0)

        train_mse = total_loss / max(seen, 1)

        # Validation
        val_pred, val_true, val_groups, *_ = evaluate(model, val_loader, device)
        val_metrics = grouped_metrics(val_true, val_pred, val_groups)
        val_mse = val_metrics["group_mse"]
        scheduler.step(val_mse)

        print(
            f"epoch={epoch:04d} "
            f"train_mse={train_mse:.4f} "
            f"val_group_mse={val_mse:.4f} "
            f"val_rmse={val_metrics['group_rmse']:.4f} "
            f"val_r={val_metrics['pearson']:.4f} "
            f"lr={optimizer.param_groups[0]['lr']:.2e}",
            flush=True
        )

        if not math.isnan(val_mse) and val_mse < best_val_mse:
            best_val_mse = val_mse
            best_epoch   = epoch
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save({
                "model": best_state, "best_epoch": best_epoch,
                "val_metrics": val_metrics, "args": vars(args)
            }, run_dir / "best.pt")
            stale = 0
        else:
            stale += 1

        if stale >= args.patience:
            print(f"Early stopping at epoch {epoch}.")
            break

        if epoch % args.save_every == 0 or epoch == start_epoch:
            torch.save({
                "epoch": epoch, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_val_mse": best_val_mse, "best_epoch": best_epoch,
                "best_state": best_state, "stale": stale,
                "args": vars(args),
            }, resume_path)
            print(f"  [checkpoint saved epoch {epoch}]", flush=True)

    # ── Final evaluation ──────────────────────────────────────────────────────
    if best_state is None:
        raise RuntimeError("No best checkpoint saved.")
    model.load_state_dict(best_state)

    all_results = {}
    for split_name, ds, loader in [
        ("train", train_ds, PLASDataLoader(train_ds, batch_size=args.eval_bs,
                                           shuffle=False, num_workers=0)),
        ("val",   val_ds,   val_loader),
        ("test",  test_ds,  test_loader),
    ]:
        y_pred, y_true, groups, plc_ids, frame_ids, _ = evaluate(model, loader, device)
        metrics = grouped_metrics(y_true, y_pred, groups)
        all_results[split_name] = metrics
        write_predictions_csv(
            run_dir / f"{split_name}_frame_predictions.csv",
            split_name, y_pred, y_true, groups, plc_ids, frame_ids
        )
        print(f"{split_name.upper()}: group_mse={metrics['group_mse']:.4f} "
              f"rmse={metrics['group_rmse']:.4f} pearson={metrics['pearson']:.4f} "
              f"n_plcs={metrics['n_plcs']}")

    with open(run_dir / "metrics.json", "w") as f:
        json.dump({"best_epoch": best_epoch, **all_results}, f, indent=2)
    print(f"\nSaved to {run_dir}")

    if resume_path.exists():
        resume_path.unlink()


if __name__ == "__main__":
    main()

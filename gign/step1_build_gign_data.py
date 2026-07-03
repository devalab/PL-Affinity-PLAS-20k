import os
import csv
import random
import argparse
import warnings
import numpy as np
from pathlib import Path
from collections import defaultdict

# ── EDIT THESE PATHS ──────────────────────────────────────────────────────────
PLAS20K_DIR  = "../plas20k"
RESULTS_CSV  = "../aggregation-analysis/results.csv"
CLUSTER_DIR  = "../clustering"
GIGN_DATA_ROOT = "./data"   # GIGN repo root's data/ folder

# Optional: explicit train/test list files. Set to None to use results.csv Set column.
TRAIN_LIST   = None  # e.g. "/path/to/train_list.txt"
TEST_LIST    = None

VAL_FRACTION = 0.1
SEED         = 42
TOTAL_FRAMES = 200
# ─────────────────────────────────────────────────────────────────────────────


def select_uniform(num_frames, total_frames=TOTAL_FRAMES):
    """Exact replica of Pafnucy uniform strategy."""
    return list(range(1, total_frames + 1, max(1, total_frames // num_frames)))


def load_clustered_frames(cluster_dir, num_frames):
    """Load selected_points_{num_frames}.txt -> {pdbid: [frame_idx, ...]}"""
    path = os.path.join(cluster_dir, f"selected_points_{num_frames}.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Clustering file not found: {path}\n"
            f"Expected CLUSTER_DIR/selected_points_{{n}}.txt"
        )
    result = {}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            pdbid = parts[0].lower()
            frames = [int(x) for x in parts[2:]]   # parts[1] is score, skip it
            result[pdbid] = frames
    print(f"  Loaded clustered frames for {len(result)} PLCs from {path}")
    return result


def get_frame_list(strategy, pdbid, num_frames, clustered_frames=None):
    """Return 1-indexed frame list. Mirrors Pafnucy logic exactly."""
    uniform = select_uniform(num_frames)
    if strategy == "uniform":
        return uniform
    if strategy == "clustered":
        frames = (clustered_frames or {}).get(pdbid, [])
        if frames:
            return list(frames)
        warnings.warn(f"No clustered frames for '{pdbid}', falling back to uniform.")
        return uniform
    raise ValueError(f"Unknown strategy: {strategy}")


def frames_on_disk(pdbid, plas20k_dir):
    """Return set of 1-indexed frame indices that have both protein and ligand files."""
    plc_dir = os.path.join(plas20k_dir, pdbid)
    prot, lig = set(), set()
    try:
        for fname in os.listdir(plc_dir):
            if ".pw.frame_" in fname and fname.endswith(".mol2"):
                try:
                    prot.add(int(fname.split(".pw.frame_")[1].replace(".mol2", "")))
                except ValueError:
                    pass
            elif ".l.frame_" in fname and fname.endswith(".mol2"):
                try:
                    lig.add(int(fname.split(".l.frame_")[1].replace(".mol2", "")))
                except ValueError:
                    pass
    except FileNotFoundError:
        return set()
    return prot & lig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=["uniform", "clustered"], required=True)
    parser.add_argument("--n_frames", type=int, required=True)
    args = parser.parse_args()

    strategy  = args.strategy
    n_frames  = args.n_frames
    exp_name  = f"{strategy}_{n_frames}"

    random.seed(SEED)
    np.random.seed(SEED)

    # ── Load labels and split from results.csv ────────────────────────────────
    print(f"Reading {RESULTS_CSV} ...")
    labels  = {}   # pdbid -> float
    set_map = {}   # pdbid -> 'training' or 'test'
    with open(RESULTS_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw = row["Pdbid_Frame"]
            parts = raw.rsplit("_", 1)
            if len(parts) != 2:
                continue
            pdbid = parts[0].lower()
            if pdbid not in labels:
                labels[pdbid]  = float(row["Real"])
                set_map[pdbid] = row["Set"].strip().lower()
    print(f"  {len(labels)} unique PLCs found")

    # Override with list files if provided
    if TRAIN_LIST and TEST_LIST and os.path.exists(TRAIN_LIST) and os.path.exists(TEST_LIST):
        with open(TRAIN_LIST) as f:
            raw_train = {l.strip().lower() for l in f if l.strip()}
        with open(TEST_LIST) as f:
            raw_test  = {l.strip().lower() for l in f if l.strip()}
        all_train = sorted(p for p in raw_train if p in labels)
        test_plcs  = {p for p in raw_test  if p in labels}
        print(f"  Using list files: {len(all_train)} train, {len(test_plcs)} test")
    else:
        all_train = sorted(p for p, s in set_map.items() if s == "training")
        test_plcs  = {p for p, s in set_map.items() if s == "test"}

    # Carve validation from training by PLC (never by frame)
    random.shuffle(all_train)
    n_val      = max(1, int(len(all_train) * VAL_FRACTION))
    val_plcs   = set(all_train[:n_val])
    train_plcs = set(all_train[n_val:])
    print(f"  train={len(train_plcs)}, val={len(val_plcs)}, test={len(test_plcs)}")

    # ── Load clustered frames if needed ───────────────────────────────────────
    clustered_frames = None
    if strategy == "clustered":
        clustered_frames = load_clustered_frames(CLUSTER_DIR, n_frames)

    # ── Build GIGN directory structure ────────────────────────────────────────
    # Each split gets: data/{exp_name}/{split}/{cid_frameN}/
    # We also write the CSV files GIGN expects.
    splits = [
        ("train", train_plcs),
        ("val",   val_plcs),
        ("test",  test_plcs),
    ]

    csv_rows    = defaultdict(list)   # split -> list of (cid, pKa)
    skipped     = []
    fallbacks   = []
    total_frames_written = 0

    for split_name, plc_set in splits:
        split_data_dir = os.path.join(GIGN_DATA_ROOT, exp_name, split_name)
        os.makedirs(split_data_dir, exist_ok=True)

        for pdbid in sorted(plc_set):
            if pdbid not in labels:
                skipped.append(f"{pdbid} (no label)")
                continue
            available = frames_on_disk(pdbid, PLAS20K_DIR)
            if not available:
                skipped.append(f"{pdbid} (no files on disk)")
                continue

            desired = get_frame_list(strategy, pdbid, n_frames, clustered_frames)
            if strategy == "clustered" and pdbid not in (clustered_frames or {}):
                fallbacks.append(pdbid)

            label = labels[pdbid]
            src_dir = os.path.join(PLAS20K_DIR, pdbid)

            for frame_idx in desired:
                if frame_idx not in available:
                    skipped.append(f"{pdbid} frame {frame_idx} (not on disk)")
                    continue

                # Complex ID used throughout GIGN pipeline
                cid = f"{pdbid}_frame{frame_idx}"
                complex_dir = os.path.join(split_data_dir, cid)
                os.makedirs(complex_dir, exist_ok=True)

                # Symlink ligand mol2
                lig_src  = os.path.abspath(
                    os.path.join(src_dir, f"{pdbid}.l.frame_{frame_idx}.mol2"))
                lig_dst  = os.path.join(complex_dir, f"{cid}_ligand.mol2")
                if not os.path.exists(lig_dst):
                    os.symlink(lig_src, lig_dst)

                # Symlink protein mol2
                prot_src = os.path.abspath(
                    os.path.join(src_dir, f"{pdbid}.pw.frame_{frame_idx}.mol2"))
                prot_dst = os.path.join(complex_dir, f"{cid}_protein.mol2")
                if not os.path.exists(prot_dst):
                    os.symlink(prot_src, prot_dst)

                csv_rows[split_name].append((cid, label))
                total_frames_written += 1

    # ── Write CSV files ───────────────────────────────────────────────────────
    csv_dir = os.path.join(GIGN_DATA_ROOT, exp_name)
    for split_name, rows in csv_rows.items():
        csv_path = os.path.join(csv_dir, f"{split_name}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["pdbid", "-logKd/Ki"])
            writer.writerows(rows)
        plc_count = len({r[0].rsplit("_frame", 1)[0] for r in rows})
        print(f"  {split_name}: {len(rows)} frames, {plc_count} PLCs -> {csv_path}")

    print(f"\nTotal frames written: {total_frames_written}")
    if skipped:
        print(f"Skipped: {len(skipped)} entries")
    if fallbacks:
        print(f"Clustered->uniform fallbacks: {len(fallbacks)} PLCs")
    print(f"\nDirectory: {os.path.join(GIGN_DATA_ROOT, exp_name)}/")
    print(f"Next step:")
    print(f"  python step2_preprocessing_plas.py --exp {exp_name}")


if __name__ == "__main__":
    main()

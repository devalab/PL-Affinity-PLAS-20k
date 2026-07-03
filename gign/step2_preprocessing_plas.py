import os
import re
import pickle
import argparse
import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

try:
    import pymol
    PYMOL_AVAILABLE = True
except ImportError:
    PYMOL_AVAILABLE = False


# ── mol2 atom-type -> element mapping (same logic as ProtMD pipeline) ────────

TWO_LETTER = {"CL": "Cl", "BR": "Br", "MG": "Mg", "NA": "Na", "ZN": "Zn", "CA": "Ca",
              "FE": "Fe", "MN": "Mn", "CU": "Cu", "CO": "Co", "NI": "Ni", "SE": "Se",
              "SI": "Si", "AL": "Al"}
ONE_LETTER = set(["H", "C", "N", "O", "S", "P", "F", "I", "B", "K"])


def canonical_element(sybyl_type, atom_name=""):
    """
    Infer real chemical element from a mol2 SYBYL/AMBER atom type string.
    """
    token = (sybyl_type or "").strip()
    name  = (atom_name or "").strip()
    name_clean = re.sub(r"[^A-Za-z]", "", name).upper()

    if token and token[0].islower():
        base = token.split(".")[0]
        base = re.sub(r"[^A-Za-z]", "", base)
        low = base.lower()
        if low.startswith("cl"):
            return "Cl"
        if low.startswith("br"):
            return "Br"
        if low and low[0] in {"c", "n", "o", "s", "p", "f", "h", "i", "b", "k"}:
            return low[0].upper()

    if name_clean:
        if name_clean[:2] in TWO_LETTER and name_clean not in {
            "CA", "CB", "CG", "CD", "CE", "CZ", "CH",   # carbon-named backbone/sidechain
        }:
            return TWO_LETTER[name_clean[:2]]
        if name_clean[:1] in ONE_LETTER:
            return name_clean[:1]

    # Fall back to the raw type string for genuine ions (e.g. 'Na+', 'Zn2+', 'Cl-')
    if token:
        base = token.split(".")[0]
        base = re.sub(r"[^A-Za-z]", "", base)
        up = base.upper()
        if up in TWO_LETTER:
            return TWO_LETTER[up]
        if len(up) >= 2 and up[:2] in TWO_LETTER:
            return TWO_LETTER[up[:2]]
        if up[:1] in ONE_LETTER:
            return up[:1]

    return None


def parse_mol2(path):
    """
    Parse a mol2 file's ATOM and BOND blocks directly.
    Returns dict with: elements (list[str]), coords (Nx3 array),
    bonds (list of (i,j) 0-indexed pairs), atom_names (list[str]).
    """
    elements, coords, atom_names = [], [], []
    bonds = []
    section = None
    with open(path, "r", errors="replace") as f:
        for line in f:
            if line.startswith("@<TRIPOS>ATOM"):
                section = "ATOM"
                continue
            if line.startswith("@<TRIPOS>BOND"):
                section = "BOND"
                continue
            if line.startswith("@<TRIPOS>"):
                section = None
                continue

            if section == "ATOM":
                parts = line.split()
                if len(parts) < 6:
                    continue
                atom_name = parts[1]
                try:
                    x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                except ValueError:
                    continue
                sybyl_type = parts[5]
                elem = canonical_element(sybyl_type, atom_name)
                if elem is None:
                    elem = "C"  # last-resort fallback, rare
                elements.append(elem)
                coords.append((x, y, z))
                atom_names.append(atom_name)

            elif section == "BOND":
                parts = line.split()
                if len(parts) < 3:
                    continue
                try:
                    i, j = int(parts[1]) - 1, int(parts[2]) - 1  # mol2 is 1-indexed
                except ValueError:
                    continue
                bonds.append((i, j))

    return {
        "elements": elements,
        "coords": np.array(coords, dtype=np.float64),
        "bonds": bonds,
        "atom_names": atom_names,
    }


def build_rwmol(elements, coords, bonds, keep_idx=None):
    """
    Build an RDKit RWMol from parsed atom/bond data, with correct elements
    and a 3D conformer. If keep_idx is given, only those atom indices
    (0-indexed into the original arrays) are included, and bonds are
    remapped/dropped accordingly.
    """
    if keep_idx is None:
        keep_idx = list(range(len(elements)))
    keep_set = set(keep_idx)
    old_to_new = {old: new for new, old in enumerate(keep_idx)}

    mol = Chem.RWMol()
    conf_coords = []
    for old_idx in keep_idx:
        elem = elements[old_idx]
        try:
            atom = Chem.Atom(elem)
        except Exception:
            atom = Chem.Atom("C")
        mol.AddAtom(atom)
        conf_coords.append(coords[old_idx])

    for i, j in bonds:
        if i in keep_set and j in keep_set:
            ni, nj = old_to_new[i], old_to_new[j]
            if ni != nj and not mol.GetBondBetweenAtoms(ni, nj):
                try:
                    mol.AddBond(ni, nj, Chem.BondType.SINGLE)
                except Exception:
                    pass

    conf = Chem.Conformer(mol.GetNumAtoms())
    for i, (x, y, z) in enumerate(conf_coords):
        conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
    mol.AddConformer(conf, assignId=True)

    final_mol = mol.GetMol()
    try:
        Chem.SanitizeMol(final_mol, catchErrors=True)
    except Exception:
        pass
    return final_mol


def select_pocket_indices_pymol(protein_mol2, ligand_mol2, cid, distance):
    """
    Use PyMOL purely as a geometric selection engine: load both mol2 files,
    select protein atoms within `distance` of ligand (byres), and return the
    0-indexed atom indices into the protein mol2 ATOM block that PyMOL kept.
    We never read element identities back from PyMOL — only the selection.
    """
    try:
        pymol.cmd.delete("all")
        pymol.cmd.load(protein_mol2, f"{cid}_protein")
        pymol.cmd.remove("resn HOH")
        pymol.cmd.load(ligand_mol2, f"{cid}_ligand")
        pymol.cmd.remove("hydrogens")
        pymol.cmd.select(
            "Pocket", f"byres {cid}_protein and {cid}_protein within {distance} of {cid}_ligand"
        )
        n = pymol.cmd.count_atoms("Pocket")
        if n == 0:
            pymol.cmd.select(
                "Pocket", f"{cid}_protein within {distance} of {cid}_ligand"
            )
            n = pymol.cmd.count_atoms("Pocket")
        if n == 0:
            pymol.cmd.delete("all")
            return None

        # Get coordinates of selected atoms; match back to mol2 order by position
        model = pymol.cmd.get_model("Pocket")
        sel_coords = np.array([a.coord for a in model.atom], dtype=np.float64)
        pymol.cmd.delete("all")
        return sel_coords
    except Exception as e:
        try:
            pymol.cmd.delete("all")
        except Exception:
            pass
        print(f"  PyMOL error for {cid}: {e}")
        return None


def select_pocket_indices_numpy(protein_coords, ligand_coords, distance):
    """Pure numpy fallback: indices of protein atoms within `distance` of any ligand atom."""
    min_dists = np.full(len(protein_coords), np.inf)
    chunk = 512
    for start in range(0, len(ligand_coords), chunk):
        lig_chunk = ligand_coords[start:start + chunk]
        diff = protein_coords[:, None, :] - lig_chunk[None, :, :]
        d = np.sqrt((diff ** 2).sum(axis=2)).min(axis=1)
        min_dists = np.minimum(min_dists, d)
    return np.where(min_dists < distance)[0]


def match_coords_to_indices(all_coords, sel_coords, tol=0.01):
    """Match PyMOL-returned coordinates back to indices in the original array."""
    idx = []
    used = set()
    for sc in sel_coords:
        diffs = np.sqrt(((all_coords - sc) ** 2).sum(axis=1))
        order = np.argsort(diffs)
        for cand in order[:5]:
            if cand not in used and diffs[cand] < tol:
                idx.append(int(cand))
                used.add(cand)
                break
    return idx


def process_one(args_tuple):
    cid, complex_dir, distance, use_pymol = args_tuple

    rdkit_path   = os.path.join(complex_dir, f"{cid}_{distance}A.rdkit")
    protein_mol2 = os.path.join(complex_dir, f"{cid}_protein.mol2")
    ligand_mol2  = os.path.join(complex_dir, f"{cid}_ligand.mol2")

    if os.path.exists(rdkit_path):
        return (cid, "skipped_already_exists")
    if not os.path.exists(protein_mol2):
        return (cid, "error: protein mol2 missing")
    if not os.path.exists(ligand_mol2):
        return (cid, "error: ligand mol2 missing")

    # ── Parse both mol2 files directly (no PDB intermediate) ─────────────────
    try:
        lig_data  = parse_mol2(ligand_mol2)
        prot_data = parse_mol2(protein_mol2)
    except Exception as e:
        return (cid, f"error: mol2 parse failed: {e}")

    if len(lig_data["coords"]) == 0:
        return (cid, "error: ligand mol2 has 0 atoms")
    if len(prot_data["coords"]) == 0:
        return (cid, "error: protein mol2 has 0 atoms")

    # Drop hydrogens from ligand (matches GIGN's removeHs=True convention)
    lig_keep = [i for i, e in enumerate(lig_data["elements"]) if e != "H"]
    if not lig_keep:
        return (cid, "error: ligand has 0 heavy atoms")

    # ── Select pocket atom indices ────────────────────────────────────────────
    pocket_idx = None
    if use_pymol and PYMOL_AVAILABLE:
        sel_coords = select_pocket_indices_pymol(protein_mol2, ligand_mol2, cid, distance)
        if sel_coords is not None and len(sel_coords) > 0:
            pocket_idx = match_coords_to_indices(prot_data["coords"], sel_coords)
    if not pocket_idx:
        # Fallback to plain numpy distance cutoff (also drops protein H here)
        heavy_prot_idx = np.array(
            [i for i, e in enumerate(prot_data["elements"]) if e != "H"])
        if len(heavy_prot_idx) == 0:
            return (cid, "error: protein has 0 heavy atoms")
        sel = select_pocket_indices_numpy(
            prot_data["coords"][heavy_prot_idx],
            lig_data["coords"][lig_keep],
            distance,
        )
        if len(sel) == 0:
            return (cid, "error: no pocket atoms within cutoff")
        pocket_idx = heavy_prot_idx[sel].tolist()
    else:
        # PyMOL selection may include H if not stripped; drop them
        pocket_idx = [i for i in pocket_idx if prot_data["elements"][i] != "H"]
        if not pocket_idx:
            return (cid, "error: pocket selection empty after H removal")

    # ── Build RDKit mols with CORRECT elements (the actual fix) ───────────────
    try:
        ligand = build_rwmol(
            lig_data["elements"], lig_data["coords"], lig_data["bonds"],
            keep_idx=lig_keep,
        )
        pocket = build_rwmol(
            prot_data["elements"], prot_data["coords"], prot_data["bonds"],
            keep_idx=pocket_idx,
        )
    except Exception as e:
        return (cid, f"error: RWMol build failed: {e}")

    if ligand is None or ligand.GetNumAtoms() == 0:
        return (cid, "error: built ligand mol is empty")
    if pocket is None or pocket.GetNumAtoms() == 0:
        return (cid, "error: built pocket mol is empty")
    if ligand.GetNumConformers() == 0 or pocket.GetNumConformers() == 0:
        return (cid, "error: missing 3D conformer")

    with open(rdkit_path, "wb") as f:
        pickle.dump((ligand, pocket), f)

    return (cid, "ok")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp",       required=True)
    parser.add_argument("--distance",  type=int, default=5)
    parser.add_argument("--workers",   type=int, default=8)
    parser.add_argument("--no_pymol",  action="store_true")
    parser.add_argument("--data_root", default="./data")
    args = parser.parse_args()

    exp_dir   = os.path.join(args.data_root, args.exp)
    distance  = args.distance
    use_pymol = (not args.no_pymol) and PYMOL_AVAILABLE

    if not PYMOL_AVAILABLE:
        print("PyMOL not found — using pure numpy fallback for pocket selection.")
        use_pymol = False

    print(f"Experiment : {args.exp}")
    print(f"Data dir   : {exp_dir}")
    print(f"Distance   : {distance}A")
    print(f"Workers    : {args.workers}")
    print(f"PyMOL      : {use_pymol}  (used only for atom SELECTION, never identity)")
    print(f"Parsing    : mol2 directly, element derived from SYBYL/AMBER atom type")

    work_items = []
    for split in ["train", "val", "test"]:
        csv_path  = os.path.join(exp_dir, f"{split}.csv")
        split_dir = os.path.join(exp_dir, split)
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        for cid in df["pdbid"]:
            complex_dir = os.path.join(split_dir, cid)
            if os.path.isdir(complex_dir):
                work_items.append((cid, complex_dir, distance, use_pymol))

    already = sum(
        1 for (cid, cdir, d, _) in work_items
        if os.path.exists(os.path.join(cdir, f"{cid}_{d}A.rdkit"))
    )
    print(f"\nTotal: {len(work_items)} | Already done: {already} | "
          f"To process: {len(work_items) - already}\n")

    n_workers = 1 if use_pymol else args.workers
    if use_pymol and args.workers > 1:
        print("PyMOL mode: single worker (PyMOL is not fork-safe).")
        print("Use --no_pymol for parallel processing.\n")

    errors, ok_count = [], 0

    if n_workers == 1:
        for item in tqdm(work_items):
            cid, status = process_one(item)
            if status in ("ok", "skipped_already_exists"):
                ok_count += 1
            else:
                errors.append((cid, status))
    else:
        with multiprocessing.Pool(n_workers) as pool:
            results = list(tqdm(
                pool.imap(process_one, work_items), total=len(work_items)))
        for (cid, cdir, d, _), (_, status) in zip(work_items, results):
            if status in ("ok", "skipped_already_exists"):
                ok_count += 1
            else:
                errors.append((cid, status))

    print(f"\nDone. ok={ok_count}, errors={len(errors)}")
    if errors:
        print(f"First 20 errors:")
        for cid, msg in errors[:20]:
            print(f"  {cid}: {msg}")
        err_log = os.path.join(exp_dir, "preprocessing_errors.txt")
        with open(err_log, "w") as f:
            for cid, msg in errors:
                f.write(f"{cid}\t{msg}\n")
        print(f"Error log: {err_log}")

    print(f"\nNext: python step3_build_graphs_plas.py --exp {args.exp}")


if __name__ == "__main__":
    main()

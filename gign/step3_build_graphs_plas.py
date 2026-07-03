import os
import pickle
import argparse
import multiprocessing
from itertools import repeat

import pandas as pd
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data
from scipy.spatial import distance_matrix
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from tqdm import tqdm


# ── Atom featurisation (identical to dataset_GIGN.py) ────────────────────────

def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_features(mol, graph,
                  atom_symbols=['C','N','O','S','F','P','Cl','Br','I'],
                  explicit_H=True):
    for atom in mol.GetAtoms():
        results = (
            one_of_k_encoding_unk(atom.GetSymbol(), atom_symbols + ['Unknown']) +
            one_of_k_encoding_unk(atom.GetDegree(), [0,1,2,3,4,5,6]) +
            one_of_k_encoding_unk(atom.GetImplicitValence(), [0,1,2,3,4,5,6]) +
            one_of_k_encoding_unk(atom.GetHybridization(), [
                Chem.rdchem.HybridizationType.SP,
                Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3,
                Chem.rdchem.HybridizationType.SP3D,
                Chem.rdchem.HybridizationType.SP3D2,
            ]) +
            [atom.GetIsAromatic()]
        )
        if explicit_H:
            results += one_of_k_encoding_unk(atom.GetTotalNumHs(), [0,1,2,3,4])
        graph.add_node(atom.GetIdx(), feats=torch.from_numpy(np.array(results, dtype=np.float32)))


def get_edge_index(mol, graph):
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        graph.add_edge(i, j)


def mol2graph(mol):
    graph = nx.Graph()
    atom_features(mol, graph)
    get_edge_index(mol, graph)
    graph = graph.to_directed()
    x = torch.stack([feats['feats'] for _, feats in graph.nodes(data=True)])
    edge_index = torch.stack([
        torch.LongTensor((u, v)) for u, v in graph.edges(data=False)
    ]).T
    return x, edge_index


def inter_graph(ligand, pocket, dis_threshold=5.):
    atom_num_l = ligand.GetNumAtoms()
    graph_inter = nx.Graph()
    pos_l = ligand.GetConformers()[0].GetPositions()
    pos_p = pocket.GetConformers()[0].GetPositions()
    dis_matrix_ = distance_matrix(pos_l, pos_p)
    node_idx = np.where(dis_matrix_ < dis_threshold)
    for i, j in zip(node_idx[0], node_idx[1]):
        graph_inter.add_edge(i, j + atom_num_l)
    graph_inter = graph_inter.to_directed()
    edge_index_inter = torch.stack([
        torch.LongTensor((u, v)) for u, v in graph_inter.edges(data=False)
    ]).T
    return edge_index_inter


def mols2graphs(complex_path, label, save_path, dis_threshold=5.):
    """Identical logic to dataset_GIGN.py mols2graphs."""
    with open(complex_path, 'rb') as f:
        ligand, pocket = pickle.load(f)

    atom_num_l = ligand.GetNumAtoms()
    atom_num_p = pocket.GetNumAtoms()

    pos_l = torch.FloatTensor(ligand.GetConformers()[0].GetPositions())
    pos_p = torch.FloatTensor(pocket.GetConformers()[0].GetPositions())
    x_l, edge_index_l = mol2graph(ligand)
    x_p, edge_index_p = mol2graph(pocket)
    x = torch.cat([x_l, x_p], dim=0)
    edge_index_intra = torch.cat([edge_index_l, edge_index_p + atom_num_l], dim=-1)
    edge_index_inter = inter_graph(ligand, pocket, dis_threshold=dis_threshold)
    y   = torch.FloatTensor([label])
    pos = torch.cat([pos_l, pos_p], dim=0)
    split = torch.cat([
        torch.zeros((atom_num_l,)), torch.ones((atom_num_p,))
    ], dim=0)

    data = Data(
        x=x,
        edge_index_intra=edge_index_intra,
        edge_index_inter=edge_index_inter,
        y=y,
        pos=pos,
        split=split,
    )
    torch.save(data, save_path)


def build_one(args_tuple):
    complex_path, label, save_path, dis_threshold = args_tuple
    if os.path.exists(save_path):
        return "skipped"
    if not os.path.exists(complex_path):
        return f"error: rdkit file missing {complex_path}"
    try:
        mols2graphs(complex_path, label, save_path, dis_threshold)
        return "ok"
    except Exception as e:
        return f"error: {e}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp",      required=True)
    parser.add_argument("--distance", type=int,   default=5)
    parser.add_argument("--workers",  type=int,   default=8)
    parser.add_argument("--data_root", default="./data")
    args = parser.parse_args()

    exp_dir      = os.path.join(args.data_root, args.exp)
    distance     = args.distance
    graph_type   = "Graph_GIGN"

    work_items = []
    for split in ["train", "val", "test"]:
        csv_path  = os.path.join(exp_dir, f"{split}.csv")
        split_dir = os.path.join(exp_dir, split)
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            cid   = row["pdbid"]
            label = float(row["-logKd/Ki"])
            cdir  = os.path.join(split_dir, cid)
            complex_path = os.path.join(cdir, f"{cid}_{distance}A.rdkit")
            save_path    = os.path.join(cdir, f"{graph_type}-{cid}_{distance}A.pyg")
            work_items.append((complex_path, label, save_path, float(distance)))

    already = sum(1 for (_, _, sp, _) in work_items if os.path.exists(sp))
    print(f"Experiment : {args.exp}")
    print(f"Total      : {len(work_items)} graphs")
    print(f"Already done (skip): {already}")
    print(f"To build   : {len(work_items) - already}")

    errors = []
    with multiprocessing.Pool(args.workers) as pool:
        results = list(tqdm(pool.imap(build_one, work_items), total=len(work_items)))

    ok = sum(1 for r in results if r in ("ok", "skipped"))
    for i, r in enumerate(results):
        if r not in ("ok", "skipped"):
            errors.append((work_items[i][0], r))

    print(f"\nDone. ok={ok}, errors={len(errors)}")
    if errors:
        print("First 20 errors:")
        for path, msg in errors[:20]:
            print(f"  {path}: {msg}")
        err_log = os.path.join(exp_dir, "graph_errors.txt")
        with open(err_log, "w") as f:
            for path, msg in errors:
                f.write(f"{path}\t{msg}\n")

    print(f"\nNext step:")
    print(f"  python step4_train_plas.py --exp {args.exp}")


if __name__ == "__main__":
    main()

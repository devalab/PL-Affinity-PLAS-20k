import numpy as np
import pandas as pd
import h5py
import os
import random
import argparse

import pybel
from tfbio.data import Featurizer

import warnings

random.seed(42)

parser = argparse.ArgumentParser(
    description='Process protein-ligand binding data with different frame selection methods.')
parser.add_argument('--num_frames', type=int, default=30, help='Number of frames to select.')
parser.add_argument('--total_frames', type=int, default=200, help='Total number of frames available.')
parser.add_argument('--selection_type', type=str, default='uniform',
                    choices=['uniform', 'random', 'starting', 'clustered'], help='Frame selection method.')
args = parser.parse_args()

num_frames = args.num_frames
total_frames = args.total_frames
selection_type = args.selection_type

path = '../plas20k/'

binding_affinity_data = pd.read_csv('../final_experimental.csv')

all_items = os.listdir(path)
pdb_ids = [item for item in all_items if os.path.isdir(os.path.join(path, item))]

random.shuffle(pdb_ids)
train_pdbids = pdb_ids[:int(len(pdb_ids) * 0.9)]
test_pdbids = pdb_ids[int(len(pdb_ids) * 0.9):]
RT = (1.987 / 1000) * 300

if selection_type == 'clustered':
    clustered_frames = {}
    cluster_file_path = f'../clustering/selected_points_{num_frames}.txt'
    if not os.path.exists(cluster_file_path):
        raise FileNotFoundError(f"Clustering file not found: {cluster_file_path}")
    with open(cluster_file_path, 'r') as cf:
        for line in cf:
            parts = line.split()
            pdbid_val = parts[0]
            # The second part is a number and is not needed for frame list.
            frames_list = [int(f) for f in parts[2:]]
            clustered_frames[pdbid_val] = frames_list

output_csv_name = f'plas20k_affinity_data_{num_frames}_{selection_type}.csv'

with open(output_csv_name, 'w') as f1:
    f1.write('pdbid affinity set\n')
    for pdbid in train_pdbids:
        affinity = binding_affinity_data[binding_affinity_data['Pdbid'] == pdbid]['Experimental'].values[0]
        affinity = -(affinity / RT) / 2.303

        frames_to_use = []
        if selection_type == 'uniform':
            step = max(1, total_frames // num_frames)
            frames_to_use = range(1, total_frames + 1, step)
        elif selection_type == 'random':
            frames_to_use = random.sample(range(1, total_frames + 1), num_frames)
        elif selection_type == 'starting':
            frames_to_use = range(1, num_frames + 1)
        elif selection_type == 'clustered':
            frames_to_use = clustered_frames.get(pdbid, [])
            if not frames_to_use:
                warnings.warn(f'No clustered frames found for {pdbid}')
                continue

        for frame in frames_to_use:
            f1.write(pdbid + '_' + str(frame) + ' ' + str(affinity) + ' ' + 'training' + '\n')

    for pdbid in test_pdbids:
        affinity = binding_affinity_data[binding_affinity_data['Pdbid'] == pdbid]['Experimental'].values[0]
        affinity = -(affinity / RT) / 2.303

        frames_to_use = []
        if selection_type == 'uniform':
            step = max(1, total_frames // num_frames)
            frames_to_use = range(1, total_frames + 1, step)
        elif selection_type == 'random':
            frames_to_use = random.sample(range(1, total_frames + 1), num_frames)
        elif selection_type == 'starting':
            frames_to_use = range(1, num_frames + 1)
        elif selection_type == 'clustered':
            frames_to_use = clustered_frames.get(pdbid, [])
            if not frames_to_use:
                warnings.warn(f'No clustered frames found for {pdbid}')
                continue

        for frame in frames_to_use:
            f1.write(pdbid + '_' + str(frame) + ' ' + str(affinity) + ' ' + 'test' + '\n')

affinity_data = pd.read_csv(output_csv_name, delimiter=' ')

featurizer = Featurizer()
charge_idx = featurizer.FEATURE_NAMES.index('partialcharge')

dest_path = f'hdf_{num_frames}_{selection_type}'
os.makedirs(dest_path, exist_ok=True)
data = affinity_data[affinity_data['set'] == 'training']
dataset_name = 'training_set'
i = 0
training_pocket_failed = set()
training_ligand_failed = set()
with h5py.File('%s/%s.hdf' % (dest_path, dataset_name), 'w') as f:
    for _, row in data.iterrows():

        name = row['pdbid']
        pdbid = name[:4]
        frame = name[5:]
        affinity = row['affinity']
        try:
            ligand = next(pybel.readfile('mol2', f'{path}/{pdbid}/{pdbid}.l.frame_{frame}.mol2'))
            # do not add the hydrogens! they are in the strucutre and it would reset the charges
        except:
            warnings.warn('ligand issues for %s (%s set)' % (name, dataset_name))
            training_ligand_failed.add(pdbid)
            continue

        try:
            pocket = next(pybel.readfile('mol2', f'{path}/{pdbid}/{pdbid}.pw.frame_{frame}.mol2'))
            # do not add the hydrogens! they were already added in chimera and it would reset the charges
        except:
            warnings.warn('pocket issues for %s (%s set)' % (name, dataset_name))
            training_pocket_failed.add(pdbid)
            continue

        try:
            ligand_coords, ligand_features = featurizer.get_features(ligand, molcode=1)
            if not (ligand_features[:, charge_idx] != 0).any():
                training_ligand_failed.add(pdbid)
                continue
        except:
            training_ligand_failed.add(pdbid)
            continue

        try:
            pocket_coords, pocket_features = featurizer.get_features(pocket, molcode=-1)
            if not (pocket_features[:, charge_idx] != 0).any():
                training_pocket_failed.add(pdbid)
                continue
        except:
            training_pocket_failed.add(pdbid)
            continue

        centroid = ligand_coords.mean(axis=0)
        ligand_coords -= centroid
        pocket_coords -= centroid

        data = np.concatenate((np.concatenate((ligand_coords, pocket_coords)),
                               np.concatenate((ligand_features, pocket_features))), axis=1)
        dataset = f.create_dataset(name, data=data, shape=data.shape, dtype='float32', compression='lzf')
        dataset.attrs['affinity'] = affinity
        i += 1
        if i % 100 == 0:
            print('prepared', i, 'complexes', flush=True)

print('prepared', i, 'complexes for training', flush=True)
print('failed ligands:', len(training_ligand_failed), flush=True)
for pdbid in training_ligand_failed:
    print(pdbid, flush=True)
print('failed pockets:', len(training_pocket_failed), flush=True)
for pdbid in training_pocket_failed:
    print(pdbid, flush=True)

featurizer = Featurizer()
charge_idx = featurizer.FEATURE_NAMES.index('partialcharge')

dest_path = f'hdf_{num_frames}_{selection_type}'
data = affinity_data[affinity_data['set'] == 'test']
dataset_name = 'test_set'
i = 0
test_pocket_failed = set()
test_ligand_failed = set()
with h5py.File('%s/%s.hdf' % (dest_path, dataset_name), 'w') as f:
    for _, row in data.iterrows():

        name = row['pdbid']
        pdbid = name[:4]
        frame = name[5:]
        affinity = row['affinity']

        try:
            ligand = next(pybel.readfile('mol2', f'{path}/{pdbid}/{pdbid}.l.frame_{frame}.mol2'))
            # do not add the hydrogens! they are in the strucutre and it would reset the charges
        except:
            warnings.warn('ligand issues for %s (%s set)' % (name, dataset_name))
            test_ligand_failed.add(pdbid)
            continue

        try:
            pocket = next(pybel.readfile('mol2', f'{path}/{pdbid}/{pdbid}.pw.frame_{frame}.mol2'))
            # do not add the hydrogens! they were already added in chimera and it would reset the charges
        except:
            warnings.warn('pocket issues for %s (%s set)' % (name, dataset_name))
            test_pocket_failed.add(pdbid)
            continue

        try:
            ligand_coords, ligand_features = featurizer.get_features(ligand, molcode=1)
            if not (ligand_features[:, charge_idx] != 0).any():
                test_ligand_failed.add(pdbid)
                continue
        except:
            test_ligand_failed.add(pdbid)
            continue

        try:
            pocket_coords, pocket_features = featurizer.get_features(pocket, molcode=-1)
            if not (pocket_features[:, charge_idx] != 0).any():
                test_pocket_failed.add(pdbid)
                continue
        except:
            test_pocket_failed.add(pdbid)
            continue

        centroid = ligand_coords.mean(axis=0)
        ligand_coords -= centroid
        pocket_coords -= centroid

        data = np.concatenate((np.concatenate((ligand_coords, pocket_coords)),
                               np.concatenate((ligand_features, pocket_features))), axis=1)
        dataset = f.create_dataset(name, data=data, shape=data.shape, dtype='float32', compression='lzf')
        dataset.attrs['affinity'] = affinity
        i += 1
        if i % 100 == 0:
            print('prepared', i, 'complexes', flush=True)

print('prepared', i, 'complexes for testing', flush=True)
print('failed ligands:', len(test_ligand_failed), flush=True)
for pdbid in test_ligand_failed:
    print(pdbid, flush=True)
print('failed pockets:', len(test_pocket_failed), flush=True)
for pdbid in test_pocket_failed:
    print(pdbid, flush=True)
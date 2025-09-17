import numpy as np
import pandas as pd
import h5py
import os
import random

import pybel
from tfbio.data import Featurizer

import warnings

random.seed(42)

path  = '../plas20k/'

binding_affinity_data = pd.read_csv('../final_experimental.csv')

all_items = os.listdir(path)
pdb_ids = [item for item in all_items if os.path.isdir(os.path.join(path, item))]

num_frames = 30

random.shuffle(pdb_ids)
train_pdbids = pdb_ids[:int(len(pdb_ids)*0.9)]
test_pdbids = pdb_ids[int(len(pdb_ids)*0.9):]
RT = (1.987 / 1000) * 300

with open('plas20k_affinity_data_30_uniform.csv', 'w') as f1:
    f1.write('pdbid affinity set\n')
    for pdbid in train_pdbids:
        affinity = binding_affinity_data[binding_affinity_data['Pdbid'] == pdbid]['Experimental'].values[0]
        affinity = -(affinity / RT) / 2.303
        for frame in range(1, 201, 200 // num_frames):
            f1.write(pdbid + '_' + str(frame) + ' ' + str(affinity) + ' ' + 'training' + '\n')
    for pdbid in test_pdbids:
        affinity = binding_affinity_data[binding_affinity_data['Pdbid'] == pdbid]['Experimental'].values[0]
        affinity = -(affinity / RT) / 2.303
        for frame in range(1, 201, 200 // num_frames):
            f1.write(pdbid + '_' + str(frame) + ' ' + str(affinity) + ' ' + 'test' + '\n')

affinity_data = pd.read_csv('plas20k_affinity_data_30_uniform.csv', delimiter=' ')

featurizer = Featurizer()
charge_idx = featurizer.FEATURE_NAMES.index('partialcharge')

dest_path = 'hdf_30_uniform'
data = affinity_data[affinity_data['set']=='training']    
dataset_name='training_set'
i=0
training_pocket_failed=set()
training_ligand_failed=set()
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

dest_path = 'hdf_30_uniform'
data = affinity_data[affinity_data['set']=='test']
dataset_name='test_set'
i=0
test_pocket_failed=set()
test_ligand_failed=set()
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

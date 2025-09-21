# Synthetic Data for More Accurate Deep Learning Models in Molecular Science: A Test Case of Protein-Ligand Binding Affinity Prediction

> Deep learning models are data-hungry, and synthetic (artificial) data has been shown to be invaluable when data availability is low. While this has been demonstrated in certain technology areas, adopting such an approach is new in machine learning (ML) applications in chemistry, except for some pre-training tasks. In drug discovery, predicting binding energy between proteins and ligands is crucial. Many ML-based studies have been proposed to predict protein-ligand binding affinity using existing experimental data. However, it has been shown that these models suffer from inherent biases. Recent efforts have resulted in PLAS-20k, a synthetic dataset of multiple protein-ligand complex (PLC) conformations generated using molecular dynamics (MD) simulation as a viable option to be used along with existing experimental data to improve binding affinity prediction. For the binding affinity prediction task, we employ Pafnucy, a deep convolutional neural network, and we propose using multiple structures for each PLC from PLAS-20k to train it. We compare four different statistical and ML-based result-aggregation techniques. This work demonstrates the utility of dynamic datasets in enhancing binding affinity predictions, laying the foundations for future improvements in predicting similar protein properties by using synthetic datasets and more sophisticated models and methods. We propose that synthetic datasets from physics-based methods can significantly help develop more accurate data-driven methods.

---

## Repository Overview

This repository contains the code and workflows used in our study. The overall pipeline is divided into two major parts:

1. **Training Pafnucy** on protein-ligand complexes (PLCs)  
2. **Aggregation and analysis** of predicted values across multiple conformational frames of each PLC  

[//]: # (The directory structure is as follows:)

[//]: # ()
[//]: # (```)

[//]: # (.)

[//]: # (├── aggregation-analysis)

[//]: # (│   ├── final-analysis.ipynb)

[//]: # (│   ├── neural-network-aggregation.py)

[//]: # (│   ├── output-predictions.csv)

[//]: # (│   └── results.csv)

[//]: # (├── clustering)

[//]: # (│   ├── cluster.py)

[//]: # (│   ├── get_rmsd.sh)

[//]: # (│   ├── list-pdbids.txt)

[//]: # (│   ├── rmsd)

[//]: # (│   │   └── rmsd_matrix_16pk.dat)

[//]: # (│   ├── rmsd_matrix.tcl)

[//]: # (│   └── selected_points_30.txt)

[//]: # (├── final_experimental.csv)

[//]: # (├── pafnucy)

[//]: # (│   ├── plas20k_custom_hdf.py)

[//]: # (│   └── training.py)

[//]: # (├── plas20k)

[//]: # (│   └── 16pk)

[//]: # (│       ├── 16pk.l.frame_1.mol2)

[//]: # (│       ├── 16pk.l.frame_200.mol2)

[//]: # (│       ├── 16pk.l.frame_2.mol2)

[//]: # (│       ├── 16pk.pw.frame_1.mol2)

[//]: # (│       ├── 16pk.pw.frame_200.mol2)

[//]: # (│       └── 16pk.pw.frame_2.mol2)

[//]: # (└── README.md)

[//]: # (```)

---

## 1. Data Preparation

Before training, the dataset must be prepared in **HDF format**.

- Place the raw PLC structure files inside the `plas20k/` directory.  
- An example is provided in this repository.  
- For the full PLAS-20k dataset, visit: [PLAS-20k dataset](https://healthcare.iiit.ac.in/d4/plas20k/about/about.html)

Run the following command to generate HDF files:

```bash
python3 ./pafnucy/plas20k_custom_hdf.py \
  --num_frames 30 \
  --total_frames 200 \
  --selection_type uniform
```

### Available Arguments
- `--num_frames`: Number of frames to select per PLC  
- `--total_frames`: Total number of frames available  
- `--selection_type`: Frame selection method  
  - Options: `uniform`, `random`, `starting`, `clustered`

---

## 2. Frame Selection (Clustering Option)

If `--selection_type clustered` is chosen, clustering-based frame selection must be performed.  

Steps:

1. **Compute RMSD Matrices**  
   Run the following (requires [VMD](https://www.ks.uiuc.edu/Research/vmd/) software):  
   ```bash
   ./clustering/get_rmsd.sh
   ```
   This generates RMSD matrices in the `clustering/rmsd/` subdirectory.

2. **Select Frames via Clustering**  
   Run:  
   ```bash
   python3 ./clustering/cluster.py \
     --num_frames 25 \
     --total_num_frames 200
   ```
   - Requires `list-pdbids.txt` (list of PDBIDs for PLCs of interest).  
   - Outputs a file like `selected_points_<num_frames>.txt` in the `clustering/` directory.

---

## 3. Training Pafnucy

Once HDF files are ready, training can be run using:

```bash
python3 ./pafnucy/training.py \
  --input_dir ./hdf_30_uniform/ \
  --output_prefix ./plas20k_results_30_uniform/output
```

- The training script is adapted from the [original Pafnucy repository](https://gitlab.com/cheminfIBB/pafnucy.git).  
- After completion, results are saved in `output-predictions.csv`, containing predicted binding affinities for each frame in **-log(K_d)** units.  

---

## 4. Aggregation & Analysis

All aggregation and downstream analysis are in the `aggregation-analysis/` directory.  

### Steps:

1. Place `output-predictions.csv` inside `aggregation-analysis/`.  

2. Open and run **`final-predictions.ipynb`**  
   - Generates `results.csv`  
   - Converts predictions from **-log(K_d)** units to **kcal/mol** units  

3. Run the **feed-forward neural network (FFNN)** aggregator:  
   ```bash
   python3 ./aggregation-analysis/neural-network-aggregation.py
   ```

4. Run the final analysis notebook:  
   ```bash
   jupyter notebook aggregation-analysis/final-analysis.ipynb
   ```
   - Produces results and plots  
   - Compares different aggregation methods  

---

## 5. Example Files

- **`plas20k/`** → Example PLC frames in `.mol2` format  
- **`clustering/rmsd/`** → Example RMSD matrix file  
- **`clustering/selected_points_30.txt`** → Example clustered frame selection output  

---

## Requirements

We provide two conda environment files to set up the required dependencies:

- **`train_env.yml`** → Environment for the **training** part (`pafnucy/` and `clustering/`)  
- **`aggregate_env.yml`** → Environment for the **aggregation and analysis** part (`aggregation-analysis/`)  

To create the environments:

```bash
# For training
conda env create -f train_env.yml
conda activate train_env

# For aggregation & analysis
conda env create -f aggregate_env.yml
conda activate aggregate_env
```
 

---

[//]: # (## Citation)

[//]: # ()
[//]: # (If you use this code or dataset, please cite:)

[//]: # ()
[//]: # (```)

[//]: # ([Paper citation, once available])

[//]: # (```)

[//]: # ()
[//]: # (---)

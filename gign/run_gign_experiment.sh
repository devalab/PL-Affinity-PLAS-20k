# ── Edit these two lines ──────────────────────────────────────────────────────
STRATEGY=uniform       # uniform | clustered
N_FRAMES=30
# ─────────────────────────────────────────────────────────────────────────────

EXP="${STRATEGY}_${N_FRAMES}"
GIGN_ROOT="../gign"   # path to cloned GIGN repo
SCRIPTS_DIR="${GIGN_ROOT}"

mkdir -p logs

# Activate GIGN environment

cd "${GIGN_ROOT}"
export PYTHONUNBUFFERED=1

echo "===== Step 1: Build directory structure ====="
python step1_build_gign_data.py --strategy ${STRATEGY} --n_frames ${N_FRAMES}

echo "===== Step 2: Preprocessing (pocket + rdkit) ====="
python step2_preprocessing_plas.py --exp ${EXP} --distance 5 --workers 8

echo "===== Step 3: Build graphs ====="
python step3_build_graphs_plas.py --exp ${EXP} --distance 5 --workers 8

echo "===== Step 4: Train ====="
python step4_train_plas.py \
    --exp ${EXP} \
    --epochs 600 \
    --patience 100 \
    --bs 128 \
    --eval_bs 256 \
    --lr 5e-4 \
    --num_workers 4 \
    --resume

echo "===== Done ====="

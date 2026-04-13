source /home/YongweiZhang/miniconda3/etc/profile.d/conda.sh
conda activate fastwam

# 测试激活了哪个 conda 环境
which python
python -c "import sys; print(sys.executable)"

# 预创建 libero 配置文件，避免交互式 input() 在无终端环境中报 EOFError
LIBERO_CFG_DIR="$HOME/.libero"
LIBERO_CFG_FILE="$LIBERO_CFG_DIR/config.yaml"
if [ ! -f "$LIBERO_CFG_FILE" ]; then
    LIBERO_ROOT="/home/YongweiZhang/Github_Projects/LIBERO/libero/libero"
    mkdir -p "$LIBERO_CFG_DIR"
    cat > "$LIBERO_CFG_FILE" <<EOF
assets: ${LIBERO_ROOT}/assets
bddl_files: ${LIBERO_ROOT}/bddl_files
benchmark_root: ${LIBERO_ROOT}
datasets: ${LIBERO_ROOT}/../datasets
init_states: ${LIBERO_ROOT}/init_files
EOF
    echo "Created libero config: $LIBERO_CFG_FILE"
else
    echo "Libero config exists: $LIBERO_CFG_FILE"
fi
python -c "from libero.libero import benchmark; print('libero ok')"

exec python experiments/libero/run_libero_manager.py \
  task=libero_uncond_2cam224_1e-4 \
  ckpt=./checkpoints/fastwam_release/libero_uncond_2cam224.pt \
  EVALUATION.dataset_stats_path=./checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json \
  MULTIRUN.num_gpus=8
export CUDA_VISIBLE_DEVICES="0,1,2,3"

# studio-ousia/luke-japanese-large
# ソフトラベルの追加
# ラベルスムージング0.9
# (CV: 0.756, LB: 0.702)
python src/train008.py \
    --model_name studio-ousia/luke-japanese-large \
    --run_name train022 \
    --k_fold 10 \
    --max_length 256 \
    --batch_size 16 \
    --epochs 100 \
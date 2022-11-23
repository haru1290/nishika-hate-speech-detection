export CUDA_VISIBLE_DEVICES="0,1,2,3"

# studio-ousia/luke-japanese-large
# ソフトラベルの追加
# ラベルスムージング0.1
# (CV: 0.749, LB: 0.763)
python src/train006.py \
    --model_name studio-ousia/luke-japanese-large \
    --run_name train020 \
    --k_fold 10 \
    --max_length 256 \
    --batch_size 16 \
    --epochs 100 \
export CUDA_VISIBLE_DEVICES="0,1,2,3"

# studio-ousia/luke-japanese-large-lite
# ソフトラベルの追加
# (CV: 0.750, LB: 0.741)
python src/train004.py \
    --model_name studio-ousia/luke-japanese-large-lite \
    --run_name train017 \
    --k_fold 10 \
    --max_length 256 \
    --batch_size 16 \
    --epochs 100 \
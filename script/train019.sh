export CUDA_VISIBLE_DEVICES="0,1,2,3"

# studio-ousia/luke-japanese-large
# ソフトラベルの追加
# (CV: ?, LB: ?)
python src/train004.py \
    --model_name studio-ousia/luke-japanese-large \
    --run_name train019 \
    --k_fold 10 \
    --max_length 256 \
    --batch_size 32 \
    --epochs 100 \
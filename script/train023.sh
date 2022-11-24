export CUDA_VISIBLE_DEVICES="0,1,2,3"

# studio-ousia/luke-japanese-large
# ソフトラベルの追加
# ラベルスムージング0.01
# (CV: 0.695, LB: ?)
python src/train009.py \
    --model_name studio-ousia/luke-japanese-large \
    --run_name train023 \
    --k_fold 10 \
    --max_length 256 \
    --batch_size 16 \
    --epochs 100 \
    
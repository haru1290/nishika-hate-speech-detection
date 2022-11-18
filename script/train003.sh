export CUDA_VISIBLE_DEVICES="0,1,2,3"

# (CV: ?, LB: ?)
python src/train002.py \
    --model_name studio-ousia/luke-japanese-large \
    --run_name train003 \
    --k_fold 15 \
    --max_length 256 \
    --batch_size 16 \
    --epochs 100 \
    --label_smoothing_factor 0.2 \
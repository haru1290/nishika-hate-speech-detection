export CUDA_VISIBLE_DEVICES="0"

# luke-japanese-large-lite (k=14)
# (CV: 0.348, LB: ?)
python src/train001.py \
    --model_name studio-ousia/luke-japanese-large-lite \
    --run_name train009 \
    --k_fold 14 \
    --max_length 256 \
    --batch_size 16 \
    --epochs 100 \
    --label_smoothing_factor 0.2 \
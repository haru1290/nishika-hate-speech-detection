export CUDA_VISIBLE_DEVICES="0,1,2,3"

# luke-japanese-large-lite (k=14)
# (CV: 0.764, LB: ?)
python src/train001.py \
    --model_name studio-ousia/luke-japanese-large-lite \
    --run_name train010 \
    --k_fold 14 \
    --max_length 256 \
    --batch_size 16 \
    --epochs 100 \
    --label_smoothing_factor 0.2 \
export CUDA_VISIBLE_DEVICES="0,1,2,3"

# luke-japanese-large-lite
# (CV: ?, LB: ?)
python src/train001.py \
    --model_name studio-ousia/luke-japanese-large-lite \
    --run_name train008 \
    --k_fold 10 \
    --max_length 256 \
    --batch_size 16 \
    --epochs 100 \
    --label_smoothing_factor 0.2 \
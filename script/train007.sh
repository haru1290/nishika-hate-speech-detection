export CUDA_VISIBLE_DEVICES="0,1,2,3"

# luke-japanese-largeでfoldを減らして実験10fold
# (CV: 0.767, LB: ?)
python src/train001.py \
    --model_name studio-ousia/luke-japanese-large \
    --run_name train007 \
    --k_fold 10 \
    --max_length 256 \
    --batch_size 16 \
    --epochs 100 \
    --label_smoothing_factor 0.2 \
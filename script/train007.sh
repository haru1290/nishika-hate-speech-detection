export CUDA_VISIBLE_DEVICES="0,1,2,3"

# 追加学習のluke8000
# (CV: 0.?, LB: ?)
python src/train001.py \
    --model_name ./models/luke-pretraining/checkpoint-8000 \
    --run_name train007 \
    --k_fold 15 \
    --max_length 256 \
    --batch_size 16 \
    --epochs 100 \
    --label_smoothing_factor 0.2 \
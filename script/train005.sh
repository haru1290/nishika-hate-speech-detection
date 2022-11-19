export CUDA_VISIBLE_DEVICES="0,1,2,3"

# 追加学習のluke20000
# (CV: 0.739, LB: ?)
python src/train001.py \
    --model_name ./models/luke-pretraining/checkpoint-20000 \
    --run_name train005 \
    --k_fold 15 \
    --max_length 256 \
    --batch_size 16 \
    --epochs 100 \
    --label_smoothing_factor 0.2 \
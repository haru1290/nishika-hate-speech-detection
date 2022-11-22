export CUDA_VISIBLE_DEVICES="0,1,2,3"

# 検証をepochからstepに変更
# studio-ousia/luke-japanese-large-lite
# (CV: 0.764, LB: 0.729)
python src/train003.py \
    --evaluation_strategy steps \
    --logging_strategy steps \
    --save_strategy steps \
    --early_stopping_patience 10 \
    --model_name studio-ousia/luke-japanese-large-lite \
    --run_name train013 \
    --k_fold 8 \
    --max_length 256 \
    --batch_size 16 \
    --epochs 100 \
    --label_smoothing_factor 0.2 \
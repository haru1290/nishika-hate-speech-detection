export CUDA_VISIBLE_DEVICES="1,2,3,4"

# 検証をepochからstepに変更
# (CV: 0.786, LB: 0.752)
python src/train003.py \
    --evaluation_strategy steps \
    --logging_strategy steps \
    --save_strategy steps \
    --early_stopping_patience 10 \
    --model_name studio-ousia/luke-japanese-large \
    --run_name train004 \
    --k_fold 15 \
    --max_length 256 \
    --batch_size 16 \
    --epochs 100 \
    --label_smoothing_factor 0.2 \
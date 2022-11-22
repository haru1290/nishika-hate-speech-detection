export CUDA_VISIBLE_DEVICES="0,1,2,3"

# 検証をepochからstepに変更
# studio-ousia/luke-japanese-large-lite
# ソフトラベルの追加
# FocalLoss追加
# (CV: 0.773, LB: 0.726)
python src/train005.py \
    --evaluation_strategy steps \
    --logging_strategy steps \
    --save_strategy steps \
    --early_stopping_patience 10 \
    --model_name studio-ousia/luke-japanese-large-lite \
    --run_name train016 \
    --k_fold 10 \
    --max_length 256 \
    --batch_size 16 \
    --epochs 100 \
    
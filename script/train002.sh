export CUDA_VISIBLE_DEVICES="0,1,2,3"

# ベースラインのmax_lengthを検証256
# (CV: 0.760, LB: 0.755)
python src/train001.py \
    --model_name studio-ousia/luke-japanese-large \
    --run_name train002 \
    --k_fold 15 \
    --max_length 256 \
    --batch_size 16 \
    --epochs 100 \
    --label_smoothing_factor 0.2 \
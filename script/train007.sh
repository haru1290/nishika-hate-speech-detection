export CUDA_VISIBLE_DEVICES="0,1,2,3"

# luke-japanese-largeでfoldを減らして実験10fold
# 学習できていないfoldが出現，おそらく性能が悪いのでLBは省略
# (CV: 0.646, LB: ?)
python src/train001.py \
    --model_name studio-ousia/luke-japanese-large \
    --run_name train007 \
    --k_fold 10 \
    --max_length 256 \
    --batch_size 16 \
    --epochs 100 \
    --label_smoothing_factor 0.2 \
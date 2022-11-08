# k=10 (LB:0.629)
# k=15 (LB:?)
python src/train.py \
    --model_name ku-nlp/roberta-base-japanese-char-wwm \
    --batch_size 16
    --epochs 100
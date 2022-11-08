# k=10 (LB:0.629)
# python src/train.py \
#     --model_name ku-nlp/roberta-base-japanese-char-wwm \
#     --batch_size 16
#     --epochs 100

# k=15 (LB:0.629)
python src/train.py \
    --model_name ku-nlp/roberta-base-japanese-char-wwm \
    --batch_size 32
    --learning_rate 2e-5
    --epochs 100
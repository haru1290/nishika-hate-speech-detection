# k=10 (LB:0.629)
# k=15 (LB:0.648)
# k=20 (LB:0.654)
# k=30 (LB:0.651)
# python src/train.py \
#     --model_name ku-nlp/roberta-base-japanese-char-wwm \
#     --batch_size 16 \
#     --epochs 100 \

# k=20 (LB: 0.670)
# python src/train.py \
#     --model_name studio-ousia/luke-japanese-base \
#     --batch_size 16 \
#     --epochs 100 \

python src/train.py \
    --model_name megagonlabs/transformers-ud-japanese-electra-base-discriminator \
    --batch_size 16 \
    --epochs 100 \

python src/submission.py
# Using all GPUs
export CUDA_VISIBLE_DEVICES="0,1,2,3"

# k=10 (LB:0.629)
# k=15 (LB:0.648)
# k=20 (LB:0.654)
# k=30 (LB:0.651)
# python src/train.py \
#     --k_fold 20 \
#     --model_name ku-nlp/roberta-base-japanese-char-wwm \
#     --batch_size 16 \
#     --epochs 100 \

# k=20 (LB: 0.670)
# python src/train.py \
#     --k_fold 20 \
#     --model_name studio-ousia/luke-japanese-base \
#     --batch_size 16 \
#     --epochs 100 \

# k=20 (LB: 0.657)
# python src/train.py \
#     --k_fold 20 \
#     --model_name megagonlabs/transformers-ud-japanese-electra-base-discriminator \
#     --batch_size 16 \
#     --epochs 100 \

# ls=0.15 (LB: 0.686)
# ls=0.2 (LB: 0.700)
# ls=0.25 (LB: 0.686)
# python src/train.py \
#     --k_fold 20 \
#     --model_name studio-ousia/luke-japanese-base \
#     --batch_size 16 \
#     --epochs 100 \
#     --label_smoothing_factor 0.2 \

# k=15 (LB: ?)
# k=20 (LB: 0.737)
# k=25 (LB: 0.750)
# python src/train.py \
#     --k_fold 25 \
#     --model_name studio-ousia/luke-japanese-large \
#     --batch_size 16 \
#     --epochs 100 \
#     --label_smoothing_factor 0.2 \

# Focal_MultiLabel_Loss
# γ=0.1 (LB: 0.758)
# python src/train.py \
#     --k_fold 25 \
#     --model_name studio-ousia/luke-japanese-large \
#     --batch_size 16 \
#     --epochs 100 \
#     --label_smoothing_factor 0.2 \

# max_length=128 (CV: 0.772, LB: ?)
# max_length=256 (CV: 0.760, LB: 0.755)
python src/train.py \
    --k_fold 15 \
    --model_name studio-ousia/luke-japanese-large \
    --max_length 128 \
    --batch_size 16 \
    --epochs 100 \
    --label_smoothing_factor 0.2 \

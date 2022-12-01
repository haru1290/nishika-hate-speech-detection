default:
	export CUDA_VISIBLE_DEVICES="0,1,2,3"
	python src/train.py \
		--model_name studio-ousia/luke-japanese-large \
		--run_name luke-large-japanese \
		--k_fold 10 \
		--max_length 256 \
		--batch_size 16 \
		--epochs 100 \
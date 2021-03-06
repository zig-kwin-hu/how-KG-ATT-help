python main.py \
	--seed 77777 \
	--lr 3e-5 --batch_size_per_gpu 128 --max_epoch 15 \
	--max_length 100 \
	--mode CM \
	--dataset nyt \
	--entity_marker --ckpt_to_load None \
	--train_prop 1 \
	--bag_size 30 \
	--entity_embedding_load_path ../../data/nyt/entity_embedding.npy \
	--kg_method TransE_re \
	--direct_feature \
	--freeze_entity

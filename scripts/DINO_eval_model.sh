python main_eval_model.py \
  --output_dir logs/eva \
	-c config/DINO_self_training/DINO_4scale.py \
	--eval --resume 'checkpoint_best_regular.pth' \
	--options dn_scalar=100 embed_init_tgt=TRUE \
	dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
	dn_box_noise_scale=1.0

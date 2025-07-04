export CUDA_VISIBLE_DEVICES=0 && python main.py \
	--output_dir logs/xView2DOTA/test -c config/DINO_self_training/DINO_4scale.py \
	--options dn_scalar=100 embed_init_tgt=TRUE \
	dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
	dn_box_noise_scale=1.0
#


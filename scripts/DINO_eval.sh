python main_eval.py \
  --output_dir logs/eva \
	-c config/DINO_self_training/DINO_4scale.py \
	--eval --resume '/data2/NCUT/个人文件夹/HJH/北理项目/域适应目标检测/第四篇source-free遥感目标检测-主动学习/4.7服务器代码—最终实验/7.DINO_self_training_with_AT_原版teacher_样本分析+自适应阈值+挑选样本同步输入+dinov2+加载保存的dinov2特征图+自适应阈值原型伪标签挖掘+特征原型对比学习_最终版代码2/logs/xView2DOTA_消融实验/最终实验结果/B4_自适应0.5_老师模型状态_加入中间层结果模型512_短边800长边800_随机挑选5con1.0_GMM4_6类三背景无重叠_求均值+原型对齐0.1改+backbonesim1.0原版/best_ema_model.pth' \
	--options dn_scalar=100 embed_init_tgt=TRUE \
	dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
	dn_box_noise_scale=1.0

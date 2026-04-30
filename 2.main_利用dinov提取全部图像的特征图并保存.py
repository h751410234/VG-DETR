# Copyright (c) 2022 IDEA. All Rights Reserved.
# ------------------------------------------------------------------------
import argparse

import torch
from torch.utils.data import DataLoader
import util.misc as utils
from datasets import build_dataset
import os
import dinov2
import numpy as np

#------自训练增添--------
torch.multiprocessing.set_sharing_strategy('file_system')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--config_file',default='config/DINO_self_training/DINO_4scale.py', type=str, required=False)
    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--fix_size', default = True, type=bool, required=False)
    # training parameters
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--note', default='',
                        help='add some notes to the experiment')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--pretrain_model_path', help='load from other checkpoint')
    parser.add_argument('--finetune_ignore', type=str, nargs='+')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--find_unused_params', action='store_true')

    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--save_log', action='store_true')
    parser.add_argument('--masks', default=False)
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='number of distributed processes')
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--amp', action='store_true',
                        help="Train with mixed precision")

    return parser

parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
args = parser.parse_args()


dataset_train_strong_aug = build_dataset(image_set='train', args=args, strong_aug=True)
sampler_train = torch.utils.data.RandomSampler(dataset_train_strong_aug)
batch_sampler_train = torch.utils.data.BatchSampler(sampler_train,1, drop_last=False)
data_loader_train = DataLoader(dataset_train_strong_aug, batch_sampler=batch_sampler_train,
                                       collate_fn=utils.collate_fn_self_training, num_workers=args.num_workers)


#--build dinov2
dinov2_model = dinov2.vit_large()

#添加使用的dinov2模型路径
checkpoint = torch.load(
    '', #dinov2_vitl14_pretrain.pth
    map_location='cpu')

print("Load pre-trained checkpoint from: %s" % ('../pretrained_models/dinov2_vitl14_pretrain.pth'))
# load pre-trained model
msg = dinov2_model.load_state_dict(checkpoint, strict=False)
print(msg)
dinov2_model.float().eval().cuda()
dinov2_preprocess = dinov2.transform([784, 784])  # 784  #输入图像大小
for child in dinov2_model.children():
    for param in child.parameters():
        param.requires_grad = False


#注意请修改coco_self_training.py ，取消水平随机翻转的增广，以保证特征图不旋转
if __name__ == "__main__":
    save_dir = ''  #提取的特征图保存路径
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():  #
        for idx, (samples, targets, strong_samples, img_no_normalize_weak, img_no_normalize_strong, flipped_flags) in enumerate(
                data_loader_train):
           # print(flipped_flags)
            img_no_normalize_weak_processed = dinov2_preprocess(img_no_normalize_weak[0]).cuda().unsqueeze(0)
            # 模型前向传播
            _, feature_maps = dinov2_model(img_no_normalize_weak_processed) #b,c,h,w

            # 保存特征
            img_id = targets[0]['image_id'].detach().cpu().numpy()[0]
            dinov2_features_numpy = feature_maps.detach().cpu().numpy()
            save_np_path = os.path.join(save_dir, f"{img_id}.npy")
            np.save(save_np_path, dinov2_features_numpy, allow_pickle=True)

            print('处理完', idx)

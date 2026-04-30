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
from torch.utils.data import DataLoader, DistributedSampler,Subset
from torchvision.ops import box_iou



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


def get_img_id(targets):
    cache_img_id_list = []
    for t in targets:
        id_name = int(t['image_id'].cpu().numpy())
        cache_img_id_list.append(id_name)
    return cache_img_id_list

# ========================================================
# 一些辅助函数
# ========================================================
def hflip_boxes(boxes, normalized=True):
    """
    水平翻转 bounding boxes.
    boxes: (N,4) Tensor,  (x0,y0,x1,y1).
    normalized=True  → x,y∈[0,1]; 否则为像素坐标。
    """
    flipped = boxes.clone()
    if normalized:
        flipped[:, 0] = 1.0 - boxes[:, 2]  # x0' = 1 - x1
        flipped[:, 2] = 1.0 - boxes[:, 0]  # x1' = 1 - x0
    else:
        print('错误，坐标应该被归一化处理')
    return flipped

def filter_overlap_bg_any_class(orig_targets, flipped_targets, iou_thr=0.0):
    """
    仅按 IoU 过滤：
    - 与原始目标（任意类别）出现任何交叠(> iou_thr) 就丢弃
    - 不做类别判断
    """
    if flipped_targets["boxes"].numel() == 0:
        return None                 # 没有候选框

    # 计算交叠 —— 完全不看类别
    ious = box_iou(flipped_targets["boxes"], orig_targets["boxes"])  # (M,N)
    keep_mask = (ious.max(dim=1).values <= iou_thr)                  # True ⇒ 无重叠

    if keep_mask.sum() == 0:
        return None

    return {
        "boxes":  flipped_targets["boxes"][keep_mask],
        "labels": flipped_targets["labels"][keep_mask],  # 原标签保留，之后再 +3
    }

def create_data_loader(selected_indices_npy_path):
    dataset_train_strong_aug = build_dataset(image_set='train', args=args, strong_aug=True)
    #-------------划分数据集----------------------
    # --------根据选择的样本，划分为两个data_loader-------------------
    # use_data_loader是原始数据集对象
    # selected_indices 是标注图像序列编号
    # 加载保存的 image_id 数组，假设保存的是整数类型的 id
    selected_image_ids = np.load(selected_indices_npy_path, allow_pickle=True)
    # 如果需要转换为 Python 的 int 类型，确保后续判断正确
    selected_image_ids = [int(x) for x in selected_image_ids]

    #dataset 为 CocoDetection 类型，并且每个样本可以通过 dataset.coco.imgs 得到图像信息
    id_to_idx = {img_id: idx for idx, img_id in enumerate(dataset_train_strong_aug.ids)}

    ordered_indices = []
    # 遍历每个 image_id 并判断是否存在于 id_to_idx 中
    for img_id in selected_image_ids:
        if img_id in id_to_idx:
            ordered_indices.append(id_to_idx[img_id])
        else:
            print(f"警告：image_id {img_id} 不存在于数据集索引映射中，将跳过！")
    # 生成完整数据集的索引列表
    all_indices = list(range(len(dataset_train_strong_aug)))
    # 生成“其他”部分索引，即不在 ordered_indices 中的样本
    set_ordered = set(ordered_indices)
    remaining_indices = [idx for idx in all_indices if idx not in set_ordered]

    # 使用 Subset 构造两个不重复的子集
    subset_ordered = Subset(dataset_train_strong_aug, ordered_indices)
    subset_ordered = Subset(dataset_train_strong_aug, remaining_indices)

    # -----------------------------------------------------------

    sampler_train_ordered = torch.utils.data.RandomSampler(subset_ordered)


    batch_sampler_train_ordered = torch.utils.data.BatchSampler(
        sampler_train_ordered, 1, drop_last=False)


    data_loader_train_ordered = DataLoader(subset_ordered, batch_sampler=batch_sampler_train_ordered,
                                   collate_fn=utils.collate_fn_self_training, num_workers=args.num_workers)
    return data_loader_train_ordered


def create_prototype_for_VFM(data_loader_train_ordered,dinov2_feateure_dir,save_npy_path):
    from models.dino.prototype_utils import get_class_wise_roi_feature_from_vit, get_GMM_prototype
    #----------------提取原型----------------------------
    #---dinov2 Feature
    features_dir = dinov2_feateure_dir
    #---
    cache_features = []
    cache_labels = []
    #---
    from util import box_ops
    #注意关闭水平翻转，防止标注 和 特征图不匹配
    for idx, (samples_ordered, targets_ordered, strong_samples_ordered,img_no_normalize_weak_ordered,img_no_normalize_strong_ordered,flipped_flags)in enumerate(data_loader_train_ordered):

        image_id_ordered = get_img_id(targets_ordered)
        targets_ordered = [{k: v.to('cuda') for k, v in t.items()} for t in targets_ordered]
        dino_feature_ordereds = []
        pooled_features_background  = None
        for idx in image_id_ordered:
            targets_ordered = targets_ordered[0]
            np_feature_path_ordered = os.path.join(features_dir,f'{idx}.npy')
            dino_feature_ordered = np.load(np_feature_path_ordered, allow_pickle=True)
            dino_feature_ordered = torch.from_numpy(dino_feature_ordered).cuda()
            if flipped_flags[0]: #标注翻转了
                # ❶ 计算翻转后的 boxes（与特征图宽度 56 对齐即可；你若用归一化坐标就设 normalized=True）
                flipped_boxes = hflip_boxes(targets_ordered["boxes"],  normalized=True)

                # 构造 flipped_targets
                flipped_targets = {
                    "boxes": flipped_boxes,
                    "labels": targets_ordered["labels"]
                }
                # ❷ 过滤掉与原目标重叠的框
                bg_targets = filter_overlap_bg_any_class(targets_ordered, flipped_targets, iou_thr=0.0)

                # ❸ 若仍有纯背景框，则提取背景特征
                if bg_targets is not None:
                    pooled_features_background, all_labels_background, _ = get_class_wise_roi_feature_from_vit(
                        dino_feature_ordered, bg_targets, scores_flag=False)
                # flip along the width dimension (last dimension, W)
                dino_feature_ordered = dino_feature_ordered.flip(-1)
            dino_feature_ordereds.append(dino_feature_ordered)
        dino_feature_ordereds = torch.cat(dino_feature_ordereds, dim=0)  #[1,1024,56,56]
        #2.提取目标特征
        pooled_features, all_labels,_ = get_class_wise_roi_feature_from_vit(dino_feature_ordereds,targets_ordered,scores_flag = False)
        if pooled_features_background is not None: #加入背景
            #加入背景特征
            all_labels_background = all_labels_background + (num_classes - 1) # 从目标类别数后面开始分配标签
            cache_features.append(pooled_features_background)
            cache_labels.append(all_labels_background)
        #加入目标特征
        cache_features.append(pooled_features)
        cache_labels.append(all_labels)


    cache_features = torch.cat(cache_features, dim=0)
    cache_labels = torch.cat(cache_labels, dim=0)

    #-------得到最终原型 GMM--------------
    # 扩展背景类别数量等同于目标类别数了 ，假设为 3类，则生成6类原型（3类目标 + 3类背景）
    prototype_class_wise_vit,unique_labels = get_GMM_prototype(cache_features,cache_labels,class_num = (num_classes-1) * 2,GMM_component_num = 4,prototype_channel = 1024)
    save_path = save_npy_path
    prototype_class_wise_vit = prototype_class_wise_vit.detach().cpu().numpy()
    np.save(save_path, prototype_class_wise_vit)

if __name__ == "__main__":
    selected_indices_npy_path = ''  #第一步选择的标注数据idx
    dinov2_feateure_dir = ''        #第二步提取的dinov2特征存放路径
    save_npy_path_prototype = 'X.npy'      #原型保存文件路径 + 文件名称
    num_classes = 4                        #数据集类别数，对应于config.py，实际目标数 + 1 （例如xView2DOTA，要求则为4，否则需要生成原型对应的索引）
    data_loader_train_ordered = create_data_loader(selected_indices_npy_path)
    create_prototype_for_VFM(data_loader_train_ordered,dinov2_feateure_dir,save_npy_path_prototype)

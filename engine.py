# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""

import math
import os
import sys
from typing import Iterable

from util.utils import slprint, to_device

import torch

import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
# ----自训练加入---------
from models.dino.dino import PostProcess
import numpy as np
from models.dino.self_training_utils import get_img, get_pseudo_label_via_threshold, deal_pesudo_label, \
    rescale_pseudo_targets, show_pesudo_label_with_gt, spilt_output, \
    get_valid_output
from models.dino.prototype_utils import get_class_wise_roi_feature_from_vit
import torch.nn.functional as F
from util import box_ops
from models.dino.prototype_utils import loss_contrast


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    wo_class_error=False, lr_scheduler=None, args=None, logger=None, ema_m=None):
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    _cnt = 0
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header, logger=logger):

        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast(enabled=args.amp):
            if need_tgt_for_training:
                outputs = model(samples, targets)
            else:
                outputs = model(samples)

            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict

            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # amp backward function
        if args.amp:
            optimizer.zero_grad()
            scaler.scale(losses).backward()
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            # original backward function
            optimizer.zero_grad()
            losses.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        if args.onecyclelr:
            lr_scheduler.step()
        if args.use_ema:
            if epoch >= args.ema_epoch:
                ema_m.update(model)

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!" * 5)
                break

    if getattr(criterion, 'loss_weight_decay', False):
        criterion.loss_weight_decay(epoch=epoch)
    if getattr(criterion, 'tuning_matching', False):
        criterion.tuning_matching(epoch)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if getattr(criterion, 'loss_weight_decay', False):
        resstat.update({f'weight_{k}': v for k, v in criterion.weight_dict.items()})
    return resstat


class CombinedDataLoader:
    def __init__(self, dl_ordered, dl_remaining):
        """
        初始化 CombinedDataLoader，并计算最大迭代次数，
        以数据量较多的 DataLoader 为准。
        """
        self.dl_ordered = dl_ordered
        self.dl_remaining = dl_remaining
        self.max_iter = max(len(dl_ordered), len(dl_remaining))

    def __iter__(self):
        # 分别创建两个 DataLoader 的迭代器
        iter_ordered = iter(self.dl_ordered)
        iter_remaining = iter(self.dl_remaining)

        # 循环 max_iter 次，每次分别获取两个 dataloader 的 batch，
        # 当某个迭代器用尽时重新迭代以保证使用全部数据。
        for _ in range(self.max_iter):
            try:
                batch_ordered = next(iter_ordered)
            except StopIteration:
                iter_ordered = iter(self.dl_ordered)
                batch_ordered = next(iter_ordered)

            try:
                batch_remaining = next(iter_remaining)
            except StopIteration:
                iter_remaining = iter(self.dl_remaining)
                batch_remaining = next(iter_remaining)

            # 假设每个 batch 均返回 3 个元素：(samples, targets, strong_samples)
            # 合并两个 batch 的返回内容，注意元组可以直接相加
            yield batch_ordered + batch_remaining

    def __len__(self):
        # 返回最大迭代次数，保证 metric_logger.log_every 可以计算总数
        return self.max_iter

def get_img_id(targets):
    cache_img_id_list = []
    for t in targets:
        id_name = int(t['image_id'].cpu().numpy())
        cache_img_id_list.append(id_name)
    return cache_img_id_list



# =============加入self-training版本======================
def train_one_epoch_with_self_training(model: torch.nn.Module, teacher_model: torch.nn.Module,
                                       criterion: torch.nn.Module,
                                       data_loader_train_ordered: Iterable, data_loader_train_remaining: Iterable,
                                       optimizer: torch.optim.Optimizer,
                                       device: torch.device, epoch: int, max_norm: float = 0,
                                       wo_class_error=False, lr_scheduler=None, args=None, logger=None, ema_m=None,
                                       pseudo_weight=None, AdaTh=None):
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    model.train()
    criterion.train()
    postprocessors = {'bbox': PostProcess()}

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    _cnt = 0

    # ----记录损失使用
    cache_loss_array = []
    cache_self_training_loss_array = []
    cache_adath_list = []
    teacher_model.eval()

    # 创建组合 dataloader 对象
    combined_loader = CombinedDataLoader(data_loader_train_ordered, data_loader_train_remaining)
    #---dinov2 Feature
    features_dir = args.dinv2_features_dir
    #----load proto
    np_feature_path_prototype = args.dinv2_prototype_numpy
    dino_prototype_feature = np.load(np_feature_path_prototype, allow_pickle=True)
    dino_prototype_feature = torch.from_numpy(dino_prototype_feature).cuda()  #[class,GMM_C,channel:1024]

    for _idx, (samples_ordered, targets_ordered, strong_samples_ordered,img_no_normalize_weak_ordered,img_no_normalize_strong_ordered,flipped_flags_ordered,
              samples_remaining, targets_remaining, strong_samples_remaining,img_no_normalize_weak_remaining,img_no_normalize_strong_remaining,flipped_flags_remaining) in enumerate(
        metric_logger.log_every(combined_loader, print_freq, header, logger=logger)):
        """
        samples: 测试图像（弱增广）图片
        source_labels:源域标注  （不使用）
        target_labels:目标域标注，用于获取图像增广信息，不参与损失计算 #（不使用）
        samples_strong_aug: 源域 + 目标域（强增广）图片
        """
        #
        # if _idx not in _idx_list:
        #     continue


       # print(targets_ordered)
        samples = samples_remaining.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets_remaining]

        # -------------0.得到强增广样本-------------
        if strong_samples_remaining is not None:
            samples_strong_aug = strong_samples_remaining.to(device)

        # -------------1.使用teacher_model对目标域图像(weak aug)推理，得到推理结果-------------
        # 1.1得到输入图像（unlabel） weak_aug
        samples_img = get_img(samples)
        #  samples_strong_aug_img = get_img(samples_strong_aug)，用于debug
        # 1.2推理得到检测结果
        with torch.no_grad():
            predict_results,_,_,_ = teacher_model(samples_img)  # 老师模型预测全部结果，使用weak_aug，img

        # 1.3处理推理结果
        orig_unlabel_target_sizes = torch.stack([torch.tensor([1, 1]).to(device) for i in range(len(targets))],
                                                dim=0)  # 保证坐标归一化
        predict_results = postprocessors['bbox'](predict_results, orig_unlabel_target_sizes,
                                                 not_to_xyxy=True)  # [{'scores': s, 'labels': l, 'boxes': b,'logits':logits}]

        # -------------2.使用阈值得到伪标签-------------
        if AdaTh:  # 自适应阈值
            threshold_high = AdaTh.masking_th(predict_results) #作为阈值门限(学生)
            cache_adath_list.append(threshold_high)
            threshold_low = np.asarray([0.3] * (args.num_classes))
        else:
            # 2.1 设置各类别置信度阈值
            threshold_low = np.asarray([args.pseudo_label_threshold] * (args.num_classes))
            threshold_high = np.asarray([1.0] * (args.num_classes))

        # 2.2 卡阈值以得到可靠伪标签
        idx_list, labels_dict, boxes_dict, scores_dcit, logits_dict = get_pseudo_label_via_threshold(predict_results,
                                                                                                     threshold=threshold_low)

        # 2.3 将伪标签处理为计算损失的格式
        pseudo_labels = deal_pesudo_label(targets, idx_list, labels_dict, boxes_dict, scores_dcit, logits_dict)

        # 2.4 对pseudo_label坐标 进行  nms 和 比例缩放，保证与target_labels格式一致
        pseudo_labels = rescale_pseudo_targets(samples_img, pseudo_labels)

        pseudo_weights = None


        # -------------3.学生模型对全图进行推理(使用强增广数据)-------------
        # 找到对应的vit feature
        image_id_remaining = get_img_id(targets)
        dino_feature_remainings = []
        for idx,idx_name in enumerate(image_id_remaining):
            np_feature_path_remaining = os.path.join(features_dir,f'{idx_name}.npy')
            dino_feature_remaining = np.load(np_feature_path_remaining, allow_pickle=True)
            dino_feature_remaining = torch.from_numpy(dino_feature_remaining).cuda() #[b,c,h,w]
            if flipped_flags_remaining[idx]:
                # flip along the width dimension (last dimension, W)
                dino_feature_remaining = dino_feature_remaining.flip(-1)
            dino_feature_remainings.append(dino_feature_remaining)
        dino_feature_remainings = torch.cat(dino_feature_remainings, dim=0)
        dino_feature_remainings.requires_grad = False
        #
        with torch.cuda.amp.autocast(enabled=args.amp):
            outputs,sim_loss_remaining,class_prototypes_remaining,valid_class_map_remaining = model(samples_strong_aug , dinov2_features = dino_feature_remainings)

        loss_c_loss_remaining = loss_contrast(class_prototypes_remaining,dino_prototype_feature,valid_class_map_remaining)
        # 4.2 根据伪标签得到对应有效的预测结果 +  格式处理
        valid_outputs, pseudo_labels = get_valid_output(outputs, pseudo_labels, idx_list)

        if len(idx_list) < 1:  # 没有伪标签则跳过
            continue

        #
        # #---------根据原型过滤伪标签-----------------
        final_pseudo_labels = []
        for batch_idx,pseudo_label in enumerate(pseudo_labels):
            idx = idx_list[batch_idx]
            dino_feature_remainings_according_p = dino_feature_remainings[idx].unsqueeze(0)
            pooled_features, all_labels,all_scores = get_class_wise_roi_feature_from_vit(dino_feature_remainings_according_p,pseudo_label)
            pooled_feats = pooled_features.mean(dim=(2, 3))  #roi特征
            #pooled_feats = pooled_features.sum(dim=(2, 3))  #roi特征


            #---判断一致性结果-------
            current_thresholds = torch.from_numpy(threshold_high).to(dino_feature_remainings_according_p.device)
            threshold_for_class = current_thresholds[all_labels]

            easy_mask = all_scores >= threshold_for_class  # [N]
            hard_mask  = all_scores < threshold_for_class  # [N]，布尔掩码

            # 过滤所有信息
            pooled_feats = pooled_feats[hard_mask]
            all_labels = all_labels[hard_mask]

            filtered_hard_mask = torch.zeros_like(hard_mask)  # 最终一致性 hard mask
            if pooled_feats.shape[0] > 0:
                #--------计算相似性--------------
                pooled_feats = F.normalize(pooled_feats, dim=1)  # L2归一化更适合计算余弦相似度
                # 假设 dino_prototype_feature 是 [num_classes, num_components, D]
                num_classes, num_components, D = dino_prototype_feature.shape
                # 展平所有原型：→ [num_classes * num_components, D]
                proto_flat = dino_prototype_feature.view(-1, D)  # [M, D]
                proto_flat = F.normalize(proto_flat, dim=1)  # 与特征同样归一化

                # 相似度矩阵：pooled_feats [N, D] × proto_flat.T [D, M] → [N, M]
                sim_matrix = pooled_feats @ proto_flat.t()  # [N, M]

                # 找最近原型：最大相似度索引
                sim_v, idx_nn = sim_matrix.max(dim=1)  # [N]


                #------------滤除相似度小于0.5的样本
                sim_v_mask = (sim_v >= 0.5)
                # 映射回类别索引：M = num_classes * num_components
                pred_labels = idx_nn // num_components  # 对每个最近原型索引求整除得到对应的类别
                consistency = (pred_labels == all_labels)
                consistency = consistency & sim_v_mask
                # 将一致的标记回原始 mask 中
                filtered_hard_mask[hard_mask] = consistency  # 保留一致的低分难例


            # === Step 3: 合并 mask ===
            valid_mask = easy_mask | filtered_hard_mask  # 最终使用的伪标签
            # 根据mask得到最终伪标签
            cache_pseudo_label_dict = {}
            mask_key_list = ['labels','boxes','scores','logits']
            for k,v in pseudo_label.items():
                if k in mask_key_list:
                    v = v[valid_mask]
                cache_pseudo_label_dict[k] = v
            final_pseudo_labels.append(cache_pseudo_label_dict)

        # # 2.5 可视化pseudo_labels，用于debug
        # show_pesudo_label_with_gt(samples_img,final_pseudo_labels,targets,idx_list,samples_img,_idx)
        # #
        # continue


        # -------------4.学生模型对全图进行推理(使用强增广数据)-------------
        samples_ordered = samples_ordered.to(device)
        targets_ordered = [{k: v.to(device) for k, v in t.items()} for t in targets_ordered]
        # 找到对应的vit feature
        image_id_ordered = get_img_id(targets_ordered)
        dino_feature_ordereds = []
        for idx,idx_name in enumerate(image_id_ordered):
            np_feature_path_ordered = os.path.join(features_dir,f'{idx_name}.npy')
            dino_feature_ordered = np.load(np_feature_path_ordered, allow_pickle=True)
            dino_feature_ordered = torch.from_numpy(dino_feature_ordered).cuda()
            if flipped_flags_ordered[idx]:
                # flip along the width dimension (last dimension, W)
                dino_feature_ordered = dino_feature_ordered.flip(-1)
            dino_feature_ordereds.append(dino_feature_ordered)
        dino_feature_ordereds = torch.cat(dino_feature_ordereds, dim=0)
        dino_feature_ordereds.requires_grad = False
        #------
        with torch.cuda.amp.autocast(enabled=args.amp):
            outputs_ordered,sim_loss_ordered,class_prototypes_ordered,valid_class_map_ordered = model(samples_ordered,dinov2_features = dino_feature_ordereds)

        loss_c_loss_ordered = loss_contrast(class_prototypes_ordered,dino_prototype_feature,valid_class_map_ordered)

        # -------------5.计算损失(注意需要单独计算伪标签损失)-------------
        weight_dict = criterion.weight_dict
        # 5.2 监督域损失计算(for ordered domain)
        loss_dict_ordered = criterion(outputs_ordered, targets_ordered, weights=pseudo_weights)
        losses_ordered = sum(loss_dict_ordered[k] * weight_dict[k] for k in loss_dict_ordered.keys() if k in weight_dict)

        # for _remaining: 伪标签损失
        loss_dict_remaining = criterion(valid_outputs, final_pseudo_labels, weights=pseudo_weights)
        losses_remaining = sum(loss_dict_remaining[k] * weight_dict[k] for k in loss_dict_remaining.keys() if k in weight_dict)

        losses = losses_ordered + 1.0 * losses_remaining + 1.0 * (sim_loss_remaining + sim_loss_ordered) + 0.1 * (loss_c_loss_ordered + loss_c_loss_remaining)


        # 若没有损失，则置零
        if losses == 0:
            losses = torch.tensor(0)
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict_remaining)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # amp backward function
        if args.amp:
            optimizer.zero_grad()
            scaler.scale(losses).backward()
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            # original backward function
            optimizer.zero_grad()
            losses.backward()

            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        if args.onecyclelr:
            lr_scheduler.step()
        if args.use_ema:
            if epoch >= args.ema_epoch:
                ema_m.update(model)

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # ==============每个iteration，更新teacher模型==========================
        # 固定间隔更新教师模
        with torch.no_grad():
            student_model_state_dict = model.state_dict()
            teacher_model_state_dict = teacher_model.state_dict()
            for entry in teacher_model_state_dict.keys():
                teacher_param = teacher_model_state_dict[entry].clone().detach()
                student_param = student_model_state_dict[entry].clone().detach()
                new_param = (teacher_param * args.alpha_ema) + (student_param * (1. - args.alpha_ema))
                teacher_model_state_dict[entry] = new_param
            teacher_model.load_state_dict(teacher_model_state_dict)

        # ==========================================================
        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!" * 5)
                break

    if getattr(criterion, 'loss_weight_decay', False):
        criterion.loss_weight_decay(epoch=epoch)
    if getattr(criterion, 'tuning_matching', False):
        criterion.tuning_matching(epoch)

    # -----记录损失-------------
    if utils.is_main_process():
        cache_loss_array.append(losses.detach().cpu().numpy())
        cache_self_training_loss_array.append(losses.detach().cpu().numpy())
        cache_loss_mean = np.asarray(cache_loss_array).mean()
        cache_ssod_loss_mean = np.asarray(cache_self_training_loss_array).mean()
        with open(os.path.join(args.output_dir, 'loss_txt'), 'a') as f:
            f.write('sup_loss: %s , ssod_loss: %s \n' % (cache_loss_mean, cache_ssod_loss_mean))

        # ---------------------------
        with open(os.path.join(args.output_dir, 'th_txt'), 'a') as f2:  # 飞机、油罐、船
            f2.write('********************************\n')
            for line in cache_adath_list:
                line = [str(i) for i in line]
                f2.write('%s %s \n' % (str(epoch), (' ').join(line)))
        # -----------------------
    # ---------------------------

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if getattr(criterion, 'loss_weight_decay', False):
        resstat.update({f'weight_{k}': v for k, v in criterion.weight_dict.items()})
    return resstat


# ===================================================


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, wo_class_error=False,
             args=None, logger=None):
    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    useCats = True
    try:
        useCats = args.useCats
    except:
        useCats = True
    if not useCats:
        print("useCats: {} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!".format(useCats))
    coco_evaluator = CocoEvaluator(base_ds, iou_types, useCats=useCats)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    _cnt = 0
    output_state_dict = {}  # for debug only
    for idx, (samples, targets, _,_,_) in enumerate(metric_logger.log_every(data_loader, 10, header, logger=logger)):

        samples = samples.to(device)
      #  print(samples.shape)
        # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast(enabled=args.amp):
            outputs,_,_,_ = model(samples)

            loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        # [scores: [100], labels: [100], boxes: [100, 4]] x B
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}

        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

        if args.save_results:
            # res_score = outputs['res_score']
            # res_label = outputs['res_label']
            # res_bbox = outputs['res_bbox']
            # res_idx = outputs['res_idx']

            for i, (tgt, res, outbbox) in enumerate(zip(targets, results, outputs['pred_boxes'])):
                """
                pred vars:
                    K: number of bbox pred
                    score: Tensor(K),
                    label: list(len: K),
                    bbox: Tensor(K, 4)
                    idx: list(len: K)
                tgt: dict.

                """
                # compare gt and res (after postprocess)
                gt_bbox = tgt['boxes']
                gt_label = tgt['labels']
                gt_info = torch.cat((gt_bbox, gt_label.unsqueeze(-1)), 1)

                # img_h, img_w = tgt['orig_size'].unbind()
                # scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=0)
                # _res_bbox = res['boxes'] / scale_fct
                _res_bbox = outbbox
                _res_prob = res['scores']
                _res_label = res['labels']
                res_info = torch.cat((_res_bbox, _res_prob.unsqueeze(-1), _res_label.unsqueeze(-1)), 1)
                # import ipdb;ipdb.set_trace()

                if 'gt_info' not in output_state_dict:
                    output_state_dict['gt_info'] = []
                output_state_dict['gt_info'].append(gt_info.cpu())

                if 'res_info' not in output_state_dict:
                    output_state_dict['res_info'] = []
                output_state_dict['res_info'].append(res_info.cpu())

        _cnt += 1
        if args.debug:
            if _cnt % 15 == 0:
                print("BREAK!" * 5)
                break

    if args.save_results:
        import os.path as osp

        # output_state_dict['gt_info'] = torch.cat(output_state_dict['gt_info'])
        # output_state_dict['res_info'] = torch.cat(output_state_dict['res_info'])
        savepath = osp.join(args.output_dir, 'results-{}.pkl'.format(utils.get_rank()))
        print("Saving res to {}".format(savepath))
        torch.save(output_state_dict, savepath)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]

    return stats, coco_evaluator


@torch.no_grad()
def test(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, wo_class_error=False, args=None,
         logger=None):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    # if not wo_class_error:
    #     metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    # coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    final_res = []
    for samples, targets in metric_logger.log_every(data_loader, 10, header, logger=logger):
        samples = samples.to(device)

        # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        # loss_dict = criterion(outputs, targets)
        # weight_dict = criterion.weight_dict

        # # reduce losses over all GPUs for logging purposes
        # loss_dict_reduced = utils.reduce_dict(loss_dict)
        # loss_dict_reduced_scaled = {k: v * weight_dict[k]
        #                             for k, v in loss_dict_reduced.items() if k in weight_dict}
        # loss_dict_reduced_unscaled = {f'{k}_unscaled': v
        #                               for k, v in loss_dict_reduced.items()}
        # metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
        #                      **loss_dict_reduced_scaled,
        #                      **loss_dict_reduced_unscaled)
        # if 'class_error' in loss_dict_reduced:
        #     metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes, not_to_xyxy=True)
        # [scores: [100], labels: [100], boxes: [100, 4]] x B
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        for image_id, outputs in res.items():
            _scores = outputs['scores'].tolist()
            _labels = outputs['labels'].tolist()
            _boxes = outputs['boxes'].tolist()
            for s, l, b in zip(_scores, _labels, _boxes):
                assert isinstance(l, int)
                itemdict = {
                    "image_id": int(image_id),
                    "category_id": l,
                    "bbox": b,
                    "score": s,
                }
                final_res.append(itemdict)

    if args.output_dir:
        import json
        with open(args.output_dir + f'/results{args.rank}.json', 'w') as f:
            json.dump(final_res, f)

    return final_res

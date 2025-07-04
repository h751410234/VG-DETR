import torch
from torch import nn
import torch.nn.functional as F
from util import box_ops
from torchvision.ops import roi_align



def l2_normalize(x):
    return F.normalize(x, p=2, dim=-1)

@torch.no_grad()
def distributed_sinkhorn_wograd(Q, sinkhorn_iterations=3):
    """
    标准的 Sinkhorn 归一化：交替对行和列进行归一化
    Q: [M, K] 的非负矩阵（初始可以是 softmax 后的分配权重）
    """
    for _ in range(sinkhorn_iterations):
        # normalize rows
        Q = Q / (Q.sum(dim=1, keepdim=True) + 1e-12)
        # normalize cols
        Q = Q / (Q.sum(dim=0, keepdim=True) + 1e-12)
    return Q


def get_prototype_class_wise(pooled_features, num_components=1):
    N,C = pooled_features.shape
    # ===== 初始化当前批次原型 =====
    prototype_class_wise=torch.zeros(num_components, C).to(pooled_features.device)

    # ===== Sinkhorn-EM 聚类 =====
    logits = torch.randn(N, num_components, device=pooled_features.device)  # [N, num_components]
    logits = l2_normalize(logits)  # 对分配概率进行归一化
    q, _ = distributed_sinkhorn_wograd(logits)  # 使用 Sinkhorn 算法对特征分配到高斯分量

    # ===== 计算每个分量的均值 =====
    for k in range(num_components):
        component_mask = q[:, k].bool()  # 当前高斯分量的掩码
        if component_mask.sum() == 0:  # 如果当前分量没有样本，跳过
            continue
        else:
            # 提取分量特征并计算均值
            component_features = pooled_features[component_mask]  # [N, C]
            component_mean = component_features.mean(dim=0)  # [C]
            # 存储当前批次原型
            prototype_class_wise[k] = component_mean
    return prototype_class_wise

def loss_contrast(
    global_proto,   # [num_classes, num_components, C]
    class_proto,    # [num_classes, num_components, C]
    class_mask,     # 可以是 bool mask ([num_classes]) 或 class 索引列表 / tensor
):
    """
    基于 InfoNCE 的原型对比损失：
    - 只对 class_mask 中的类别计算损失
    - 视每个类别的每个成分为一个样本对，正例是相同成分的全局原型，所有其他全局原型都是负例
    """
    class_num = class_mask.shape[0]
    class_proto = class_proto[:class_num,:,:] #去除来自dinov2参考原型的背景类
    class_mask[-1] = False #忽视背景类的对比学习

    # (1) 归一化
    global_proto = F.normalize(global_proto, dim=2)  # [C 整体归一]
    class_proto  = F.normalize(class_proto,  dim=2)

    # (2) 解析 class_mask
    if not torch.is_tensor(class_mask):
        valid_cls = torch.tensor(class_mask, device=class_proto.device, dtype=torch.long)
    elif class_mask.dtype == torch.bool:
        valid_cls = torch.nonzero(class_mask, as_tuple=False).squeeze(1)
    else:
        valid_cls = class_mask

    # (3) 筛出有效类别的原型
    #    形状均变为 [num_valid, num_components, C]
    cp = class_proto[valid_cls]
    gp = global_proto[valid_cls]

    # (4) 展平成 M = num_valid * num_components 个样本
    #    P[i] 的正例是 G[i]
    M, K, C = cp.shape[0] * cp.shape[1], cp.shape[1], cp.shape[2]
    P = cp.reshape(-1, C)  # [M, C]
    G = gp.reshape(-1, C)  # [M, C]

    # (5) 计算相似性矩阵并加温度
    logits = torch.matmul(P, G.t())   # [M, M]

    # (6) 构造标签：第 i 行的正例是第 i 列
    targets = torch.arange(M, device=logits.device, dtype=torch.long)

    # (7) 交叉熵
    ce_loss = nn.CrossEntropyLoss()   # 修改位置: 在函数内部也可，每次都会复用
    loss = ce_loss(logits, targets)

    return loss



def get_prototype_class_wise_object_query(
    object_query_last_layer,  # [B, N, C]
    outputs_class,            # [B, N, num_classes] (未归一化 logits)
    num_classes,
    num_components=3,
    sinkhorn_iterations=3
):
    B, N, C = object_query_last_layer.shape

    # --- 修改位置1: 用 softmax 而不是 sigmoid + argmax ----
   # prob = F.softmax(outputs_class, dim=-1)      # [B, N, num_classes]
    prob = F.sigmoid(outputs_class)      # [B, N, num_classes]
    predicted_labels = torch.argmax(prob, dim=2) # [B, N]

    # 展平到 [B*N, C] 和 [B*N]
    outputs_target = object_query_last_layer.reshape(-1, C)
    predicted_flat = predicted_labels.flatten()

    # --- 修改位置2: 用 bool mask 表示哪些类别在本批次出现 ---
    valid_class_map = torch.bincount(predicted_flat, minlength=num_classes) > 0  # [num_classes], dtype=torch.bool

    # 初始化返回张量
    prototype_class_wise = torch.zeros(num_classes, num_components, C, device=outputs_target.device)

    # 对每个类别做多成分聚类
    for cls in range(num_classes):
        if not valid_class_map[cls]:
            continue

        # 筛出该类的所有特征
        cls_mask = (predicted_flat == cls)
        cls_features = outputs_target[cls_mask]  # [M_cls, C]
        if cls_features.size(0) == 0:
            continue

        # --- 修改位置3: 用 softmax 初始化 q，避免 L2-normalize 难以区分 ----
        raw_q = torch.randn(cls_features.size(0), num_components, device=cls_features.device)
        q = F.softmax(raw_q, dim=1)  # [M_cls, num_components]

        # --- 修改位置4: 调用新的 Sinkhorn 函数做平衡 ----
        q = distributed_sinkhorn_wograd(q, sinkhorn_iterations=sinkhorn_iterations)  # 归一化后的软分配矩阵

        # --- 修改位置7: 用软权重计算加权均值，而不是硬分配 ----
        for k in range(num_components):
            weights = q[:, k].unsqueeze(1)  # [M_cls, 1]
            if weights.sum() < 1e-6:
                continue
            # 加权均值
            mean_feat = (weights * cls_features).sum(dim=0) / (weights.sum() + 1e-12)
            # 归一化原型向量
            mean_feat = F.normalize(mean_feat, dim=0)
            prototype_class_wise[cls, k] = mean_feat
    return prototype_class_wise, valid_class_map


def get_class_wise_roi_feature_from_vit(vit_feature_maps, pseudo_labels, scores_flag=True):
    b, c, h, w = vit_feature_maps.shape

    # 1. 把 cxcywh ∈ [0,1] → xyxy (同样 ∈ [0,1])
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(pseudo_labels['boxes'])

    # 2. 缩放到特征图坐标系
    scale = vit_feature_maps.new_tensor([w, h, w, h])
    boxes_xyxy = boxes_xyxy * scale           # [N,4] float32/float16

    # 3. 拼 batch_idx
    batch_idx = boxes_xyxy.new_zeros((boxes_xyxy.size(0), 1))
    roi_boxes = torch.cat([batch_idx, boxes_xyxy], dim=1)  # [N,5]

    # 4. ROI Align
    pooled_feats = roi_align(
        vit_feature_maps, roi_boxes,
        output_size=(7, 7),
        spatial_scale=1.0,
        sampling_ratio=-1
    )  # [N, C, 7, 7]

    labels = pseudo_labels['labels']
    scores = pseudo_labels['scores'] if scores_flag else None
    return pooled_feats, labels, scores

from sklearn.cluster import KMeans
def get_prototype_class_wise_k_means(features, num_components=1):
    """
    使用KMeans从一组特征中提取num_components个原型中心

    Args:
        features: Tensor [N, C]，一个类别的所有ROI特征
        num_components: 想提取的原型数

    Returns:
        Tensor [num_components, C]
    """
    features = F.normalize(features, dim=1)  # L2归一化，提升聚类稳定性
    features_np = features.detach().cpu().numpy()  # 转为 numpy 用于KMeans

    kmeans = KMeans(n_clusters=num_components, n_init='auto', max_iter=100).fit(features_np)
    centroids = torch.tensor(kmeans.cluster_centers_, dtype=features.dtype, device=features.device)

    return centroids


def get_GMM_prototype(pooled_features, labels, class_num, GMM_component_num=1, prototype_channel=256):
    """
    利用KMeans提取每类的GMM-like原型（每类多中心表示）

    Args:
        pooled_features: Tensor [N, C, H, W] or [N, C, 1, 1]
        labels: Tensor [N]，对应每个ROI的类别
        class_num: 类别总数
        GMM_component_num: 每类要聚多少个原型
        prototype_channel: 特征维度（通常为C）

    Returns:
        prototype_class_wise_vit: [class_num, GMM_component_num, C]
        unique_labels: 实际出现的类别
    """
    prototype_class_wise_vit = torch.zeros(class_num, GMM_component_num, prototype_channel,
                                           device=pooled_features.device)
    unique_labels = labels.unique()

    for label_id in unique_labels.tolist():
        mask = (labels == label_id)
        features_cur_cat = pooled_features[mask]  # [N, C, H, W]

        if features_cur_cat.dim() == 4:
           # features_cur_cat = features_cur_cat.mean(dim=(2, 3))  # → [N, C]
            features_cur_cat = features_cur_cat.mean(dim=(2, 3))  # → [N, C]

        features_cur_cat = F.normalize(features_cur_cat, dim=1)  # 归一化防止尺度不一致

        # 正常使用KMeans提取类内原型
        class_prototypes_vit = get_prototype_class_wise_k_means(features_cur_cat, num_components=GMM_component_num)
        prototype_class_wise_vit[label_id] = class_prototypes_vit

    return prototype_class_wise_vit, unique_labels






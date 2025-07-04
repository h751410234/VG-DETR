import torch

class AdaptiveThreshold():
    """
    reference SAT in FreeMatch
    """
    def __init__(self, num_classes, th=0.5,momentum=0.999):
        self.num_classes = num_classes
        self.m = momentum
        self.p_model = torch.ones((self.num_classes)) * th# / self.num_classes
        self.label_hist = torch.ones((self.num_classes))  # / self.num_classes
        self.time_p = self.p_model.mean()
        self.class_th = self.p_model.mean()
        self.clip_thresh = True

    @torch.no_grad()
    def update(self,scores_result,probs_result):


        #1.计算  time_p
        mean_scores = scores_result.mean()
        self.time_p = self.time_p * self.m + (1 - self.m) * mean_scores  # 更新  根据预测的平均置信度 更新全局,单一数值

        if self.clip_thresh:  # 默认不使用
            self.time_p = torch.clip(self.time_p, 0.0, 0.95)

        #2，计算 计算 p_model，针对每一类的阈值
        probs_result = probs_result[:,:]  #暂时不去除背景类
        max_probs, max_idx = torch.max(probs_result, dim=-1, keepdim=True)

        #calculate_probs each class
        cache_probs = torch.zeros(probs_result.shape[-1])
        for c in range(probs_result.shape[-1]):
            c_idx = (max_idx == c)
            c_max_probs = max_probs[c_idx]
            if int(c_max_probs.shape[0]) > 0: #exist class in img
                c_max_probs_mean = c_max_probs.mean(dim = 0)
                cache_probs[c] = c_max_probs_mean
            else:
                cache_probs[c] = self.p_model[c]
        self.p_model = self.p_model * self.m + (1 - self.m) * cache_probs.to(self.p_model.device) #计算每类阈值


    @torch.no_grad()
    def masking_th(self, predict_unlabel_result_list):
        if not self.p_model.is_cuda:
            self.p_model = self.p_model.to(predict_unlabel_result_list[0]['scores'].device)
        if not self.label_hist.is_cuda:
            self.label_hist = self.label_hist.to(predict_unlabel_result_list[0]['scores'].device)
        if not self.time_p.is_cuda:
            self.time_p = self.time_p.to(predict_unlabel_result_list[0]['scores'].device)
        #1.处理预测结果
        pur_scores_all = []
        pur_probs_all = []

        for pul in predict_unlabel_result_list:
            pur_scores_all.append(pul['scores'].detach())
            pur_probs_all.append(pul['logits'].detach())

        pur_scores_all = torch.cat(pur_scores_all,dim = 0)
        pur_probs_all = torch.cat(pur_probs_all,dim = 0)

        pur_scores_all = pur_scores_all.detach()
        pur_probs_all =  torch.softmax(pur_probs_all.detach(), dim=-1)

        #去除阈值小于0.1的预测结果
        probs_result = pur_probs_all[pur_scores_all >= 0.1]
        scores_result = pur_scores_all[pur_scores_all >= 0.1]  # 考虑加入

        #2.根据预测结果更新自适应阈值权重
        if len(probs_result) > 0:
            self.update(scores_result,probs_result)

            #3.得到各个类别自适应阈值大小
            mod = self.p_model / torch.max(self.p_model, dim=-1)[0]
            self.class_th = self.time_p * mod
            return self.class_th.cpu().numpy()
        else:
            return self.class_th.cpu().numpy()

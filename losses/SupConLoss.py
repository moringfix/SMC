import torch
from torch import nn

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, contrast_mode='all'):
        super(SupConLoss, self).__init__()
        self.contrast_mode = contrast_mode

    def forward(self, features, labels=None, mask=None, temperature = 0.07, device = None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        # device = (torch.device('cuda')
        #           if features.is_cuda
        #           else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)  # （B，特征，其他）

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device) # 仅对角线为1
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1) # (B, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device) # NOTE 逐元素比较，把相同的元素置为1，得到（B, B）的mask矩阵，挑选cluster级别的正样本
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1] # 每个样本有多少个视角
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0) # （B * n_views, 特征），所有视角也扩张成为样本
        if self.contrast_mode == 'one': # 只有一个视角
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all': # 所有视角
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            temperature)    # 计算相似度,基于自己/全视角
        
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True) # 
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
    
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        

        # loss
        loss = - mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
    

# 加权loss

class SupConLossWeighted(nn.Module):
    """
    SupConLoss + 相似度权重:
      pos_weight = 'sim'   -> w_ij = (cos_sim) ** sim_power
                  'uniform' (退化成原版)
    """
    def __init__(self, contrast_mode='all',
                 pos_weight='sim',  # 或 'uniform'
                 sim_power=1.0):   # γ
        super().__init__()
        self.contrast_mode = contrast_mode
        self.pos_weight = pos_weight
        self.sim_power = sim_power

    def forward(self, features, labels=None, mask=None,
                temperature=0.07, device=None):

        if len(features.shape) < 3:
            raise ValueError("`features` must be [B, V, ...]")
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        B = features.shape[0]

        # ---------- 构造 supervised mask ----------
        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            # SimCLR 情形: 仅自己另一视图算正
            mask = torch.eye(B, dtype=torch.float32, device=device)
        elif labels is not None:
            labels = labels.view(-1, 1)
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        # ---------- 展平视图 ----------
        V = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  # (B·V, D)

        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]            # (B, D)
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature          # (B·V, D)
            anchor_count = V
        else:
            raise ValueError(f"Unknown mode: {self.contrast_mode}")

        # ---------- 相似度 logits ----------
        logits = torch.matmul(anchor_feature, contrast_feature.T) / temperature  # (A, B·V)
        # 数值稳定
        logits_max, _ = logits.max(dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # ---------- 扩展 mask 到所有 anchor×contrast ----------
        mask = mask.repeat(anchor_count, V)                      # (A, B·V)

        # 去掉 self‑contrast (对角线)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(B * anchor_count, device=device).view(-1, 1),
            0
        )
        mask = mask * logits_mask

        # ---------- 计算对数概率 ----------
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # ---------- ★ 加权平均正样本 ★ ----------
        if self.pos_weight == 'uniform':
            # 原版: 等权
            pos_weight = mask
        elif self.pos_weight == 'sim':
            # 用相似度^γ 当权重，仅保留正位置，其余置 0
            sim = torch.matmul(anchor_feature, contrast_feature.T)  # 未/τ 的 cos_sim
            pos_weight = (sim.clamp(min=0) ** self.sim_power) * mask
        elif self.pos_weight == 'hard':
            # 用(1-相似度)^γ 当权重，仅保留正位置，其余置 0
            sim = torch.matmul(anchor_feature, contrast_feature.T)  # 未/τ 的 cos_sim
            pos_weight = (((1-sim).clamp(min=0)) ** self.sim_power) * mask
        else:
            raise ValueError("pos_weight must be 'uniform' or 'sim'")

        # 避免除 0
        weight_sum = pos_weight.sum(1, keepdim=True) + 1e-12
        weighted_log_prob = (pos_weight * log_prob).sum(1, keepdim=True) / weight_sum
        loss = -weighted_log_prob.squeeze()

        # ---------- reshape 回原 batch 取均值 ----------
        loss = loss.view(anchor_count, B).mean()
        return loss
import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
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
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class SoftNearestNeighbours(torch.nn.Module):
    def __init__(self, x, y, temperature_init: float = 0.1, eps: float = 1e-8, raise_on_inf=False, raise_on_single_point_for_class=False, cosine=False):
        super(SoftNearestNeighbours, self).__init__()
        weights = torch.zeros(1) + 1/temperature_init
        self.weights = torch.nn.Parameter(weights)
        self.x = x
        self.y = y
        self.n = self.x.shape[0]
        if cosine:
            self.distances = self.pairwise_cosine_similarity(x)
        else:
            self.distances = torch.cdist(x, x, p=2)
        self.eps = eps
        self.raise_on_inf = raise_on_inf
        self.raise_on_single_point_for_class = raise_on_single_point_for_class

    @staticmethod
    def pairwise_cosine_similarity(x):
        dots = x @ x.T
        inv_magnitude = torch.sqrt(1 / dots.diag())
        cosines = (dots * inv_magnitude).T * inv_magnitude
        return cosines

    def get_bool_mask(self, idx, class_subset=False):
        if class_subset:
            mask = self.y == self.y[idx]
        else:
            mask = torch.ones(self.n, dtype=bool)
        mask[idx] = False
        return mask

    def forward(self):
        fracs = torch.zeros(self.n)
        exponents = torch.exp(-self.distances * self.weights[0])
        ignore_idx = []
        for i in range(self.n):
            top = exponents[i, self.get_bool_mask(i, True)]
            bot = exponents[i, self.get_bool_mask(i, False)]
            fracs[i] = torch.log(top.sum()) - torch.log(bot.sum()+self.eps)
            if len(top) == 0:
                if self.raise_on_single_point_for_class:
                    raise ValueError("No other points with the same label in batch")
                else:
                    ignore_idx.append(i)
            elif fracs[i].isnan().item():
                raise ValueError("Nan detected in loss calculation")
            elif fracs[i].isinf().item() and self.raise_on_inf:
                raise ValueError("inf detected in loss calculation, if optimising try reducing the lr")
        ignore_mask = torch.ones(self.n, dtype=torch.bool)
        for i in ignore_idx:
            ignore_mask[i] = False
        fracs = fracs[ignore_mask]
        loss = (-1 / len(fracs)) * fracs.sum()
        return loss

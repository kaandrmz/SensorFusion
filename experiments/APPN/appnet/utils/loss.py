from torch import nn
import torch
import math
from torch.nn import functional as F
import numpy as np
np.set_printoptions(threshold=np.inf)
from .lovasz_loss import lovasz_softmax



class OhemCrossEntropy(nn.Module):
    """Ohem cross entropy

    Args:
        ignore_label (int): ignore label
        thres (float): maximum probability of prediction to be ignored
        min_kept (int): maximum number of pixels to be consider to compute loss
        weight (torch.Tensor): weight for cross entropy loss
    """

    def __init__(self, ignore_label=-1, thres=0.70, min_kept=100000,gamma = 2, weight=None):
        super(OhemCrossEntropy, self).__init__()
        self.thresh = thres
        self.min_kept = max(1, min_kept)
        self.gamma = gamma
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label, reduction="none")

    def forward(self, score, target,c_sore=None,ohem=False,focal=False,**kwargs):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.interpolate(input=score, size=(h, w), mode="bilinear", align_corners=False)
        if c_sore is not None:
            uh, uw = c_sore.size(1), c_sore.size(2)
            if uh != h or uw != w:
                c_sore = F.upsample(input=c_sore, size=(h, w), mode="bilinear",align_corners=False)
            pixel_losses = (self.criterion(score, target)*c_sore).contiguous().view(-1)
        else:
            pixel_losses = (self.criterion(score, target)).contiguous().view(-1)
        if c_sore is not None:

            mask = (target.contiguous().view(-1) != self.ignore_label) | (c_sore.contiguous().view(-1) != 0)
        else:
            mask = target.contiguous().view(-1) != self.ignore_label
        if focal:
            pt = torch.exp(-pixel_losses)
            pixel_losses = (1-pt)**self.gamma*pixel_losses
        # target_onehot = F.one_hot(target,num_class=score.size(1))
        # print(pixel_losses.shape)
        pred = F.softmax(score,dim=1)
        tmp_target = target.clone()
        tmp_target[tmp_target == self.ignore_label] = 0
        pred = pred.gather(1, tmp_target.unsqueeze(1))
        pred, ind = (
            pred.contiguous()
            .view(
                -1,
            )[mask]
            .contiguous()
            .sort()
        )
        #more attention on best confidengce point

        # threshold = max(min_value, self.thresh)
        min_value = pred[min(self.min_kept, pred.numel() - 1)]
        threshold = max(min_value, self.thresh)

        pixel_losses = pixel_losses[mask][ind]
        if c_sore is not None:
            return pixel_losses.mean()


        if ohem:
            pixel_losses = pixel_losses[pred < threshold]
            return pixel_losses.mean()
        else:
            return pixel_losses.mean()


        # more attention on best confidengce point
        # min_value = pred[pred.numel() - 2001]
        # pixel_losses_1000 = pixel_losses[pred > min_value]
        # return pixel_losses.mean()+pixel_losses_1000.mean()


def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """

    C = tensor.size(1)  # 获得图像的维数
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)  # 将维数的数据转换到第一位
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)


class DiceLoss(nn.Module):
    def __init__(self, ignore_label=-1,epsilon = 1e-5):
        super(DiceLoss, self).__init__()
        self.epsilon = epsilon
        self.ignore_label = ignore_label

    def forward(self, output, target):
        ph, pw = output.size(2), output.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            output = F.interpolate(input=output, size=(h, w), mode="bilinear", align_corners=False)
        tmp_target = target.clone()
        tmp_target[tmp_target == self.ignore_label] = output.size(1)
        target_onhot = F.one_hot(tmp_target, num_classes=(output.size(1) + 1))
        target_onhot = target_onhot[:, :, :, :output.size(1)].permute(0, 3, 1, 2)
        assert output.size() == target_onhot.size(), "'input' and 'target' must have the same shape"
        output = F.softmax(output, dim=1)
        output = flatten(output)
        target = flatten(target_onhot)
        # intersect = (output * target).sum(-1).sum() + self.epsilon
        # denominator = ((output + target).sum(-1)).sum() + self.epsilon

        intersect = (output * target).sum(-1)
        denominator = (output + target).sum(-1)
        dice = intersect / denominator
        dice = torch.mean(dice)
        return 1 - dice
        # return 1 - 2. * intersect / denominator

class RectifyCrossEntropy(nn.Module):
    """Ohem cross entropy

    Args:
        ignore_label (int): ignore label
        thres (float): maximum probability of prediction to be ignored
        min_kept (int): maximum number of pixels to be consider to compute loss
        weight (torch.Tensor): weight for cross entropy loss
    """

    def __init__(self, ignore_label=-1, thres=0.5, min_kept=100000, weight=None):
        super(RectifyCrossEntropy, self).__init__()
        self.thresh = thres
        self.min_kept = max(1, min_kept)
        self.ignore_label = ignore_label
        self.criterion = nn.BCELoss()

    def forward(self, score, target,uncertainty_score=None,**kwargs):
        #second method
        tmp_target = target.clone()
        tmp_target[tmp_target == self.ignore_label] = score.size(1)
        target_onhot = F.one_hot(tmp_target, num_classes=(score.size(1) + 1))
        target_onhot = target_onhot[:, :, :, :score.size(1)].permute(0, 3, 1, 2)
        target_distribution = torch.sum(target_onhot.float(),dim=(2,3))
        target_distribution = torch.where(target_distribution > 0,torch.ones_like(target_distribution),target_distribution)
        # target_distribution[target_distribution!=0] = 1
        # target_distribution = F.sigmoid(target_distribution)
        pred = F.sigmoid(score)
        # print(pred)
        # print(target_distribution)
        loss = self.criterion(pred,target_distribution)
        print(loss)


        return loss










        #original method
        # ph, pw = score.size(2), score.size(3)
        # h, w = target.size(1), target.size(2)
        # if ph != h or pw != w:
        #     score = F.upsample(input=score, size=(h, w), mode="bilinear", align_corners=False)
        # if uncertainty_score is not None:
        #     uh, uw = uncertainty_score.size(1), uncertainty_score.size(2)
        #     if uh != h or uw != w:
        #         uncertainty_score = F.upsample(input=uncertainty_score, size=(h, w), mode="bilinear", align_corners=False)
        # # class_mask = torch.where(target==self.ignore_label,torch.zeros_like(target),torch.ones_like(target))
        # # print(class_mask.shape)
        # tmp_target = target.clone()
        # tmp_target[tmp_target == self.ignore_label] = score.size(1)
        # target_onhot = F.one_hot(tmp_target,num_classes=(score.size(1)+1))
        # # print(target_onhot.shape)
        # # print(target_onhot.shape)
        # mask = target_onhot[:,:,:,score.size(1)].unsqueeze(3).permute(0,3,1,2).clone()
        # target_onhot = target_onhot[:,:,:,:score.size(1)].permute(0,3,1,2)
        # # print(target_onhot.shape)
        # target_onhot = target_onhot.float()
        # # print(target_onhot)
        # # pred = F.softmax(score, dim=1)
        # # target_onhot = torch.where(class_mask.unsqueeze(1).expand_as(target_onhot)==0,pred,target_onhot.float())
        # # target_onhot = torch.where(target_onhot==1,torch.full_like(target_onhot,0.9),torch.full_like(target_onhot,0.1/(score.size(1)-1)))
        # # print(target_onhot)
        # pred = (1 - mask) * (score.softmax(1))
        # class_pred = torch.sum(pred,dim=(2,3))
        # class_labelnum = torch.sum(target_onhot,dim=(2,3))
        # class_labelnum = F.sigmoid(class_labelnum-torch.mean(class_labelnum,0))
        # class_pred = F.sigmoid(class_pred-torch.mean(class_labelnum,0))
        # # print(class_labelnum)
        # # print(class_pred)
        # # class_labelnum = F.softmax(class_labelnum,dim=1)
        # # print(class_labelnum)
        # # class_labelnum = F.softmax(class_labelnum,dim=0)
        # # print(masked_pred.shape)
        # # class_pred = torch.sum(pred,dim=(2,3))
        # # class_pred = class_pred/(ratio.expand_as(class_pred.permute(1,0)).permute(1,0))
        # # class_labelnum = F.softmax(class_labelnum, dim=1)
        # # print(class_pred)
        # # class_pred = F.log_softmax(class_pred,dim=1)
        # # print(class_pred)
        #
        # # print(class_pred.sum(1))
        # # print(class_pred)
        # # print(pred.shape)
        # # print(target.shape)
        # if uncertainty_score is not None:
        #     # uncertrainty_map = uncertainty_score[uncertainty_score<max(torch.mean(1-uncertainty_score,dim=0),self.thresh)]
        #     # print(self.criterion(score, target).unsqueeze(1).shape)
        #     pixel_losses = (self.criterion(score, target).unsqueeze(1)*uncertainty_score).contiguous().view(-1)
        #
        # else:
        #     pixel_losses = self.criterion(class_pred, class_labelnum)
        # # del class_pred,class_mask,target_onhot,class_labelnum,masked_pred
        #
        # # mask = target.contiguous().view(-1) != self.ignore_label
        # # # print(pixel_losses.shape)
        # # tmp_target = target.clone()
        # # tmp_target[tmp_target == self.ignore_label] = 0
        # # pred = pred.gather(1, tmp_target.unsqueeze(1))
        # # pred, ind = (
        # #     pred.contiguous()
        # #     .view(
        # #         -1,
        # #     )[mask]
        # #     .contiguous()
        # #     .sort()
        # # )
        # # # min_value = pred[min(self.min_kept, pred.numel() - 1)]
        # # # threshold = max(min_value, self.thresh)
        # # #
        # # pixel_losses = pixel_losses[mask][ind]
        # # # if uncertainty_score is None:
        # # #     pixel_losses = pixel_losses[pred < threshold]
        # # print(pixel_losses)
        # # print(pixel_losses)
        # return pixel_losses
class IoUloss(nn.Module):
    def __init__(self, ignore_label=-1):
        super(IoUloss, self).__init__()
        # self.thresh = thres
        # self.min_kept = max(1, min_kept)
        self.ignore_label = ignore_label
        # self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label, reduction="none")

    def forward(self, score, target,**kwargs):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.upsample(input=score, size=(h, w), mode="bilinear", align_corners=False)
        with torch.no_grad():
            # num_classes = logits.size(1)
            label = target.clone().detach()
            ignore = label.eq(self.ignore_label)
            n_valid = ignore.eq(0).sum()
            label[ignore] = 0
            lb_pos, lb_neg = 1., 0.
            # pos_w ,neg_w= 1,0
            lb_one_hot = torch.empty_like(score).fill_(
                lb_neg).scatter_(1, label.unsqueeze(1), lb_pos).detach()
        score = score.argmax(1)
        # score[ind] = one
        # score[score!=1] = zero
        # loss = torch.sum((lb_one_hot - score) ** 2, dim=1)
        loss = (target - score)
        loss[ignore] = 0
        loss = loss[loss!=0]/loss[loss!=0]
        loss = loss.sum() / n_valid


        return loss
        # # error_mat  = torch.where(score!=target,torch.ones_like(score),torch.zeros_like(score))
        # # loss = error_mat.float().contiguous().view(-1)
        # loss = lovasz_softmax(score,target,classes='present', per_image=False, ignore=self.ignore_label)
        # return loss.mean()

class ERRORLoss(nn.Module):
    def __init__(self, ignore_label=-1):
        super(ERRORLoss, self).__init__()
        # self.thresh = thres
        # self.min_kept = max(1, min_kept)
        self.ignore_label = ignore_label
        self.criterion = nn.BCELoss()
    def forward(self, score,c_pred, target,**kwargs):
        ph, pw = score.size(2), score.size(3)
        ch,cw = c_pred.size(2),c_pred.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.upsample(input=score, size=(h, w), mode="bilinear", align_corners=False)
        if ch != h or cw != w:
            c_pred = F.upsample(input=c_pred, size=(h, w), mode="bilinear", align_corners=False)
        coarsepred = c_pred.argmax(1)
        score = score.squeeze(1)
        pred = F.sigmoid(score)
        # print(pred)
        # print(target)
        target_copy = torch.where(target.float()==self.ignore_label,coarsepred.float(),target.float())
        # print(target_copy)
        target_copy = torch.where(target_copy!=coarsepred,torch.ones_like(target_copy),torch.zeros_like(target_copy))
        # print(target_copy)
        loss = self.criterion(pred,target_copy)
        # print(loss)
        return loss

class CertaintyLoss(nn.Module):
    def __init__(self, ignore_label=-1):
        super(CertaintyLoss, self).__init__()
        self.ignore_label = ignore_label
    def forward(self, score,label):
        ph, pw = score.size(2), score.size(3)
        h, w = label.size(1), label.size(2)
        if ph != h or pw != w:
            score = F.upsample(input=score, size=(h, w), mode="bilinear", align_corners=False)
        loss = -torch.log(score.contiguous().view(-1))
        mask = label.contiguous().view(-1) != self.ignore_label
        loss = loss[mask]
        return loss.mean()


class FocalLoss(nn.Module):
    def __init__(self, ignore_label=-1,gamma=1,alpha=1):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_label = ignore_label
        self.critia = nn.NLLLoss(ignore_index=ignore_label, reduction="none")
    def forward(self, score, target):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.upsample(input=score, size=(h, w), mode="bilinear", align_corners=False)
        score = torch.log(score)
        ce_loss = self.critia(score,target).contiguous().view(-1)
        mask = target.contiguous().view(-1) != self.ignore_label
        ce_loss = ce_loss[mask]
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

class LabelSmoothLoss(nn.Module):
    '''
    This is the autograd version, you can also try the LabelSmoothSoftmaxCEV2 that uses derived gradients
    '''

    def __init__(self, lb_smooth=0.1, reduction='mean', ignore_index=-100):
        super(LabelSmoothLoss, self).__init__()
        self.lb_smooth = lb_smooth
        self.reduction = reduction
        self.lb_ignore = ignore_index
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, logits, label,key_weight=None):
        '''
        Same usage method as nn.CrossEntropyLoss:
            >>> criteria = LabelSmoothSoftmaxCEV1()
            >>> logits = torch.randn(8, 19, 384, 384) # nchw, float/half
            >>> lbs = torch.randint(0, 19, (8, 384, 384)) # nhw, int64_t
            >>> loss = criteria(logits, lbs)
        '''
        # overcome ignored label
        ph, pw = logits.size(2), logits.size(3)
        h, w = label.size(1), label.size(2)
        if ph != h or pw != w:
            logits = F.upsample(input=logits, size=(h, w), mode="bilinear", align_corners=False)
        logits = logits.float() # use fp32 to avoid nan
        with torch.no_grad():
            num_classes = logits.size(1)
            label = label.clone().detach()
            ignore = label.eq(self.lb_ignore)
            n_valid = ignore.eq(0).sum()
            label[ignore] = 0
            lb_pos, lb_neg = 1. - self.lb_smooth, self.lb_smooth / num_classes
            pos_w ,neg_w= 1,0
            lb_one_hot = torch.empty_like(logits).fill_(
                lb_neg).scatter_(1, label.unsqueeze(1), lb_pos).detach()
            # lb_one_hot_w = torch.empty_like(logits).fill_(
            #     neg_w).scatter_(1, label.unsqueeze(1), pos_w).detach()
        # 交叉熵损失
        # logs = self.log_softmax(logits)
        # loss = -torch.sum(logs * lb_one_hot, dim=1)
        # 均方差损失
        logs = F.softmax(logits,dim=1)
        # wight_loss = lb_one_hot_w*(lb_one_hot-logs)**2

        # loss = torch.sum(wight_loss, dim=1)
        loss = torch.sum((lb_one_hot-logs)**2, dim=1)
        if key_weight is not None:
            loss = key_weight*loss
        loss[ignore] = 0
        if self.reduction == 'mean':
            loss = loss.sum() / n_valid
        if self.reduction == 'sum':
            loss = loss.sum()

        return loss
class ClassLoss(nn.Module):
    def __init__(self):
        super(ClassLoss, self).__init__()

    def forward(self, logits, label, key_weight=None):
        prediction = torch.sigmoid(logits)
        loss = torch.sum((label - prediction) ** 2, dim=1)
        return loss.mean()

class AsymmetricLossOptimized(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self, gamma_neg=0, gamma_pos=0, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y
        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos
        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                with torch.no_grad():
                    # if self.disable_torch_grad_focal_loss:
                    #     torch._C.set_grad_enabled(False)
                    self.xs_pos = self.xs_pos * self.targets
                    self.xs_neg = self.xs_neg * self.anti_targets
                    self.asymmetric_w = torch.pow(
                        1 - self.xs_pos - self.xs_neg,
                        self.gamma_pos * self.targets +
                        self.gamma_neg * self.anti_targets,
                    )
                    # if self.disable_torch_grad_focal_loss:
                    #     torch._C.set_grad_enabled(True)
                self.loss *= self.asymmetric_w
            else:
                self.xs_pos = self.xs_pos * self.targets
                self.xs_neg = self.xs_neg * self.anti_targets
                self.asymmetric_w = torch.pow(
                    1 - self.xs_pos - self.xs_neg,
                    self.gamma_pos * self.targets +
                    self.gamma_neg * self.anti_targets,
                )
                self.loss *= self.asymmetric_w
        _loss = -self.loss.sum() / x.size(0)

        _loss = _loss / y.size(1)

        return _loss



class PixelContrastLoss(nn.Module):
    def __init__(self, ignorelabel = -1,temperature=0.1,base_temperature=0.07,max_samples=10000,max_views=200,min_pos_views = 20):
        super(PixelContrastLoss, self).__init__()

        self.temperature = temperature
        self.base_temperature = base_temperature

        self.ignore_label = ignorelabel

        self.max_samples = max_samples
        self.max_views = max_views
        self.min_pos_views = min_pos_views
    def _hard_anchor_sampling(self, X, y_hat, y):
        batch_size, feat_dim = X.shape[0], X.shape[-1]
        classes_index = []
        classes = []
        min_class_num = []
        total_classes = 0
        for ii in range(batch_size):
            this_y = y[ii]
            this_classes = torch.unique(this_y)
            this_classes = [x for x in this_classes if x != self.ignore_label]
            classes_index.extend(this_classes)
            # this_classes = [x for x in this_classes if ((this_y_hat == x) & (this_y == x)).nonzero().shape[0] >= self.min_pos_views]
            this_classes = [x for x in this_classes if (this_y == x).nonzero().shape[0] >= self.max_views]
            class_num = ((this_y == x).nonzero().shape[0] for x in this_classes)
            min_class_num.extend(class_num)
            classes.append(this_classes)
            total_classes += len(this_classes)
        # print(classes)
        if total_classes == 0:
            return None, None, None,None

        # n_view = self.max_samples // total_classes
        n_view = torch.min(torch.tensor(min_class_num),dim=0)[0]

        X_ = torch.zeros((total_classes, n_view+1, feat_dim), dtype=torch.float).cuda()
        y_ = torch.zeros(total_classes, dtype=torch.float).cuda()
        classes_index = torch.unique(torch.tensor(classes_index))
        X_ptr = 0
        for ii in range(batch_size):
            this_y = y[ii]
            this_classes = classes[ii]

            for cls_id in this_classes:
                # hard_indices = ((this_y_hat == cls_id) & (this_y != cls_id)).nonzero()
                easy_indices = (this_y == cls_id).nonzero()

                # num_hard = hard_indices.shape[0]
                num_easy = easy_indices.shape[0]
                # print(num_easy,num_hard)

                X_easy_featrue = X[ii,easy_indices,:].squeeze(1)
                X_easy_featrue_top2= torch.topk(X_easy_featrue, k=2, dim=1)[0]
                X_easy_featrue_certainty = (X_easy_featrue_top2[:, 0] - X_easy_featrue_top2[:, 1])
                # if num_easy < int(n_view/2):
                #     X_score, best_index = torch.topk(X_easy_featrue_certainty, num_easy)
                # else:
                #     X_score,best_index = torch.topk(X_easy_featrue_certainty,int(n_view/2))
                # X_easy_certainty_featrue = X_easy_featrue[best_index]
                # X_easy_certainty_featrue = torch.sum(X_easy_certainty_featrue,dim=0)/X_easy_certainty_featrue.shape[0]
                # X_easy_featrue = X_easy_certainty_featrue.unsqueeze(0)
                # if num_easy < int(n_view/2):
                #     X_score, best_index = torch.topk(X_easy_featrue_certainty, num_easy)
                # else:
                X_score,best_index = torch.topk(X_easy_featrue_certainty,int(n_view/2))
                X_easy_certainty_featrue = X_easy_featrue[best_index]
                X_easy_certainty_featrue = torch.sum(X_easy_certainty_featrue,dim=0)/X_easy_certainty_featrue.shape[0]
                X_easy_featrue = X_easy_certainty_featrue.unsqueeze(0)




                # if num_hard<=num_easy and num_hard>=n_view:
                #     num_hard_keep = num_hard
                #     num_easy_keep = n_view
                # elif num_hard>num_easy and num_easy>=n_view:
                #     num_easy_keep = num_easy
                #     num_hard_keep = n_view
                # elif num_hard < n_view:
                #     num_hard_keep = num_hard
                #     num_easy_keep = num_easy
                num_easy_keep = n_view
                # if  num_easy >= n_view:
                #     # num_hard_keep = n_view
                #     num_easy_keep = n_view
                # elif num_hard >= n_view:
                #     num_easy_keep = n_view
                #     num_hard_keep = n_view
                # elif num_easy >= n_view:
                #     num_hard_keep = n_view
                #     num_easy_keep = n_view
                # else:
                #     Log.info('this shoud be never touched! {} {} {}'.format(num_hard, num_easy, n_view))
                #     raise Exception

                # perm = torch.randperm(num_hard)
                # hard_indices = hard_indices[perm[:num_hard_keep]]
                perm = torch.randperm(num_easy)
                easy_indices = easy_indices[perm[:num_easy_keep]]
                # indices = torch.cat((hard_indices, easy_indices), dim=0)
                # print(indices.shape)
                X_[X_ptr, :, :] = torch.cat([X[ii, easy_indices, :].squeeze(1),X_easy_featrue],dim=0)
                y_[X_ptr] = cls_id
                X_ptr += 1

        return X_, y_,classes_index,n_view

    # def _contrastive(self, feats_, labels_):
    #     feats_ = feats_[:,:-1,:]
    #     anchor_num, n_view = feats_.shape[0], feats_.shape[1]
    #
    #     labels_ = labels_.contiguous().view(-1, 1)
    #     mask = torch.eq(labels_, torch.transpose(labels_, 0, 1)).float().cuda()
    #     contrast_count = n_view
    #     contrast_feature = torch.cat(torch.unbind(feats_, dim=1), dim=0)
    #
    #     anchor_feature = contrast_feature
    #     anchor_count = contrast_count
    #
    #     anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, torch.transpose(contrast_feature, 0, 1)),
    #                                     10)
    #     logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    #     logits = anchor_dot_contrast - logits_max.detach()
    #
    #     mask = mask.repeat(anchor_count, contrast_count)
    #     # neg_mask = 1 - mask
    #
    #     logits_mask = torch.ones_like(mask).scatter_(1,
    #                                                  torch.arange(anchor_num * anchor_count).view(-1, 1).cuda(),
    #                                                  0)
    #     mask = mask * logits_mask
    #     neg_mask = 1 - mask
    #     neg_logits = torch.exp(logits) * neg_mask
    #     neg_logits = neg_logits.sum(1, keepdim=True)
    #
    #     exp_logits = torch.exp(logits)
    #     log_prob = logits - torch.log(exp_logits + neg_logits)
    #     mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
    #
    #     # exp_logits = torch.exp(logits)*mask
    #     # log_prob = - torch.log(exp_logits/(exp_logits + neg_logits))
    #     # mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
    #
    #     loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
    #     loss = loss.mean()
    #
    #     return loss

    def _contrastive(self, feats_, labels_,classes_index,n_view):
        feats_ = feats_[:,:,classes_index]
        pos_all_feats = feats_[:,:-1,:]
        pos_feats = feats_[:,-1,:].unsqueeze(1).expand_as(pos_all_feats)
        pos_all_feats = pos_all_feats.reshape(-1,feats_.shape[2])
        pos_feats = pos_feats.reshape(-1,feats_.shape[2])
        anchor_num, n_view = feats_.shape[0], feats_.shape[1]-1
        sum_mean = anchor_num*n_view
        target = torch.ones_like(pos_feats[:,0])
        loss = torch.nn.CosineEmbeddingLoss(margin=(math.pi)/len(classes_index), reduction='none')

        # class_inter_loss = loss(F.sigmoid(neg_feats), F.sigmoid(pos_feats), target)
        class_inter_loss = loss(pos_all_feats,pos_feats,target).mean()
        # print(class_inter_loss)
        # pos_ = torch.softmax(feats_[:,-1,:],dim=0)
        pos_ = feats_[:, -1, :]
        pos_muti_each = torch.matmul(pos_, torch.transpose(pos_, 0, 1))
        pos_sub = torch.sqrt(torch.sum(pos_**2,dim=1))
        pos_sub = pos_sub.contiguous().view(-1, 1)
        pos_dis_sub_each = torch.matmul(pos_sub, torch.transpose(pos_sub, 0, 1))
        # print(class_intra_loss)
        labels_ = labels_.contiguous().view(-1, 1)
        mask = torch.eq(labels_, torch.transpose(labels_, 0, 1)).float().cuda()
        mask[mask==1] = -1
        mask[mask==0] = 1
        mask[mask==-1] = 0
        cos_ = (pos_muti_each/pos_dis_sub_each+1)*mask
        class_intra_loss = torch.sum(cos_.contiguous().view(-1,1),dim=0)/torch.sum(mask)

        # contrast_count = n_view
        # contrast_feature = torch.cat(torch.unbind(feats_, dim=1), dim=0)
        #
        # anchor_feature = contrast_feature
        # anchor_count = contrast_count
        #
        # anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, torch.transpose(contrast_feature, 0, 1)),
        #                                 10)
        # logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        # logits = anchor_dot_contrast - logits_max.detach()
        #
        # mask = mask.repeat(anchor_count, contrast_count)
        # # neg_mask = 1 - mask
        #
        # logits_mask = torch.ones_like(mask).scatter_(1,
        #                                              torch.arange(anchor_num * anchor_count).view(-1, 1).cuda(),
        #                                              0)
        # mask = mask * logits_mask
        # neg_mask = 1 - mask
        # neg_logits = torch.exp(logits) * neg_mask
        # neg_logits = neg_logits.sum(1, keepdim=True)
        #
        # exp_logits = torch.exp(logits)
        # log_prob = logits - torch.log(exp_logits + neg_logits)
        # mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        #
        # # exp_logits = torch.exp(logits)*mask
        # # log_prob = - torch.log(exp_logits/(exp_logits + neg_logits))
        # # mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        #
        # loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        # loss = loss.mean()

        return class_intra_loss+class_inter_loss

    def forward(self, feats, labels=None, predict=None):
        labels = labels.unsqueeze(1).float().clone()
        labels = torch.nn.functional.interpolate(labels,
                                                 (feats.shape[2], feats.shape[3]), mode='nearest')
        labels = labels.squeeze(1).long()
        assert labels.shape[-1] == feats.shape[-1], '{} {}'.format(labels.shape, feats.shape)

        batch_size = feats.shape[0]

        labels = labels.contiguous().view(batch_size, -1)
        predict = predict.contiguous().view(batch_size, -1)
        feats = feats.permute(0, 2, 3, 1)
        feats = feats.contiguous().view(feats.shape[0], -1, feats.shape[-1])
        feats_, labels_,classes_index,n_view= self._hard_anchor_sampling(feats, labels, predict)
        # print(feats_,labels_)
        if feats_ is None or labels_ is None or classes_index is None:
            return 0
        else:
            # loss = self._contrastive(feats_, labels_)
            loss = self._contrastive(feats_, labels_,classes_index,n_view)
            return loss
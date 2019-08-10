import torch
import torch.nn as nn
import torch.nn.functional as F

import loss.lovasz_losses as Ls

# https://github.com/marvis/pytorch-yolo2/blob/master/FocalLoss.py
# https://github.com/unsky/focal-loss
torch_ver = torch.__version__[:3]


class LovaszLoss(nn.Module):

    def __init__(self, only_present=False, per_image=False, ignore=255):

        super(LovaszLoss, self).__init__()
        self.only_present = only_present
        self.per_image = per_image
        self.ignore = ignore

    def forward(self, inputs, targets, index=None):

        if index == None:
            input_original = inputs
        else:
            input_original = inputs[index]

        h, w = targets.size(1), targets.size(2)
        if torch_ver == '0.4':
            input_scaled = F.upsample(input=input_original, size=(h, w), mode='bilinear', align_corners=True)
        else:
            input_scaled = F.upsample(input=input_original, size=(h, w), mode='bilinear')

        probas = F.softmax(input_scaled)
        return Ls.lovasz_softmax(probas, targets, self.only_present, self.per_image, self.ignore)


class LovaszLoss2d(nn.Module):
    """Lovasz Loss.

    See: https://arxiv.org/abs/1705.08790
    """

    def __init__(self):
        """Creates a `LovaszLoss2d` instance."""
        super().__init__()

    def forward(self, inputs, targets):

        h, w = targets.size(1), targets.size(2)
        if torch_ver == '0.4' and inputs.size() != targets.size():
            input_scaled = F.upsample(input=inputs, size=(h, w), mode='bilinear', align_corners=True)
        else:
            input_scaled = F.upsample(input=inputs, size=(h, w), mode='bilinear')

        N, C, H, W = input_scaled.size()
        masks = torch.zeros(N, C, H, W).to(targets.device).scatter_(1, targets.view(N, 1, H, W), 1)

        loss = 0.

        for mask, input in zip(masks.view(N, -1), input_scaled.view(N, -1)):

            max_margin_errors = 1. - ((mask * 2 - 1) * input)
            errors_sorted, indices = torch.sort(max_margin_errors, descending=True)
            labels_sorted = mask[indices.data]

            inter = labels_sorted.sum() - labels_sorted.cumsum(0)
            union = labels_sorted.sum() + (1. - labels_sorted).cumsum(0)
            iou = 1. - inter / union

            p = len(labels_sorted)
            if p > 1:
                iou[1:p] = iou[1:p] - iou[0:-1]

            loss += torch.dot(nn.functional.relu(errors_sorted), iou)

        return loss / N

class FocalLoss2d(nn.Module):

    def __init__(self, gamma=2, size_average=True):
        super(FocalLoss2d, self).__init__()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, logit, target, class_weight=[1, 5], type='softmax'):
        target = target.view(-1, 1).long()

        if type == 'sigmoid':
            if class_weight is None:
                class_weight = [1] * 2  # [0.5, 0.5]

            prob = F.sigmoid(logit)
            prob = prob.view(-1, 1)
            prob = torch.cat((1 - prob, prob), 1)
            select = torch.FloatTensor(len(prob), 2).zero_().cuda()
            select.scatter_(1, target, 1.)

        elif type == 'softmax':
            B, C, H, W = logit.size()
            if class_weight is None:
                class_weight = [1] * C  # [1/C]*C

            logit = logit.permute(0, 2, 3, 1).contiguous().view(-1, C)
            prob = F.softmax(logit, 1)
            select = torch.FloatTensor(len(prob), C).zero_().cuda()
            select.scatter_(1, target, 1.)

        class_weight = torch.FloatTensor(class_weight).cuda().view(-1, 1)
        class_weight = torch.gather(class_weight, 0, target)

        prob = (prob * select).sum(1).view(-1, 1)
        prob = torch.clamp(prob, 1e-8, 1 - 1e-8)
        batch_loss = - class_weight * (torch.pow((1 - prob), self.gamma)) * prob.log()

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss

        return loss


def sigmoid_cross_entropy_loss(prediction, label):
    label = label.long()
    mask = (label != 0).float()
    num_positive = torch.sum(mask).float()
    num_negative = mask.numel() - num_positive
    # print (num_positive, num_negative)
    mask[mask != 0] = num_negative / (num_positive + num_negative)
    mask[mask == 0] = num_positive / (num_positive + num_negative)
    cost = torch.nn.functional.binary_cross_entropy_with_logits(
        prediction.float(), label.float(), weight=mask, reduce=False)
    return torch.sum(cost)


class RobustFocalLoss2d(nn.Module):
    # assume top 10% is outliers
    def __init__(self, gamma=2, size_average=True):
        super(RobustFocalLoss2d, self).__init__()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, logit, target, class_weight=None, type='softmax'):
        target = target.view(-1, 1).long()

        if type == 'sigmoid':
            if class_weight is None:
                class_weight = [1] * 2  # [0.5, 0.5]

            prob = F.sigmoid(logit)
            prob = prob.view(-1, 1)
            prob = torch.cat((1 - prob, prob), 1)
            select = torch.FloatTensor(len(prob), 2).zero_().cuda()
            select.scatter_(1, target, 1.)

        elif type == 'softmax':
            B, C, H, W = logit.size()
            if class_weight is None:
                class_weight = [1] * C  # [1/C]*C

            logit = logit.permute(0, 2, 3, 1).contiguous().view(-1, C)
            prob = F.softmax(logit, 1)
            select = torch.FloatTensor(len(prob), C).zero_().cuda()
            select.scatter_(1, target, 1.)

        class_weight = torch.FloatTensor(class_weight).cuda().view(-1, 1)
        class_weight = torch.gather(class_weight, 0, target)

        prob = (prob * select).sum(1).view(-1, 1)
        prob = torch.clamp(prob, 1e-8, 1 - 1e-8)

        focus = torch.pow((1 - prob), self.gamma)
        # focus = torch.where(focus < 2.0, focus, torch.zeros(prob.size()).cuda())
        focus = torch.clamp(focus, 0, 2)

        batch_loss = - class_weight * focus * prob.log()

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss

        return loss


##  http://geek.csdn.net/news/detail/126833
class PseudoBCELoss2d(nn.Module):
    def __init__(self):
        super(PseudoBCELoss2d, self).__init__()

    def forward(self, logit, truth):
        z = logit.view(-1)
        t = truth.view(-1)
        loss = z.clamp(min=0) - z * t + torch.log(1 + torch.exp(-z.abs()))
        loss = loss.sum() / len(t)  # w.sum()
        return loss


def cross_entropy_loss(prediction, label):
    # print (label,label.max(),label.min())
    label = label.long()
    mask = (label != 0).float()
    num_positive = torch.sum(mask).float()
    num_negative = mask.numel() - num_positive
    # print (num_positive, num_negative)
    mask[mask != 0] = num_negative / (num_positive + num_negative)
    mask[mask == 0] = num_positive / (num_positive + num_negative)
    cost = F.binary_cross_entropy(
        F.sigmoid(prediction).float(), label.float(), weight=mask, reduction='mean')
    return cost  # torch.sum(cost)


class SegmentationLoss(nn.Module):
    '''nn.Module warpper for segmentation loss'''

    def __init__(self, c_weight=None, ce_weight=1, iou_weight=0, ignore_index=255, only_present=False, per_image=False):
        super(SegmentationLoss, self).__init__()
        self.ignore_index = ignore_index
        self.only_present = only_present
        self.per_image = per_image
        self.ce_weight = ce_weight
        self.iou_weight = iou_weight
        self.ignore_index = ignore_index

        if c_weight != None:
            c_weight = torch.FloatTensor(c_weight).cuda()

        self.cross_entropy = torch.nn.CrossEntropyLoss(weight=c_weight, ignore_index=self.ignore_index)

    def forward(self, out, target):
        h, w = target.size(1), target.size(2)

        # scaled_out = F.upsample(input=out, size=(h, w), mode='bilinear')

        ce_loss = self.cross_entropy(out, target)
        iou_loss = Ls.lovasz_softmax(F.softmax(out, dim=1), target, only_present=self.only_present,
                                     per_image=self.per_image, ignore=self.ignore_index)

        return self.ce_weight * ce_loss + self.iou_weight * iou_loss


class EdgeLoss(nn.Module):
    '''nn.Module warpper for segmentation loss'''

    def __init__(self):
        super(EdgeLoss, self).__init__()

    def forward(self, out, target):
        out = out.float()
        mask = (target != 0).float()
        num_positive = torch.sum(mask).float()
        num_negative = mask.numel() - num_positive
        # print (num_positive, num_negative)
        mask[mask != 0] = num_negative / (num_positive + num_negative)
        mask[mask == 0] = num_positive / (num_positive + num_negative)
        ce_loss = F.binary_cross_entropy(F.sigmoid(out), target.float(), weight=mask, reduction='mean')
        return ce_loss

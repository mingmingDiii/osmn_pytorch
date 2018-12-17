import torch
from torch import nn


class class_balanced_cross_entropy_loss(nn.Module):

    def __init__(self):
        super(class_balanced_cross_entropy_loss, self).__init__()

    def forward(self, logits,target):




        label_pos_num = torch.sum(target)
        label_neg_num = torch.sum(1.0-target)

        beta = label_neg_num/(label_pos_num+label_neg_num)

        pos_weight = beta*target
        neg_weight = (1.0-beta)*(1-target)

        bceloss = nn.BCEWithLogitsLoss(reduce=False,reduction='None')(logits,target)

        cb_bceloss = bceloss*pos_weight+bceloss*neg_weight
        cb_bceloss = torch.sum(cb_bceloss)/logits.size(0)


        return cb_bceloss




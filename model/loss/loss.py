import torch
import torch.nn as nn

class CMD(nn.Module):
    """
    Adapted from https://github.com/wzell/cmd/blob/master/models/domain_regularizer.py
    """

    def __init__(self):
        super(CMD, self).__init__()

    def forward(self, x1, x2, n_moments):
        mx1 = torch.mean(x1, 0)
        mx2 = torch.mean(x2, 0)
        sx1 = x1-mx1
        sx2 = x2-mx2
        dm = self.matchnorm(mx1, mx2)
        scms = dm
        for i in range(n_moments - 1):
            scms += self.scm(sx1, sx2, i + 2)
        return scms

    def matchnorm(self, x1, x2):
        power = torch.pow(x1-x2,2)
        summed = torch.sum(power)
        sqrt = summed**(0.5)
        return sqrt
        # return ((x1-x2)**2).sum().sqrt()

    def scm(self, sx1, sx2, k):
        ss1 = torch.mean(torch.pow(sx1, k), 0)
        ss2 = torch.mean(torch.pow(sx2, k), 0)
        return self.matchnorm(ss1, ss2)

class DiffLoss(nn.Module):

    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, input1, input2):

        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)

        # Zero mean
        input1_mean = torch.mean(input1, dim=0, keepdims=True)
        input2_mean = torch.mean(input2, dim=0, keepdims=True)
        input1 = input1 - input1_mean
        input2 = input2 - input2_mean

        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)
        

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

        diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))

        return diff_loss

class MarginLoss(nn.Module):

    def __init__(self, margin=0.03):
        super(MarginLoss, self).__init__()
        self.margin = margin
        # self.margin = 0.01

    def forward(self, pred, label):
        batch, label_num = label.shape
        
        label_sort, index_sort = torch.sort(label, dim=-1, descending=True)
        pred_sorted_by_true = pred.gather(dim=1, index=index_sort)
        
        pred_dis = pred_sorted_by_true.unsqueeze(2) - pred_sorted_by_true.unsqueeze(1)
        label_dis =  label_sort.unsqueeze(2) - label_sort.unsqueeze(1)
        
        mask = torch.triu(torch.ones(label_num, label_num), diagonal=2) + torch.tril(torch.ones(label_num, label_num), diagonal=0)
        mask = mask.to(torch.bool).to(label.device)
        dis_dis = self.margin + label_dis - pred_dis
        dis_dis_mask = dis_dis.masked_fill(mask, 0)
        loss = torch.relu(dis_dis_mask)

        return loss.mean()

class ListnetLoss(nn.Module):

    def __init__(self):
        super(ListnetLoss, self).__init__()

    def forward(self, pred, label):

        
        top1_target = torch.softmax(label, dim=-1)
        top1_predict = torch.softmax(pred, dim=-1)
        return torch.mean(-torch.sum(top1_target * torch.log(top1_predict)))

class ListMleLoss(nn.Module):

    def __init__(self):
        super(ListMleLoss, self).__init__()

    def forward(self, y_pred, y_true, k=None):
        # y_pred : batch x n_items
        # y_true : batch x n_items 
        if k is not None:
            sublist_indices = (y_pred.shape[1] * torch.rand(size=k)).long()
            y_pred = y_pred[:, sublist_indices] 
            y_true = y_true[:, sublist_indices] 
    
        _, indices = y_true.sort(descending=True, dim=-1)
    
        pred_sorted_by_true = y_pred.gather(dim=1, index=indices)
    
        cumsums = pred_sorted_by_true.exp().flip(dims=[1]).cumsum(dim=1).flip(dims=[1])
    
        listmle_loss = torch.log(cumsums + 1e-10) - pred_sorted_by_true
    
        return listmle_loss.sum(dim=1).mean()

class ThresholdLoss(nn.Module):

    def __init__(self, threshold=0.05, margin_left=0.03, margin_right=0.03):
        super(ThresholdLoss, self).__init__()
        self.threshold = threshold
        self.margin_left = margin_left
        self.margin_right = margin_right

    def forward(self, pred, label):

        
        sign = ((label - self.threshold) + 1e-6) / torch.abs((label - self.threshold) + 1e-6)
        sign = sign.detach()
        
        ts_loss = (0.5 - 0.5 * sign) * (pred - self.threshold + self.margin_left) + (0.5 + 0.5 * sign) * (self.threshold - pred + self.margin_right)

        loss = torch.relu(ts_loss)
        return loss.mean()



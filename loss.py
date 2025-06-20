import torch
import torch.nn as nn
import torch.nn.functional as F
from model import ResUNetAtt
from utils import loadData
from data import DriveDataset
from torch.utils.data import DataLoader
import numpy as np
from typing import List
from sklearn.model_selection import train_test_split

class JaccardLossSigmoid(nn.Module):
    
    def __init__(self):
        super(JaccardLossSigmoid, self).__init__()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, smooth=1) -> torch.Tensor:

        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        jaccard = (intersection + smooth)/(inputs.sum() + targets.sum() + smooth)

        return 1 - jaccard

class DiceLossSigmoid(nn.Module):
    
    def __init__(self):
        super(DiceLossSigmoid, self).__init__()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, smooth=1) -> torch.Tensor:

        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        

        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)

        return 1 - dice

class JaccardBCELossSigmoid(nn.Module):
    
    def __init__(self, alfa: float):
        super(JaccardBCELossSigmoid, self).__init__()
        self.alfa = alfa
        self.jaccardloss = JaccardLossSigmoid()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, smooth=0.0001) -> torch.Tensor:

        inputs = inputs.squeeze()
        targets = targets.squeeze()

        jaccard = self.jaccardloss(inputs, targets, smooth)
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)

        #flatten label and prediction tensors
        targets = targets.view(-1)
        
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean') 

        loss = self.alfa * BCE + (1-self.alfa) * jaccard

        return loss
    
class DiceBCELossSigmoid(nn.Module):
    
    def __init__(self, alfa: float):

        super(DiceBCELossSigmoid, self).__init__()
        self.alfa = alfa
        self.diceloss = DiceLossSigmoid()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, smooth=0.0001) -> torch.Tensor:

        inputs = inputs.squeeze()
        targets = targets.squeeze()

        dice = self.diceloss(inputs, targets, smooth)
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)

        #flatten label and prediction tensors
        targets = targets.view(-1)
        
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')

        loss = self.alfa * BCE + (1-self.alfa) * dice

        return loss
    
class DiceBCELossSigmoidClassify(nn.Module):
    
    def __init__(self, alfa: float):
        super(DiceBCELossSigmoid, self).__init__()
        self.alfa = alfa
        self.diceloss = DiceLossSigmoid()
        self.bce = nn.CrossEntropyLoss()

    def forward(self, inputs_class: torch.Tensor, targets_class: torch.Tensor, 
                inputs_seg: torch.Tensor, targets_seg: torch.Tensor, smooth=0.0001) -> torch.Tensor:

        inputs_seg = inputs_seg.squeeze()
        targets_seg = targets_seg.squeeze()

        dice = self.diceloss(inputs_seg, targets_seg, smooth)
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs_seg = torch.sigmoid(inputs_seg)
        inputs_seg = inputs_seg.view(-1)

        #flatten label and prediction tensors
        targets_seg = targets_seg.view(-1)
        
        BCE = F.binary_cross_entropy(inputs_seg, targets_seg, reduction='mean')

        loss_seg = self.alfa * BCE + (1-self.alfa) * dice

        loss_class = self.bce(inputs_class, targets_class)

        return (loss_class + loss_seg)/2

class BoundaryLossSigmoid(nn.Module):
    """Boundary Loss proposed in:
    Alexey Bokhovkin et al., Boundary Loss for Remote Sensing Imagery Semantic Segmentation
    https://arxiv.org/abs/1905.07852
    """

    def __init__(self, theta0=3, theta=5):
        super().__init__()

        self.theta0 = theta0
        self.theta = theta
        self.diceloss = DiceLossSigmoid()

    def forward(self, pred: torch.Tensor, gt: torch.Tensor):
        """
        Input:
            - pred: the output from model (before softmax)
                    shape (N, C, H, W)
            - gt: ground truth map
                    shape (N, C, H, w)
        Return:
            - boundary loss, averaged over mini-bathc
        """

        n, c, _, _ = pred.shape

        # softmax so that predicted map can be distributed in [0, 1]
        pred = torch.sigmoid(pred)

        # one-hot vector of ground truth
        # one_hot_gt = one_hot(gt, c)
        # one_hot_gt = gt.permute(0, 3, 1, 2)
        pred = pred.squeeze()
        gt = gt.squeeze()
        dice = self.diceloss(pred, gt, 0.0001)
        # print(pred.shape, gt.shape)
        pred_bce = pred.view(-1)
        gt_bce = gt.view(-1)

        # print(pred.shape, gt.shape)


        bceloss = F.binary_cross_entropy(pred_bce, gt_bce, reduction='mean')

        # boundary map
        gt_b = F.max_pool2d(
            1 - gt, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        gt_b -= 1 - gt

        pred_b = F.max_pool2d(
            1 - pred, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        pred_b -= 1 - pred

        # extended boundary map
        gt_b_ext = F.max_pool2d(
            gt_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)

        pred_b_ext = F.max_pool2d(
            pred_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)

        # reshape
        gt_b = gt_b.view(n, c, -1)
        pred_b = pred_b.view(n, c, -1)
        gt_b_ext = gt_b_ext.view(n, c, -1)
        pred_b_ext = pred_b_ext.view(n, c, -1)

        # Precision, Recall
        P = torch.sum(pred_b * gt_b_ext, dim=2) / (torch.sum(pred_b, dim=2) + 1e-7)
        R = torch.sum(pred_b_ext * gt_b, dim=2) / (torch.sum(gt_b, dim=2) + 1e-7)

        # Boundary F1 Score
        BF1 = 2 * P * R / (P + R + 1e-7)

        # summing BF1 Score for each class and average over mini-batch
        bloss = torch.mean(1 - BF1)
        loss = 0.4*bceloss + 0.2*bloss + 0.4*dice
        
        return loss

class DiceBCELoss(nn.Module):
    
    def __init__(self, alfa: float, weight=None, size_average=True, num_classes=2):
        super(DiceBCELoss, self).__init__()
        self.num_classes = num_classes
        self.alfa = alfa

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, smooth=0.0001):

        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.softmax(inputs, dim=1)

        #flatten label and prediction tensors
        targets = targets.view(-1)

        losses = []
        for c in range(0,self.num_classes):

            inputs_ = inputs[:,c,:,:]
            inputs_ = inputs_.reshape(-1)
            targets_ = (targets == (c)).float()

            intersection = (inputs_ * targets_).sum()
            dice_loss = 1 - (2.*intersection + smooth)/(inputs_.sum() + targets_.sum() + smooth)
            BCE = self.alfa * F.binary_cross_entropy(inputs_, targets_, reduction='mean') + (1 - self.alfa) * dice_loss
            losses.append(BCE)

        return torch.sum(torch.stack(losses))

class DiceLoss(nn.Module):
    """
    Classe responsavel por calcular o Loss de um modelo utilizando o Dice.

    Atributos:
        num_classes (int): número de classes
        class_weights (torch.tensor): Pesos para cada classe.

    Métodos:
        __init__(num_classes, class_weights): inicializa os atributos.
        forward(logits, targets): Aplica uma softmax na saida do modelo e gera as probabilidades. Então, calcula o dice para cada classe e tira a média ponderada.
  """
    def __init__(self, num_classes, pesos, alpha):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.pesos = torch.softmax(torch.FloatTensor(pesos), dim=0)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        
        # Aplica a função softmax para obter probabilidades
        probabilities = torch.softmax(logits, dim=1)
        losses = []
        bceloss_fn = nn.BCELoss()

        for class_index in range(0, self.num_classes):
           
            # Calcula a máscara para a classe atual
            class_mask = (targets == class_index).float()
            # Calcula a máscara de previsão para a classe atual
            predicted_mask = probabilities[:, class_index, ...]
            # Calcula o numerador do Dice Score
            intersection = torch.sum(class_mask * predicted_mask)
            dice_numerator = 2.0 * intersection

            # Calcula o denominador do Dice Score
            dice_denominator = torch.sum(class_mask) + torch.sum(predicted_mask)

            # Calcula o Dice Loss para a classe atual
            dice_loss = 1.0 - (dice_numerator + 1e-5) / (dice_denominator + 1e-5)  # Adiciona uma pequena constante para evitar divisão por zero
            dice_loss = dice_loss * self.pesos[class_index]  # Aplica o peso da classe

            loss = self.alpha * bceloss_fn(predicted_mask, class_mask) + (1-self.alpha) * dice_loss

            losses.append(loss)

        # Soma os Dice Losses de todas as classes
        total_dice_loss = torch.sum(torch.stack(losses)) #antes era torch.sum() ao invés de torch.mean()

        return total_dice_loss
    
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        probabiliteis = F.softmax(logits, dim=1)
        loss = F.cross_entropy(probabiliteis, targets, reduction='none')

        if self.alpha is not None:
            # Calcula os pesos ponderados para cada exemplo
            alpha_c = self.alpha[targets.view(-1)].view_as(targets)
            loss = alpha_c * loss

        pt = torch.exp(-loss)
        focal_loss = (1 - pt) ** self.gamma * loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
        
if __name__ == '__main__':

    base = "C:/Projetos/Datasets/thiago.freire"
    origa = f"{base}/ORIGA/"
    refuge = f"{base}/REFUGE/other"
    origa_paths, refuge_paths = loadData(Origa_path=origa, Refuge_path=refuge)

    # train, test = train_test_split(origa_paths, test_size=0.2)

    train_dataset = DriveDataset(origa_paths, 1)
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=35,
        shuffle=True,
        num_workers=1
    )

    blocks = ['AT', 'AT', 'AT', 'NT', 'AT', 'NT', 'AT', 'NT']
    layers = [2,3,3,5]
    skip = [True, False, True, False]
    model = ResUNetAtt(blocks=blocks, layers=layers, skips=skip)

    device = torch.device('cuda')
    model = model.to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

    dice_loss_fn = DiceBCELossSigmoid(alfa=0.5)
    boundary_loss_fn = BoundaryLossSigmoid()

    for x, y in train_loader:

        x = x.to(device)
        y = y.to(device)

        y = y.type(torch.float32)[:,:,:,0]

        optimizer.zero_grad()
        y_pred = model(x)

        dice_loss = dice_loss_fn(y_pred, y)

        loss = dice_loss

        loss.backward()

        optimizer.step()

        print(loss.item())

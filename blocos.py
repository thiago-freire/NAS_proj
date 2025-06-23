import torch
import torch.nn as nn

class STEM(nn.Module):

    def __init__(self):
        super(STEM, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        return x

class SimAM(nn.Module):
    
    def __init__(self, channels = None, e_lambda = 1e-4):
        super(SimAM, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "SimAM"

    def forward(self, x: torch.Tensor):

        b, c, h, w = x.size()
        
        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2,3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2,3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)
    
class ResBlock(nn.Module):

    def __init__(self, in_c, out_c, stride=1, downsample=None):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out

class ConvBlock(nn.Module):

    def __init__(self, in_c, out_c, stride=1):
        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor):
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        return out

class ResBlockAT(nn.Module):

    def __init__(self, in_c, out_c, stride=1, downsample=None):
        super(ResBlockAT, self).__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.at = SimAM()

        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor):
        
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        att = self.at(identity)
        out += att
        out = self.relu(out)
        
        return out
    
class ResBlockUP(nn.Module):

    def __init__(self, in_c, out_c, skip):
        super(ResBlockUP, self).__init__()

        self.conv1 = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.bn1 = nn.BatchNorm2d(out_c)
        
        self.conv2 = nn.Conv2d(out_c*2, out_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.upsample = nn.Sequential(nn.Upsample(mode='bilinear', scale_factor=2), 
                                      nn.Conv2d(in_c, out_c, kernel_size=1, stride=1, padding=0, bias=False))
        
        self.relu = nn.ReLU(inplace=True)

        self.hasSkip = skip


        if skip:
            self.skipBlock = SimAM()

    def forward(self, x: torch.Tensor, skip: torch.Tensor):
        identity = self.upsample(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if self.hasSkip:
            skip = self.skipBlock(skip)

        out = torch.cat([out, skip], axis=1)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = self.relu(out)
        
        return out

class ConvBlockUP(nn.Module):

    def __init__(self, in_c, out_c, skip):
        super(ConvBlockUP, self).__init__()

        self.conv1 = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.bn1 = nn.BatchNorm2d(out_c)
        
        self.conv2 = nn.Conv2d(out_c*2, out_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)        
        self.relu = nn.ReLU(inplace=True)

        self.hasSkip = skip


        if skip:
            self.skipBlock = SimAM()

    def forward(self, x: torch.Tensor, skip: torch.Tensor):
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if self.hasSkip:
            skip = self.skipBlock(skip)

        out = torch.cat([out, skip], axis=1)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = self.relu(out)
        
        return out

class ResBlockUPAT(nn.Module):

    def __init__(self, in_c, out_c, skip):
        super(ResBlockUPAT, self).__init__()

        self.conv1 = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.bn1 = nn.BatchNorm2d(out_c)
        
        self.conv2 = nn.Conv2d(out_c*2, out_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.upsample = nn.Sequential(nn.Upsample(mode='bilinear', scale_factor=2), 
                                      nn.Conv2d(in_c, out_c, kernel_size=1, stride=1, padding=0, bias=False))
        self.at = SimAM()
        self.relu = nn.ReLU(inplace=True)

        self.hasSkip = skip

        if skip:
            self.skipBlock = SimAM()

    def forward(self, x: torch.Tensor, skip: torch.Tensor):
        identity = self.upsample(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if self.hasSkip:
            skip = self.skipBlock(skip)

        out = torch.cat([out, skip], axis=1)
        
        out = self.conv2(out)
        out = self.bn2(out)
        att = self.at(identity)
        out += att
        out = self.relu(out)
        
        return out
    
class SegHEAD(nn.Module):

    def __init__(self, in_c, out_c, num_classes):
        super().__init__()

        self.conv1 = nn.Sequential(nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0),
                                   nn.BatchNorm2d(out_c),
                                   nn.ReLU())
        
        self.conv2 = nn.Sequential(nn.Conv2d(out_c+3, num_classes, kernel_size=1, padding=0),
                                   nn.BatchNorm2d(num_classes),
                                   nn.ReLU())

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:        
        
        x = self.conv1(x)
        x = torch.cat([x, skip], axis=1)
        x = self.conv2(x)

        return x

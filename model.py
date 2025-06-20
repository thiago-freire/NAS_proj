import torch.nn as nn
from blocos import STEM, ResBlock, ResBlockAT, ResBlockUP, ResBlockUPAT, SegHEAD
from torchinfo import summary

class ResUNetAtt(nn.Module):
    
    def __init__(self, blocks, layers, skips):
        super(ResUNetAtt, self).__init__()  

        self.in_channels = 32     
        
        self.stem = STEM()
        self.enc1 = self.__make_encoder(blocks[0], 64, layers[0])   # Sem downsample
        self.enc2 = self.__make_encoder(blocks[1], 128, layers[1], stride=2)
        self.enc3 = self.__make_encoder(blocks[2], 256, layers[2], stride=2)
        self.enc4 = self.__make_encoder(blocks[3], 512, layers[3], stride=2)
        self.dec1 = self.__make_decoder(blocks[4], 256, layers[3], skips[0])
        self.dec2 = self.__make_decoder(blocks[5], 128, layers[2], skips[1])
        self.dec3 = self.__make_decoder(blocks[6], 64, layers[1], skips[2])
        self.dec4 = self.__make_decoder(blocks[7], 32, layers[0], skips[3])
        self.segHead = SegHEAD(32, 16, 2)

    def __make_encoder(self, block, out_channels, blocks, stride=1):
        
        layers = []
        if self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels,
                          kernel_size=1, stride=2, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        if block == 'NT':
            layers.append(ResBlock(self.in_channels, out_channels, stride=2, downsample=downsample))
        else:
            layers.append(ResBlockAT(self.in_channels, out_channels, stride=2, downsample=downsample))
        self.in_channels = out_channels
        
        for _ in range(1, blocks):
            if block == 'AT':
                layers.append(ResBlockAT(out_channels, out_channels))
            else:
                layers.append(ResBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def __make_decoder(self, block, out_channels, blocks, skip):
        
        if self.in_channels != out_channels:
            if block == 'NT':
                frist = ResBlockUP(self.in_channels, out_channels, skip)
            else:
                frist = ResBlockUPAT(self.in_channels, out_channels, skip)

        self.in_channels = out_channels
        
        layers = []
        for _ in range(1, blocks):
            if block == 'AT':
                layers.append(ResBlockAT(out_channels, out_channels))
            else:
                layers.append(ResBlock(out_channels, out_channels))
        
        tail = nn.Sequential(*layers)
        
        return nn.ModuleList([tail,frist])

    def forward(self, x):
        stem = self.stem(x)
        
        ec1 = self.enc1(stem)
        ec2 = self.enc2(ec1)
        ec3 = self.enc3(ec2)
        ec4 = self.enc4(ec3)
        
        out = self.dec1[0](self.dec1[1](ec4, ec3))
        out = self.dec2[0](self.dec2[1](out, ec2))
        out = self.dec3[0](self.dec3[1](out, ec1))
        out = self.dec4[0](self.dec4[1](out, stem))
        out = self.segHead(out, x)
        
        return out
    
if __name__ == "__main__":

    blocks = ['AT', 'AT', 'AT', 'NT', 'AT', 'NT', 'AT', 'NT']
    layers = [2,3,3,5]
    skip = [True, False, True, False]
    model = ResUNetAtt(blocks=blocks, layers=layers, skips=skip)
    model.to(device='cuda')

    summary(model, (30, 3, 256, 256), device='cuda', depth=2)
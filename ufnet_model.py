import torch
import torch.nn as nn
import torch.nn.functional as F
from ufnet_parts import *
import matplotlib.pyplot as plt
class UFNet(nn.Module):
    def __init__(self,n_channels,n_classes):
        super(UFNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        #UNet
        self.Stage1InputConv=UNetInputConv(n_channels,32)
        self.Down1=Down(32,64)
        self.Down2=Down(64,128)
        #upsample1
        self.upRSEB1=RSEB(128)
        self.UP1=Up(128,64)
        self.ResidualRSEB1=RSEB(64)
        # upsample2
        self.upRSEB2 = RSEB(64)
        self.UP2 = Up(64, 32)
        self.ResidualRSEB2 = RSEB(32)
        #Fd1
        self.Fd1DoubleRSEB=RSEB(32)
        #P___Stage1Output
        self.Stage1Output=Stage1Output(32,1)

        # FNet
        self.Stage2InputConv=FNetInputConv(n_channels,32)
        # first FRB
        self.FRB1=FRB(64)
        self.sub1_twice_1_UP=Up(128,64)
        self.sub1_twice_2_UP = Up(64, 64)
        self.sub2_twice_1_UP = Up(128, 64)
        self.sub2_twice_2_UP = Up(64, 64)
        # second FRB
        self.FRB2=FRB(64)
        self.sub1_once_UP=Up(64,64)
        self.sub2_once_UP = Up(64, 64)
        # third FRB
        self.FRB3=FRB(64)
        self.sub1_zero_UP=DoubleConv(32,64)
        self.sub2_zero_UP = DoubleConv(32, 64)
        # FNet-output
        self.FoutConv=DoubleConv(64,1)
    def forward(self,Xin):
        # UNet

        Xin1=Xin[:,:1,:,:]

        Fe1=self.Stage1InputConv(Xin)
        Fe2=self.Down1(Fe1)
        Fe3=self.Down2(Fe2)
        Fd3=self.upRSEB1(Fe3)
        up1_out=self.UP1(Fd3,Fe2)
        up1_out=up1_out+self.ResidualRSEB1(Fe2)
        Fd2=self.upRSEB2(up1_out)
        up2_out=self.UP2(Fd2,Fe1)
        up2_out=up2_out+self.ResidualRSEB2(Fe1)
        Fd1=self.Fd1DoubleRSEB(up2_out)
        P=Fd1
        stage1Output, Fatt=self.Stage1Output(P,Xin1)
        # FNet
        Finit=torch.cat([self.Stage2InputConv(Xin),Fatt],dim=1)
        #first FRB

        Fo1=self.FRB1(Finit)

        Fe3=self.sub1_twice_1_UP(Fe3,Fe2)
        Fe3=self.sub1_twice_2_UP(Fe3,Fo1)

        Fd3=self.sub2_twice_1_UP(Fd3,Fd2)
        Fd3=self.sub2_twice_2_UP(Fd3,Fo1)

        Fo1=Fo1+Fd3+Fe3
        # second FRB
        Fo2=self.FRB2(Fo1)

        Fe2=self.sub1_once_UP(Fe2,Fo2)

        Fd2=self.sub2_once_UP(Fd2,Fo2)

        Fo2=Fo2+Fe2+Fd2
        # third FRB
        Fo3=self.FRB3(Fo2)

        Fe1=self.sub1_zero_UP(Fe1)

        Fd1=self.sub2_zero_UP(Fd1)

        Fo3=Fo3+Fe1+Fd1
        #Fnet outConv

        FinalOutput=self.FoutConv(Fo3)+Xin1
        return stage1Output,FinalOutput








if __name__=="__main__":
    net=UFNet(n_channels=1,n_classes=1)
    print(net)
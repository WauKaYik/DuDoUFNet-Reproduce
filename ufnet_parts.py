import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            # # nn.BatchNorm2d(out_channels),
            # nn.LeakyReLU(inplace=True)
        )
    def forward(self, x):
        x = self.double_conv(x)
        return x

class sub_RSEB(nn.Module):
    """
    3D extension of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
        *Zhu et al., AnatomyNet, arXiv:arXiv:1808.05238*
    """

    def __init__(self, num_channels, reduction_ratio=2, norm='None'):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(sub_RSEB, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.norm = norm
        self.bn = nn.BatchNorm3d(num_channels)

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output tensor
        """
        batch_size, num_channels, D, H = input_tensor.size()
        # Average along each channel
        squeeze_tensor = self.avg_pool(input_tensor)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor.view(batch_size, num_channels)))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        output_tensor = torch.mul(input_tensor, fc_out_2.view(batch_size, num_channels, 1, 1))

        if self.norm == 'BN':
            output_tensor = self.bn(output_tensor)

        return output_tensor
class RSEB(nn.Module):
    def __init__(self,  out_channels):
        super().__init__()
        # self.se=nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=1),
        #     nn.PReLU(out_channels),
        #     nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1),
        #     nn.Sigmoid()
        # )
        self.sub_RSEB=sub_RSEB(out_channels)
        self.feature=nn.Sequential(
            DoubleConv(out_channels,out_channels),
            nn.PReLU(out_channels),
            DoubleConv(out_channels,out_channels)

               )
    def forward(self,Fin):
        Ftr=self.feature(Fin)
        # Fscale=self.se(Ftr)
        Fscale=self.sub_RSEB(Ftr)
        # Fout=(Ftr*Fscale)+Fin
        Fout = (Fscale) + Fin
        return Fout

# class RSEB(nn.Module):
#     def __init__(self, channel, reduction=16):
#         super(RSEB, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(channel, channel // reduction, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(channel // reduction, channel, bias=False),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         b, c, _, _ = x.size()
#         y = self.avg_pool(x).view(b, c)
#         y = self.fc(y).view(b, c, 1, 1)
#         return x * y.expand_as(x)


class DoubleRSEB(nn.Module):
    def __init__(self,out_channels):
        super().__init__()
        self.doubleRSEB=nn.Sequential(
            RSEB(out_channels),
            RSEB(out_channels)
        )
    def forward(self,x):
        return self.doubleRSEB(x)

class UNetInputConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.uinput=nn.Sequential(
            DoubleConv(in_channels,out_channels),
            RSEB(out_channels),
            DoubleRSEB(out_channels)
        )
    def forward(self,x):
        return self.uinput(x)

class FNetInputConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.finput=nn.Sequential(
            DoubleConv(in_channels,out_channels),
            RSEB(out_channels)
        )
    def forward(self,x):
        return self.finput(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down=nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
            RSEB(out_channels)


        )
    def forward(self,x):
        return self.down(x)

class Up(nn.Module):
    def __init__(self,in_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv=DoubleConv(in_channels, out_channels)
    def forward(self,x,y):
        x = self.up(x)
        diffX = y.size()[3] - x.size()[3]
        diffY = y.size()[2] - x.size()[2]
        x = F.pad(x, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
        x = self.conv(x)




        return x

class FRB(nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        self.frb=nn.Sequential(
            # DoubleRSEB(in_channels),
            DoubleRSEB(in_channels),
            DoubleRSEB(in_channels),
            DoubleConv(in_channels,in_channels)
        )
    def forward(self,x):
         return (self.frb(x)+x)

class Stage1Output(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.P2=nn.Sequential(
            DoubleConv(in_channels,out_channels)
        )
        self.P1=nn.Sequential(
            DoubleConv(in_channels,out_channels)
        )
        self.behindOut=nn.Sequential(
            DoubleConv(out_channels,out_channels),
            nn.Sigmoid()
        )
    def forward(self,P,Xin1):
        stage1Output=Xin1+self.P2(P)
        Fatt=self.P1(P)*self.behindOut(stage1Output)+P
        return stage1Output,Fatt


















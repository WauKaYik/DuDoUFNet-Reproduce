from ufnet_model import UFNet
import torch.nn as nn
import torch
from deeplesion.build_gemotry320 import initialization,build_gemotry
from odl.contrib import torch as odl_torch

para_ini = initialization()                                                               #对重建用到的参数进行初始化
fp = build_gemotry(para_ini)                                                              #创建对应前投影的几何对象？
op_modfp = odl_torch.OperatorModule(fp)                                                   #前投影？
op_modpT = odl_torch.OperatorModule(fp.adjoint)                                           #反投影？

class DuDoUFNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.sino_Net=UFNet(2,1)
        self.image_Net=UFNet(3,1)
        # self.image_Net = UFNet(2, 1)
    def forward(self,Sldma,Xldma ,mask):
        Mproj=op_modfp(mask/255)/ 4.0 * 255
        # Mproj_num = Mproj.data.cpu().numpy()
        sino_input=torch.cat([Sldma,Mproj],dim=1)
        Su,So=self.sino_Net(sino_input)
        M=mask
        Xo=op_modpT((So/255)*4.0)
        Xo=Xo.transpose(3,2).transpose(3,2).transpose(3,2).flip([2])
        # Xo_num=Xo.data.cpu().numpy()
        image_input=torch.cat([Xldma,Xo,M],dim=1)
        # image_input=torch.cat([Xldma,M],dim=1)
        Xu,Xfinal=self.image_Net(image_input)
        return Su,So,Xu,Xo,Xfinal,Mproj,M
        # return  Xu,Xfinal





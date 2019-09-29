import torch as th
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from DeepPeyre.simplex import L1BallProj

class ProjStep(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x,rad):
        #input (Nb x N^2)
        #output (Nb x N^2)
        projL1 = L1BallProj(z=rad)
        projTerm, _ = projL1(x)
        return projTerm

class GradOp(nn.Module):
    def __init__(self, Nb, N=28):
        """
        Write gradient operator of an Nb x 1 x N x N BW image as a convolution
        """
        super().__init__()
        self.Nb = Nb
        self.L = nn.ReplicationPad2d((1,1,1,1))
        self.N = N

        self.register_buffer('weightx', th.FloatTensor([ [0,0,0], [-0.5,0,0.5],[0,0,0]  ]))
        self.register_buffer('weighty', th.FloatTensor([ [0,0.5,0],[0,0,0],[0,-0.5,0] ]))

        self.weightx.unsqueeze_(0)
        self.weightx.unsqueeze_(0)
        self.weighty.unsqueeze_(0)
        self.weighty.unsqueeze_(0)

        self.weightx = self.weightx.expand(1,1,3,3) #nc,1,3,3
        self.weighty = self.weighty.expand(1,1,3,3)

    def forward(self,u):
        u_ = self.L(u)
        ux = F.conv2d( u_ , self.weightx, padding=0,groups=1 )
        uy = F.conv2d( u_ , self.weighty, padding=0,groups=1 )
        ux, uy = ux.unsqueeze_(4), uy.unsqueeze_(4)
        return th.cat([ux,uy],dim=-1)


class ComputeTV(nn.Module):
    def __init__(self,Nb,N=28,has_cuda=True):
        super().__init__()
        self.Nb = Nb
        self.N = N
        self.has_cuda = has_cuda

    def forward(self,u):
        gradop = GradOp(self.Nb,self.N)
        if self.has_cuda:
            gradop = gradop.cuda()
        GradU = gradop(u)
        SumAbsGradUx = GradU[:,:,:,:,0].view(self.Nb,-1).abs().sum(dim=-1)
        SumAbsGradUy = GradU[:,:,:,:,1].view(self.Nb,-1).abs().sum(dim=-1)
        return SumAbsGradUx + SumAbsGradUy

class DivOp(nn.Module):
    def __init__(self, Nb, N=28):
        """
        Write gradient operator of an Nb x 1 x N x N BW image as a convolution
        """
        super().__init__()
        self.Nb = Nb
        self.L = nn.ReplicationPad2d((1,1,1,1))
        self.N = N

        self.register_buffer('weightx', th.FloatTensor([ [0,0,0], [-0.5,0,0.5],[0,0,0]  ]))
        self.register_buffer('weighty', th.FloatTensor([ [0,0.5,0],[0,0,0],[0,-0.5,0] ]))

        self.weightx.unsqueeze_(0)
        self.weightx.unsqueeze_(0)
        self.weighty.unsqueeze_(0)
        self.weighty.unsqueeze_(0)

        self.weightx = self.weightx.expand(1,1,3,3) #nc,1,3,3
        self.weighty = self.weighty.expand(1,1,3,3)

    def forward(self,u):
        u1 = u[:,:,:,:,0]
        u2 = u[:,:,:,:,1]
        u1_ = self.L(u1)
        u2_ = self.L(u2)
        u1_x = F.conv2d( u1_ , self.weightx, padding=0,groups=1 )
        u2_y = F.conv2d( u2_ , self.weighty, padding=0,groups=1 )
        return u1_x + u2_y


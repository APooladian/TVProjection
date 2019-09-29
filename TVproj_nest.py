import torch as th
import numpy as np
import torch.nn as nn

from DeepPeyre.utils import *

class TVBallProj(nn.Module):
    def __init__(self,Niters=200,mu=0.25,has_cuda=True):
        super().__init__()

        self.Niters=Niters
        self.mu = mu
        self.has_cuda = has_cuda

    def forward(self,M,tv0):
        Nb,nc,N,N = M.shape

        mu=self.mu
        u = th.zeros(Nb,1,N,N,2)
        xi = th.zeros(Nb,1,N,N,2)
        A = 0

        if self.has_cuda:
            u = u.cuda()
            xi = xi.cuda()

        divop = DivOp(Nb=Nb,N=N)
        gradop = GradOp(Nb,N=N)
        if self.has_cuda:
            divop = divop.cuda()
            gradop = gradop.cuda()

        projstep = ProjStep()

        for k in range(self.Niters):
            a = (mu + np.sqrt(mu**2 + 4*mu*A))/2

            v = u - xi
            if k >= 1:
                vFlat_x = v[:,:,:,:,0].view(Nb,N*N)
                vFlat_y = v[:,:,:,:,1].view(Nb,N*N)
                vFlat_x = vFlat_x - projstep(vFlat_x,tv0*A)
                vFlat_y = vFlat_y - projstep(vFlat_y,tv0*A)

                v[:,:,:,:,0] = vFlat_x.view(Nb,1,N,N)
                v[:,:,:,:,1] = vFlat_y.view(Nb,1,N,N)

            y = (A*u + a*v)/(A+a)

            u = y - (mu/2) * gradop( M - divop(y) )

            uFlat_x = u[:,:,:,:,0].view(Nb,N**2)
            uFlat_y = u[:,:,:,:,1].view(Nb,N**2)
            uFlat_x = uFlat_x - projstep(uFlat_x,tv0*mu/2)
            uFlat_y = uFlat_y - projstep(uFlat_y,tv0*mu/2)

            u[:,:,:,:,0] = uFlat_x.view(Nb,1,N,N)
            u[:,:,:,:,1] = uFlat_y.view(Nb,1,N,N)

            Mnew = M - divop(u)
            xi = xi + (a * gradop(Mnew))

            M1=Mnew
            A = A + a

        return M1, u

    #size(u) = Nb x 1 x N x N




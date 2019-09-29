import torch as th
import torch.nn as nn
import numpy as np
import imageio

class torch2png(nn.Module):
    def __init__(self,has_cuda=True):
        super().__init__()
        self.has_cuda = has_cuda

    def forward(self,tens,img_name='default.png'):
        #tens is (1,1,N,N)
        #img_name is a string
        tensSqueeze = tens.squeeze(0).squeeze(0).clamp_(0,1)
        if self.has_cuda:
            tensSqueeze = tensSqueeze.cpu()

        tensnp = tensSqueeze.numpy()
        imageio.imwrite(img_name,tensnp)
        return


class png2torch(nn.Module):
    def __init__(self,has_cuda=True):
        super().__init__()
        self.has_cuda = has_cuda
    def forward(self,x):
        #input is string of an (N,N) grayscale PNG between 0 and 255

        xnp = np.array(imageio.imread(x))
        xth = th.Tensor(xnp).float().unsqueeze_(0).unsqueeze_(0)

        if self.has_cuda:
            xth = xth.cuda()

        xth = xth/xth.max()

        #xth is (1,1,N,N) between 0 and 1

        return xth


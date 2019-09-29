import torch as th
import numpy as np
import imageio
from DeepPeyre.TVproj_nest import *

lena_np = np.array(imageio.imread('lena.png'))

lena_th = th.Tensor(lena_np).float()
lena_th = lena_th.unsqueeze_(0)
lena_th = lena_th.unsqueeze_(0)
lena_th = lena_th.cuda()
max_val = lena_th.max()
lena_th = lena_th/max_val

lena_noise = lena_th + (th.randn(1,1,512,512)*0.1).cuda()
lena_noise.clamp_(0,1)
lena_noise_png = imageio.imwrite('lena_noise.png',
                        lena_noise.squeeze(0).squeeze(0).cpu().numpy())

tvproj = TVBallProj(Niters=100)
computetv = ComputeTV(Nb=1,N=512)

tvLena = computetv(lena_th)
print(tvLena)
tv0 = computetv(lena_noise)
print(tv0)
MFinal, uFinal = tvproj(lena_noise,tv0.item()/10)
tvFinal = computetv(MFinal.clamp_(0,1))
print(tvFinal)
lena_new = (MFinal).squeeze_(0).squeeze_(0).cpu().numpy()
lena_save = imageio.imwrite('lena_new.png',lena_new)

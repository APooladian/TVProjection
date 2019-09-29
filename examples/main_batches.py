import torch as th
import numpy as np
import imageio
from TVProjection.TVproj_nest import *
from TVProjection.image_loader import *
import time

png2torch = png2torch()
torch2png = torch2png()

#import image
lena_th = png2torch('lena.png')

#add gaussian noise
lena_noise = (lena_th + (th.randn(1,1,512,512)*0.1).cuda()).clamp_(0,1)

#save noisy image
torch2png(lena_noise,'lena_noisy.png')

#denoise several images at once
BatchSize = 50
Nb,nc,Ndim,Ndim = lena_th.shape

lena_noise_collection = th.zeros(BatchSize,nc,Ndim,Ndim).cuda()

for k in range(BatchSize):
    lena_noise_collection[k] = lena_th

### compute tv of noisy images
computetv = ComputeTV(Nb=BatchSize,N=Ndim)
tv0 = computetv(lena_noise_collection)

#project onto TV ball
tvproj = TVBallProj(Niters=100)
t = time.time()
MFinal, uFinal = tvproj(lena_noise_collection,tv0[0].item()/10)
elapsed = time.time() - t
print('time taken',elapsed)

#save one denoised image
torch2png(MFinal[0].view(1,nc,Ndim,Ndim),'lena_denoised.png')

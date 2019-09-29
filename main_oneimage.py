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

#
tvproj = TVBallProj(Niters=100)
computetv = ComputeTV(Nb=1,N=512)
tv0 = computetv(lena_noise)

#project onto TV ball
t = time.time()
MFinal, uFinal = tvproj(lena_noise,tv0.item()/10)
elapsed = time.time() - t
print('time taken',elapsed)

#save denoised image
torch2png(MFinal,'lena_denoised.png')

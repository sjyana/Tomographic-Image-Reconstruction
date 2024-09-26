#simple iterative image reconstruction algorithm
# ML-EM (maximum likelihood - expectation maximisation)

from skimage.data import shepp_logan_phantom
from skimage.transform import radon, iradon, rescale #irandom with filter = filtered back projection, w no filter = back projection, will use no back filter
import matplotlib.pyplot as plt
import numpy as np

plt.ion() #interactive mode on

activity_level = 0.1
true_object = shepp_logan_phantom()
true_object = rescale(activity_level * true_object, 0.5)

fig, axs = plt.subplots(2, 3, figsize=(20,10)) #consisting 6 subplots, 2 rows & 3 columns
axs[0,0].imshow(true_object, cmap='Greys_r');   axs[0,0].set_title('Object') #setting the object image in first subplot


# generate simulated sinoogram data
azi_angles = np.linspace(0.0, 180.0, 180, endpoint=False)
sinogram = radon(true_object, azi_angles, circle=False)

axs[0, 1].imshow(sinogram.T, cmap='Greys_r');   axs[0,1].set_title('Sinogram')


#reconstruction
# formula:
#  x^(k+1) = (x^k / (A^T)1) (A^T) (m / Ax^k)

mlem_rec = np.ones(true_object.shape) #reconstructed image
sino_ones = np.ones(sinogram.shape)
sens_image = iradon(sino_ones, azi_angles, circle =False, filter_name=None) #back projection of the ones

for iter in range(20):
    fp = radon(mlem_rec, azi_angles, circle=False) #forward projection of of reconstruction
    ratio = sinogram / (fp + 0.000001) #ratio sinogram
    correction = iradon(ratio, azi_angles, circle =False, filter_name=None) / sens_image

    axs[1, 0].imshow(mlem_rec, cmap='Greys_r');   axs[1,0].set_title('MLEM recon')
    axs[1,1].imshow(fp.T, cmap="Greys_r");  axs[1,1].set_title('FP of recon') #forward projection
    axs[0,2].imshow(ratio.T, cmap="Greys_r");  axs[0,2].set_title('Ratio Sinogram')

    axs[1,2].imshow(correction, cmap="Greys_r");  axs[1,2].set_title('BP of ratio') #back projection

    mlem_rec = mlem_rec * correction
    axs[1, 0].imshow(mlem_rec, cmap='Greys_r');   axs[1,0].set_title('MLEM Reconstructed image iteration = %d' % (iter+1))
    plt.show()
    plt.pause(0.05)

plt.show(block=True)
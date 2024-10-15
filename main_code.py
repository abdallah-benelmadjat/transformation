import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import scipy.io as sio

image_path = 'transformationL.png'
image = Image.open(image_path).convert('L')
U = np.array(image)

D = U < 50

Mr, Mc = D.shape
B = np.ones_like(D, dtype=bool)

PixelList = np.column_stack(np.where(D))
PixelValues = D[D].flatten()
PixelListValues = np.column_stack((PixelList, PixelValues))

NDX = np.where(PixelListValues[:, 2])[0]

cv, ch = 2300, 4450
Rad = 1200

IM = (cv - PixelListValues[NDX, 0]) / Rad
RE = (PixelListValues[NDX, 1] - ch) / Rad
COMPLD = RE + 1j * IM

plt.figure(3)
plt.plot(RE, IM, 'r.')
plt.axis('equal')
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.title('Transformation Domain')

sio.savemat('TransPoints.mat', {'COMPLD': COMPLD})

data = sio.loadmat('TransPoints.mat')
COMPLD = data['COMPLD'].flatten()

II = 1j
TR = II * COMPLD + 1 + II
IMtr = np.imag(TR)
REtr = np.real(TR)

plt.figure(4)
plt.plot(REtr, IMtr, 'k.')
plt.axis('equal')
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.title('Transformed Domain')
plt.show()

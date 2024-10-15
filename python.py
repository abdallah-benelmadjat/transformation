import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import scipy.io as sio

# Load the image
image_path = 'C:/Users/abdou/Desktop/MATHmatlab/transformationL.png'
image = Image.open(image_path).convert('L')  # Convert to grayscale
U = np.array(image)

# Identify non-white pixels
D = U < 50

# Get image dimensions
Mr, Mc = D.shape
B = np.ones_like(D, dtype=bool)

# Find pixel coordinates of non-white pixels
PixelList = np.column_stack(np.where(D))
PixelValues = D[D].flatten()
PixelListValues = np.column_stack((PixelList, PixelValues))

# Domain pixel indices
NDX = np.where(PixelListValues[:, 2])[0]

# Center of the unit circle in image coordinates and radius
cv, ch = 2300, 4450
Rad = 1200

# Transform coordinates onto the complex plane
IM = (cv - PixelListValues[NDX, 0]) / Rad
RE = (PixelListValues[NDX, 1] - ch) / Rad
COMPLD = RE + 1j * IM

# Plot domain points in red on the complex plane
plt.figure(3)
plt.plot(RE, IM, 'r.')
plt.axis('equal')
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.title('Transformation Domain')

# Save the complex domain points
sio.savemat('TransPoints.mat', {'COMPLD': COMPLD})

# Load the saved complex domain points
data = sio.loadmat('TransPoints.mat')
COMPLD = data['COMPLD'].flatten()

# Apply transformation f(z) = i * z + 1 + i
II = 1j
TR = II * COMPLD + 1 + II
IMtr = np.imag(TR)
REtr = np.real(TR)

# Plot the transformed image in black
plt.figure(4)
plt.plot(REtr, IMtr, 'k.')
plt.axis('equal')
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.title('Transformed Domain')
plt.show()
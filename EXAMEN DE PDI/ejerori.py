import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Leer la imagen
image = cv2.imread('caras.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convertir la imagen de RGB a L*a*b*
image_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
# Aplanar la imagen
pixel_values = image_lab.reshape((-1, 3))
pixel_values = np.float32(pixel_values)

# Definir los criterios de k-means y el número de clusters (K)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
K = 4  # Número de clusters
_, labels, (centers) = cv2.kmeans(pixel_values, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# Convertir los centros a entero
centers = np.uint8(centers)

# Mapear las etiquetas a los valores de los centros
segmented_image = centers[labels.flatten()]

# Redimensionar la imagen segmentada a la forma original
segmented_image = segmented_image.reshape(image_lab.shape)
segmented_image_rgb = cv2.cvtColor(segmented_image, cv2.COLOR_LAB2RGB) 
# Mostrar la imagen original y la segmentada
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('Imagen Original')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(segmented_image_rgb)
plt.title('Imagen Segmentada')
plt.axis('off')

plt.show()

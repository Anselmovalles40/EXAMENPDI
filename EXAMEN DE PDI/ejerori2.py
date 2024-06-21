import cv2
import matplotlib.pyplot as plt

# Leer dos imágenes
image1 = cv2.imread('caras.jpg', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('usar.jpg', cv2.IMREAD_GRAYSCALE)

# Inicializar el detector ORB
orb = cv2.ORB_create()

# Detectar características clave y describirlas
keypoints1, descriptors1 = orb.detectAndCompute(image1, None)
keypoints2, descriptors2 = orb.detectAndCompute(image2, None)

# Crear un objeto BFMatcher (Brute Force Matcher)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Emparejar las características
matches = bf.match(descriptors1, descriptors2)

# Ordenar los emparejamientos por distancia (los mejores primero)
matches = sorted(matches, key=lambda x: x.distance)

# Dibujar los emparejamientos
matched_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Convertir la imagen emparejada a RGB (ya que Matplotlib usa RGB en lugar de BGR)
matched_image_rgb = cv2.cvtColor(matched_image, cv2.COLOR_BGR2RGB)

# Visualizar los emparejamientos
plt.figure(figsize=(20, 10))
plt.imshow(matched_image_rgb)
plt.title('Emparejamiento de Características ORB')
plt.axis('off')
plt.show()
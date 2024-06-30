import numpy as np
import cv2
def region_growing(image, mask, seeds, delta, p=8): #p=4 u 8, #range: [100,170]
  # Obtener las dimensiones de la imagen
  height, width = image.shape
  # Crear una matriz para almacenar la región resultante
  #region = np.zeros_like(image) #imagen negra

  # Direcciones de vecindad 8 (puedes cambiar a 4 si prefieres)
  if p==8:
    neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]
  elif p==4:
    neighbors = neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1)]
  else:
    raise ValueError("p debe ser 4 u 8")

  #Para cada punto semilla que se ingresa
  for k in range(len(seeds)):
    coord_semilla=seeds[k]
    seed_points = [coord_semilla]
    print(coord_semilla)
    # Obtener el nivel de gris de la semilla
    seed_value = image[coord_semilla[0], coord_semilla[1]]
    def is_in_range(pixel):
            return seed_value - delta <= image[pixel[0], pixel[1]] <= seed_value + delta
    
    while seed_points:
      current_seed = seed_points.pop()
      # Agregar el punto semilla a la región
      mask[current_seed[0], current_seed[1]] = 255

      # Expandir la región verificando los vecinos
      for neighbor in neighbors:
          x, y = current_seed[0] + neighbor[0], current_seed[1] + neighbor[1]
          if 0 <= x < height and 0 <= y < width and mask[x, y] == 0 and is_in_range((x, y)):
              # Agregar el punto vecino a la lista de puntos semilla
              seed_points.append((x, y))
    kernel = np.ones((3,3), np.uint8)
    mask=cv2.dilate(mask,kernel,iterations=1)
    mask=cv2.erode(mask,kernel,iterations=1)
  return mask

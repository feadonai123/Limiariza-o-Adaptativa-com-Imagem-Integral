import time
import numpy as np
import cv2

print("\nINICIO\n")

def adaptive_thresholding_integral_image(in_img, threshold, block_size):
  iterations_count = 0
  
  # Obtém as dimensões da imagem.
  w, h = in_img.shape

  # Calcula a imagem integral.
  intImg = np.zeros((w, h))
  for i in range(w):
    sum = 0
    for j in range(h):
      iterations_count += 1
      sum += in_img[i, j]
      if i == 0:
        intImg[i, j] = sum
      else:
        intImg[i, j] = intImg[i-1, j] + sum

  out_img = np.zeros((w, h))

  # Itera sobre cada pixel da imagem.
  for i in range(w - 1):
    for j in range(h - 1):
      iterations_count += 1
      # Define as coordenadas do bloco.
      x1 = max(0, i - block_size // 2)
      x2 = min(w - 1, i + block_size // 2)
      y1 = max(0, j - block_size // 2)
      y2 = min(h - 1, j + block_size // 2)
      # Calcula a soma dos pixels do bloco
      block_sum = intImg[x2, y2] - intImg[x1, y2] - intImg[x2, y1] + intImg[x1, y1]
      # Calcula a quantidade de pixels do bloco
      block_count = (x2 - x1) * (y2 - y1)
      # Compara o pixel com a média do bloco.
      if in_img[i, j] * block_count <= (block_sum * (100 - threshold) / 100):
        out_img[i, j] = 0
      else:
        out_img[i, j] = 255

  return out_img, iterations_count

def adaptive_thresholding_mean(in_img, threshold, block_size):
  iterations_count = 0
  # Obtém as dimensões da imagem.
  w, h = in_img.shape

  # Inicializa a imagem de saída.
  out_img = np.zeros((w, h))

  # Itera sobre cada pixel da imagem.
  for i in range(w):
    for j in range(h):
      iterations_count += 1
      # Define os limites do bloco.
      x1 = max(0, i - block_size // 2)
      x2 = min(w - 1, i + block_size // 2)
      y1 = max(0, j - block_size // 2)
      y2 = min(h - 1, j + block_size // 2)

      # Calcula a média do bloco.
      block_mean = np.mean(in_img[x1:x2, y1:y2])
      iterations_count += (x2 - x1) * (y2 - y1)

      # Compara o pixel com a média do bloco.
      if in_img[i, j] <= (block_mean * (100 - threshold) / 100):
        out_img[i, j] = 0
      else:
        out_img[i, j] = 255

  return out_img, iterations_count

def wellners_method(in_img, threshold, s):
  iterations_count = 0
  # Obtém as dimensões da imagem.
  w, h = in_img.shape

  # Inicializa a imagem de saída.
  out_img = np.zeros((w, h), dtype=np.uint8)

  # Inicializa a média móvel.
  moving_average = 0

  # Itera sobre cada pixel da imagem.
  for i in range(w):
    for j in range(h):
      iterations_count += 1
      # Atualiza a média móvel.
      moving_average = (moving_average * (s - 1) + in_img[i, j]) / s

      # Compara o pixel com a média móvel.
      if in_img[i, j] <= (moving_average * (100 - threshold) / 100):
        out_img[i, j] = 0
      else:
        out_img[i, j] = 255

  return out_img, iterations_count

def otsu_thresholding(image):
  iterations_count = 0
  # Calcula o histograma da imagem.
  w, h = image.shape
  
  histogram = np.zeros(256)
  for i in range(w):
    for j in range(h):
      iterations_count += 1
      histogram[image[i, j]] += 1

  # Calcula a probabilidade de cada nível de cinza.
  probabilities = histogram / np.sum(histogram)

  # Calcula a média da imagem.
  mean = np.sum(probabilities * np.arange(256))

  # Inicializa as variâncias entre as classes.
  variance_between_classes = 0

  # Itera sobre os níveis de cinza.
  best_threshold = 0

  # calcula o melhor limiar
  for i in range(1, 256):
    iterations_count += 1
    # Calcula a probabilidade acumulada.
    p0 = np.sum(probabilities[:i])
    p1 = 1 - p0

    # Calcula a média acumulada.
    m0 = np.sum(probabilities[:i] * np.arange(i))
    m1 = np.sum(probabilities[i:] * np.arange(i, 256))

    # Calcula a variância entre as classes.
    current_variance = p0 * (m0 - mean) ** 2 + p1 * (m1 - mean) ** 2

    # Atualiza o limiar ótimo.
    if current_variance > variance_between_classes:
      variance_between_classes = current_variance
      best_threshold = i
  
  # Limiariza a imagem.
  thresholded_image = np.zeros_like(image)
  
  for i in range(w):
    for j in range(h):
      iterations_count += 1
      if image[i, j] < best_threshold:
        thresholded_image[i, j] = 0
      else:
        thresholded_image[i, j] = 255
  
  return thresholded_image, iterations_count

# Carrega a imagem.
imageFile = 'teste.png'
image = cv2.imread(imageFile, cv2.IMREAD_GRAYSCALE)

print(f"{image.shape =}")
block_size = int(image.shape[0] / 8)
threshold = 15

print(f"{block_size =}")
print(f"{threshold =}")

# Aplica a limiarização adaptativa com a imagem integral.
start_time = time.time()
thresholded_integral_image, iterations_integral = adaptive_thresholding_integral_image(image, threshold=threshold, block_size=block_size)
end_time = time.time()
total_time = end_time - start_time

# Aplica a limiarização com o método de Otsu.
start_time = time.time()
threshold_otsu, iterations_otsu = otsu_thresholding(image)
end_time = time.time()
otsu_time = end_time - start_time

# Aplica a limiarização com wellners method
start_time = time.time()
threshold_wellners, iterations_wellners = wellners_method(image, threshold=threshold, s=block_size)
end_time = time.time()
wellners_time = end_time - start_time

# Aplica a limiarização adaptativa com a média.
start_time = time.time()
threshold_adaptive, iterations_mean = adaptive_thresholding_mean(image, threshold=threshold, block_size=block_size)
end_time = time.time()
adaptive_time = end_time - start_time

# Exibe as imagens
# print("salvando as imagens")
# cv2.imshow('Imagem original', image)
# cv2.imshow('Imagem limiarizada Otsu', threshold_otsu)
# cv2.imshow('Imagem limiarizada Wellners', threshold_wellners)
# cv2.imshow('Imagem limiarizada adaptativa', threshold_adaptive)
# cv2.imshow('Imagem limiarizada integral', thresholded_integral_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Imprime os tempos de execução
print("\nTempo de execução e iterações:")
print(f"Otsu      {otsu_time:.4f} s     {iterations_otsu} iterações")
print(f"Wellners  {wellners_time:.4f}s  {iterations_wellners} iterações")
print(f"Média     {adaptive_time:.4f}s  {iterations_mean} iterações")
print(f"Integral  {total_time:.4f}s     {iterations_integral} iterações")

# salva as imagens
cv2.imwrite(f"{imageFile}_integral.jpg", thresholded_integral_image)
cv2.imwrite(f"{imageFile}_otsu.jpg", threshold_otsu)
cv2.imwrite(f"{imageFile}_adaptive.jpg", threshold_adaptive)
cv2.imwrite(f"{imageFile}_wellners.jpg", threshold_wellners)

print("\nFIM\n")
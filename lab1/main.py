# Луценко Александр 9 группа 3 курс
# Задача №12 (Равномерное выравнивание гистограммы яркости)

# С использованием готовых функций из библиотек
# import cv2
# import numpy as np

# def histogram_equalization(image):

#     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

#     hist, bins = np.histogram(hsv[:,:,2].flatten(),256,[0,256])

#     cdf = hist.cumsum()
#     cdf_normalized = cdf * hist.max() / cdf.max()
    
#     cdf_m = np.ma.masked_equal(cdf,0)
#     cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
#     cdf = np.ma.filled(cdf_m,0).astype('uint8')

#     hsv[:,:,2] = cdf[hsv[:,:,2]]
    
#     equalized_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
#     return equalized_image

# # Подключение
# image = cv2.imread('image.jpg')

# equalized_image = histogram_equalization(image)

# # Отображение результатов
# cv2.imshow('Original Image', image)
# cv2.imshow('Equalized Image', equalized_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# v 2.0 Без использования готовых методов
import cv2
import numpy as np
import matplotlib.pyplot as plt

def histogram_equalization(image):
    # Создаем список для хранения гистограммы яркости
    histogram = [0] * 256
    
    # Вычисляем количество пикселей в изображении
    total_pixels = image.shape[0] * image.shape[1]
    
    # Вычисляем гистограмму яркости
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            pixel_value = image[i, j]
            histogram[pixel_value] += 1
    
    # Вычисляем кумулятивную гистограмму
    cumulative_histogram = [sum(histogram[:i+1]) for i in range(len(histogram))]
    
    # Вычисляем минимальное значение яркости
    min_value = min(cumulative_histogram)
    
    # Вычисляем масштабирующий коэффициент
    scale = 255 / (total_pixels - min_value)
    
    # Применяем равномерное выравнивание гистограммы
    equalized_image = image.copy()
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            pixel_value = image[i, j]
            equalized_value = int((cumulative_histogram[pixel_value] - min_value) * scale)
            equalized_image[i, j] = equalized_value
    
    return equalized_image

# Загружаем изображение
input_image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Применяем равномерное выравнивание гистограммы
equalized_image = histogram_equalization(input_image)

# Выводим изображение до и после выравнивания гистограммы
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(input_image, cmap='gray')
plt.title('Input Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(equalized_image, cmap='gray')
plt.title('Equalized Image')
plt.axis('off')

plt.show()
# Луценко Александр 3 курс 9 группа
# Тип фигуры: часть полушара заданного диаметра и c заданным углом;
# Модель освещения фигуры: несколько бесконечно удалённых источников с заданным направлением света.
# Тип проекции: параллельная (задаются углы поворота);

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def hemisphere(diameter, angle):
    # Создаем массив значений углов theta и phi
    theta = np.linspace(0, np.pi/2, 100)
    phi = np.linspace(0, angle*np.pi/180, 100)

    # Создаем сетку для значений theta и phi
    theta, phi = np.meshgrid(theta, phi)

    # Вычисляем координаты x, y, z для точек на поверхности полушара
    x = diameter/2 * np.sin(theta) * np.cos(phi)
    y = diameter/2 * np.sin(theta) * np.sin(phi)
    z = diameter/2 * np.cos(theta)

    return x, y, z

# Параметры части полушара
diameter = 10  # Диаметр полушара
angle = 90  # Угол (в градусах)

# Создаем данные для части полушара
x, y, z = hemisphere(diameter, angle)

# Визуализируем часть полушара
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Добавляем источники света
light_source_1 = np.array([-10, -10, 10])  # Координаты источника света 1
light_source_2 = np.array([10, 10, 10])    # Координаты источника света 2
ax.scatter(light_source_1[0], light_source_1[1], light_source_1[2], color='yellow', s=100)  # Источник света 1
ax.scatter(light_source_2[0], light_source_2[1], light_source_2[2], color='yellow', s=100)  # Источник света 2

# Включаем параллельную проекцию и устанавливаем углы поворота
ax.view_init(elev=30, azim=45)

# Построение поверхности полушара
ax.plot_surface(x, y, z, alpha=0.7)

# Создаем массивы координат для лучей света от источников света к каждой точке поверхности фигуры
light_ray_1_x = np.concatenate((np.full_like(x.flatten(), light_source_1[0]), x.flatten()))
light_ray_1_y = np.concatenate((np.full_like(y.flatten(), light_source_1[1]), y.flatten()))
light_ray_1_z = np.concatenate((np.full_like(z.flatten(), light_source_1[2]), z.flatten()))

light_ray_2_x = np.concatenate((np.full_like(x.flatten(), light_source_2[0]), x.flatten()))
light_ray_2_y = np.concatenate((np.full_like(y.flatten(), light_source_2[1]), y.flatten()))
light_ray_2_z = np.concatenate((np.full_like(z.flatten(), light_source_2[2]), z.flatten()))

# Визуализация лучей света
ax.plot(light_ray_1_x, light_ray_1_y, light_ray_1_z, color='white', alpha=0.3)
ax.plot(light_ray_2_x, light_ray_2_y, light_ray_2_z, color='red', alpha=0.3)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('Part of a Hemisphere')
plt.show()
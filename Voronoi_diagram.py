#!/usr/bin/env python
# -*- coding: UTF-8 -*-

# импортируем необходимые библиотеки
import random
import matplotlib.pyplot as plt
import numpy as np
from scipy import spatial
import uuid
from shapely.geometry import LineString
from shapely.geometry import Point


def count_point(R): # функция для вычисления числа точек
    try:
        N = int(float(R)) * 10
        return N
    except ValueError:
        print('The entered radius is not valid')


def get_radius(R):
    try:
        rad = float(R)
        return rad
    except ValueError:
        print('The entered radius is not valid')


print('Input circle radius:')

R = input()  # радиус окружности
num_of_points = count_point(R)  # число точек
rad = get_radius(R) # радиус переводим число

list_of_coordinates = []
for i in range(num_of_points):
    x = random.uniform(-rad, rad)
    y = random.uniform(-(rad - abs(x)), abs(rad) - abs(x))
    list_of_coordinates.append([x, y]) # список точек

points = np.array(list_of_coordinates) # переводим в numpy для удобства вычисления

vor = spatial.Voronoi(list_of_coordinates) # вычисляем диаграммы Вороного

guids = {}
for i in range(len(points)):
    guids[i] = uuid.uuid1()

vertices = vor.vertices.copy() # извлекаем вершины диаграмм Вороного
# Рисуем исходные точки и вершины диаграмм
plt.plot(points[:, 0], points[:, 1], 'o')
plt.plot(vor.vertices[:, 0], vor.vertices[:, 1], '*')
plt.xlim(-(rad + 1), rad + 1); plt.ylim(-(rad + 1), rad + 1)

p = Point(0, 0) # cоздаем геометрическую фигуру - круг
c = p.buffer(rad).boundary # делаем из круга окружность

# В данном цикле вычисляем все вершины диаграммы Вороного, причем
# если некоторые из них выходят за пределы круга, то ищем координаты точек пересечения с окружностью
# и запоминаем их
for simplex in vor.ridge_vertices:
    # пробегаем по всем точкам, между которыми будут проводиться перпендикуляры
    simplex = np.asarray(simplex)
    # если индекс точки равен -1,следовательно, уходит в бесконечность, с такими точками разберемся ниже
    if np.all(simplex >= 0):
        plt.plot(vertices[simplex, 0], vertices[simplex, 1], 'k-')
        # Проверяем выходит ли точка за пределы окружности
        if vertices[simplex, 0][0] ** 2 + vertices[simplex, 1][0] ** 2 > 4:
            # Соседняя точка должна не выходить за пределы окружности, так как иначе она не представляет интерес
            if vertices[simplex, 0][1]**2 + vertices[simplex, 1][1]**2 <= 4:
                # получаем х-координаты и y-координаты отрезка, выходящего за окружность
                tmp_vertice_i_neighbor = (vertices[simplex, 0][0], vertices[simplex, 1][0])
                tmp_vertice_i = (vertices[simplex, 0][1], vertices[simplex, 1][1])
                # задаем прямую по двум точкам
                l = LineString([tmp_vertice_i, tmp_vertice_i_neighbor])
                # ищем точки пересечения с окружностью
                inter = c.intersection(l)
                i_np = np.array(inter)
                if len(i_np) != 0:
                    plt.scatter(i_np[0], i_np[1], marker='*', c='g')
                    # меняем значение вершин, сохраняя в них точки пересечения
                    vertices[simplex, 0] = [i_np[0], vertices[simplex, 0][1]]
                    vertices[simplex, 1] = [i_np[1], vertices[simplex, 1][1]]
        # проверяем выходит ли за пределы круга соседняя точка
        elif vertices[simplex, 0][1] ** 2 + vertices[simplex, 1][1] ** 2 > 4 and len(vertices[simplex, 1]) != 0:
            tmp_vertice_i_neighbor = (vertices[simplex, 0][0], vertices[simplex, 1][0])
            tmp_vertice_i = (vertices[simplex, 0][1], vertices[simplex, 1][1])
            l = LineString([tmp_vertice_i, tmp_vertice_i_neighbor])
            i = c.intersection(l)
            i_np = np.array(i)
            if len(i_np) != 0:
                plt.scatter(i_np[0], i_np[1], marker='*', c='g')
                vertices[simplex, 0] = [vertices[simplex, 0][0], i_np[0]]
                vertices[simplex, 1] = [vertices[simplex, 1][0], i_np[1]]

# делаем словарь из наших исходных точек и соответствующих им координат вершин диаграммы Вороного
# ключ - индекс точки в исходном списке, значение - координаты ограничивающих её вершин
dict_of_points = {}
for i, ind in zip(range(len(points)), vor.regions):
    dict_of_points[i] = list()
    for j in range(len(ind)):
        if all(vertices[ind[j]]) > rad:
            dict_of_points[i].append(vertices[ind[j]])

# рисуем диаграммы Вороного
#spatial.voronoi_plot_2d(vor, show_vertices=False, line_colors='orange',line_width=2)

# тут проецируем удаленные в бесконечность вершины диаграммы Вороного на окружность
center = points.mean(axis=0)
# cоздаем словарь граничных точек
# ключ - индекс точки в исходных данных, значение - координаты точек, перескающих её область с окружностью с окружностью
boundary_points = {}
for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
    simplex = np.asarray(simplex)
    if np.any(simplex < 0):
        i = simplex[simplex >= 0][0]
        t = points[pointidx[1]] - points[pointidx[0]]  # касательная
        t = t / np.linalg.norm(t)
        n = np.array([-t[1], t[0]]) # нормаль
        midpoint = points[pointidx].mean(axis=0)
        far_point = vor.vertices[i] + np.sign(np.dot(midpoint - center, n)) * n * 100
        # убеждаемся, что точка действительно выходит за пределы окружности
        if far_point[0]**2 + far_point[1]**2 > 4:
            if vor.vertices[i][0]**2 + vor.vertices[i][1]**2 <= 4:
                # получаем х-координаты и y-координаты отрезка, выходящего за окружность
                tmp_ver_i_1 = (vor.vertices[i][0], vor.vertices[i][1])
                tmp_ver_i = (far_point[0], far_point[1])
                # задаем прямую по двум точкам
                l = LineString([tmp_ver_i, tmp_ver_i_1])
                # ищем точки пересечения с окружностью
                inter = c.intersection(l)
                i_np = np.array(inter)
                if len(i_np) != 0:
                    plt.scatter(i_np[0], i_np[1], marker='*', c='g')
                boundary_points[pointidx[0]] = i_np
                boundary_points[pointidx[1]] = i_np
        plt.plot([vor.vertices[i, 0], far_point[0]],
                     [vor.vertices[i, 1], far_point[1]], 'k--')

t = np.linspace(0, 2 * np.pi, 100) #исходная окружность
plt.plot(rad * np.cos(t), rad * np.sin(t))

plt.title('Voronoi diagram')
plt.show()

# добавляем найденные точки к точкам в словарь, удаляя при этом "далекие" вершины
for now in boundary_points:
    for tmp in dict_of_points:
        if tmp == now:
            dict_of_points[tmp].append(boundary_points[now])

# Сделаем guid'ы - ключом словаря
total_result = {}

for now, tmp in zip(guids, dict_of_points):
    total_result[guids[now]] = dict_of_points[tmp]

# Выводим финальный результат
print(total_result)

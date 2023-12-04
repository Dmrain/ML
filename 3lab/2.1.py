import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.datasets import fetch_california_housing

# Данные о жилье в Калифорнии загружаются с помощью fetch_california_housing().data,
# и целевой вектор (стоимость жилья) загружается с помощью fetch_california_housing().target.
# Данные затем нормализуются с помощью функции scale для обеспечения одного масштаба.
california = scale(fetch_california_housing().data)
target = fetch_california_housing().target
# Создается объект KFold для проведения кросс-валидации.
# Указывается разбиение на 5 блоков с фиксированным random_state для воспроизводимости результатов.
kf = KFold(n_splits=5, random_state=42, shuffle=True)
#Параметр shuffle=True в контексте KFold (и других кросс-валидационных методов в scikit-learn) означает,
# что данные будут случайно перемешаны перед разбиением на блоки. В других словах, порядок объектов в
# наборе данных будет случайным образом изменен перед тем, как он будет разделен на блоки для
# кросс-валидации. Это полезное свойство, так как оно помогает избежать возможных проблем,
# которые могут возникнуть, если данные остаются упорядоченными. Если данные упорядочены
# по какому-либо признаку (например, времени), и кросс-валидация выполняется без перемешивания (shuffle=False),
# то в одном из блоков могут попасть только объекты с низкими или высокими значениями этого признака.
# Это может привести к искаженным оценкам модели.

best_p = 0
best_score = -float('inf')
# Далее, итерируясь по различным значениям параметра p в диапазоне от 1 до 10 с 15 равномерно распределенными точками, выполняется следующее:

# Создается модель KNeighborsRegressor с параметрами: 5 соседей (n_neighbors=5), веса,
# зависящие от расстояния до соседей (weights='distance'), метрика Минковского с заданным значением p.

# Затем применяется кросс-валидация с помощью cross_val_score для оценки качества модели.
# Оценка основана на среднеквадратичной ошибке (scoring='neg_mean_squared_error'), и результаты сохраняются в переменной scores.

# Для каждого значения p, вычисляется средняя оценка качества mean_score по всем блокам кросс-валидации.

for p in np.linspace(1, 10, 10):
    neigh = KNeighborsRegressor(n_neighbors=5, weights='distance', metric='minkowski', p=p)
    scores = cross_val_score(neigh, california, target, cv=kf, scoring='neg_mean_squared_error')
    mean_score = scores.mean()

    # Если текущее значение mean_score оказывается лучше (меньше ошибки) чем лучший ранее найденный результат,
    # то значение best_score обновляется, и значение p сохраняется в best_p.
    if mean_score > best_score:
        best_score = mean_score
        best_p = p

print(best_p)
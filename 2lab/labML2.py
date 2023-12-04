import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# Задание 1
titanic_data = pd.read_csv('titanic.csv')
#Здесь создается объект DataFrame (titanic_data), который содержит данные из файла 'titanic.csv'.
#DataFrame - это структура данных библиотеки pandas, предназначенная для работы с табличными данными.

# Задание 2
selected_features = ['Pclass', 'Fare', 'Age', 'Sex']
titanic_data = titanic_data[selected_features + ['Survived']]
#Выбираются определенные признаки ('Pclass', 'Fare', 'Age', 'Sex') и целевая переменная ('Survived') для анализа.
# Остаются только эти признаки в DataFrame titanic_data.

# Задание 3
# Преобразование строкового значения 'Sex' в числовое
titanic_data['Sex'] = titanic_data['Sex'].map({'male': 0, 'female': 1})
#Колонка 'Sex' преобразуется из строкового формата (значения 'male' и 'female') в числовой формат, где 'male' становится 0, а 'female' становится 1.

# Задание 4
X = titanic_data.drop('Survived', axis=1)
y = titanic_data['Survived']
#Данные разделяются на матрицу признаков X (все столбцы, кроме 'Survived') и целевую переменную y ('Survived').

# Задание 5
# Удаление объектов с пропущенными значениями
X = X.dropna()
y = y[X.index]
#Удаляются строки, содержащие пропущенные значения. X.dropna() удаляет строки из матрицы признаков, а y[X.index] обновляет целевую переменную соответственно.

# Задание 6
clf = DecisionTreeClassifier(random_state=241)
clf.fit(X, y)
#Создается и обучается модель DecisionTreeClassifier с использованием данных X и y. Аргумент random_state=241 устанавливает случайное начальное состояние для воспроизводимости результатов.

# Задание 7
importances = clf.feature_importances_
important_features = X.columns[np.argsort(importances)[::-1]][:2]
print("Two most important features:", important_features.tolist())
#Оценка важности признаков в модели. clf.feature_importances_ возвращает массив, содержащий важности каждого признака.
# np.argsort(importances)[::-1] возвращает индексы признаков в порядке убывания важности.
# [:2] выбирает два наиболее важных признака. Результат выводится на экран.

# Задание 8
# Предсказание выживания для нового человека
new_passenger = pd.DataFrame([[1, 100, 25, 1]], columns=['Pclass', 'Fare', 'Age', 'Sex'])
prediction = clf.predict(new_passenger)
print("Survived" if prediction[0] == 1 else "Not Survived")
#Создание DataFrame new_passenger, представляющего нового пассажира с указанными значениями признаков ('Pclass', 'Fare', 'Age', 'Sex').
# Затем используется обученная модель (clf.predict()) для предсказания выживания нового пассажира.
# Результат выводится на экран в виде "Survived" или "Not Survived".
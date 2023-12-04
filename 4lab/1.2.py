import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Загрузка данных
data = pd.read_csv('apples_pears.csv')

# Построение изображения набора данных "Яблоки-Груши"
plt.figure(figsize=(10, 8))
plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c=data['target'], cmap='rainbow')
plt.title('Яблоки и груши', fontsize=15)
plt.xlabel('симметричность', fontsize=14)
plt.ylabel('желтизна', fontsize=14)
plt.show()

# Выделение матрицы признаков (X) и вектора ответов (y)
X = data.iloc[:, :2]
y = data['target']

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение персептрона
clf = Perceptron()
clf.fit(X_train, y_train)

# Построение изображения набора данных "Яблоки-Груши" с учетом результатов классификации
plt.figure(figsize=(10, 8))
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=clf.predict(X), cmap='spring')
plt.title('Яблоки и груши с результатами классификации', fontsize=15)
plt.xlabel('симметричность', fontsize=14)
plt.ylabel('желтизна', fontsize=14)
plt.show()

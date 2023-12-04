import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Загрузка обучающей и тестовой выборок
train_data = pd.read_csv('perceptron-train.csv', header=None)
test_data = pd.read_csv('perceptron-test.csv', header=None)

# Разделение на целевую переменную и признаки
X_train = train_data.iloc[:, 1:]
y_train = train_data.iloc[:, 0]

X_test = test_data.iloc[:, 1:]
y_test = test_data.iloc[:, 0]

# Обучение персептрона
clf = Perceptron(random_state=241)
clf.fit(X_train, y_train)

# Подсчет качества на тестовой выборке
predictions_before_scaling = clf.predict(X_test)
accuracy_before_scaling = accuracy_score(y_test, predictions_before_scaling)

# Нормализация данных
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Обучение персептрона на нормализованных данных
clf.fit(X_train_scaled, y_train)

# Подсчет качества на тестовой выборке после нормализации
predictions_after_scaling = clf.predict(X_test_scaled)
accuracy_after_scaling = accuracy_score(y_test, predictions_after_scaling)

# Разность между качеством до и после нормализации
difference = accuracy_after_scaling - accuracy_before_scaling

print("Accuracy before scaling:", accuracy_before_scaling)
print("Accuracy after scaling:", accuracy_after_scaling)
print("Difference:", difference)

import pandas as pd

# Загрузка датасета
titanic_data = pd.read_csv('E:\\ML\\1lab\\titanic.csv')

# 1. Количество мужчин и женщин на корабле
gender_counts = titanic_data['Sex'].value_counts()
male_count = gender_counts['male']
female_count = gender_counts['female']
#объект gender_counts, который содержит количество уникальных значений в столбце "Sex" (пол) датасета titanic_data.
#male_count = gender_counts['male']: Затем переменной male_count присваивается количество мужчин, извлеченное из gender_counts.
#female_count = gender_counts['female']: Аналогично, переменной female_count присваивается количество женщин.


# 2. Доля выживших пассажиров
survival_rate = (titanic_data['Survived'].sum() / len(titanic_data)) * 100
#survival_rate = (titanic_data['Survived'].sum() / len(titanic_data)) * 100: Рассчитывается доля выживших пассажиров.
#titanic_data['Survived'].sum() возвращает количество выживших, а len(titanic_data) - общее количество пассажиров. Умножение на 100 преобразует долю в проценты.

# 3. Доля пассажиров первого класса
first_class_percentage = (titanic_data['Pclass'] == 1).mean() * 100
#first_class_percentage = (titanic_data['Pclass'] == 1).mean() * 100: Рассчитывается доля пассажиров первого класса.
# titanic_data['Pclass'] == 1 создает булеву серию, где True соответствует первому классу. mean() вычисляет среднее значение булевой серии, что равно доле первого класса.
# Умножение на 100 преобразует долю в проценты.

# 4. Средний и медианный возраст пассажиров
average_age = titanic_data['Age'].mean()
median_age = titanic_data['Age'].median()
#average_age = titanic_data['Age'].mean(): Рассчитывается средний возраст пассажиров.
#median_age = titanic_data['Age'].median(): Рассчитывается медианный возраст пассажиров.

# 5. Корреляция между SibSp и Parch (число братьев/сестер и число родителей/детей)
correlation = titanic_data['SibSp'].corr(titanic_data['Parch'], method='pearson')
# Рассчитывается корреляция между количеством братьев/сестер (SibSp) и количеством родителей/детей (Parch) с использованием метода Пирсона (параметр method='pearson')


# 6. Самое популярное женское имя
def extract_first_name(name):
    if '(' in name:
        start = name.find('(') + 1
        end = name.find(')')
        return name[start:end]
    else:
        name_parts = name.split('.')
        if len(name_parts) > 1:
            first_name = name_parts[1].split(' ')[1]  # Исправлено здесь
            return first_name
        else:
            return name
#Описание: Извлекает первое имя из строки, представляющей полное имя пассажира.
#Параметры:
#name: Строка с именем пассажира.
#Внутренняя логика:
#Если в строке присутствует скобка, извлекает имя, находящееся между скобками.
#В противном случае, разбивает строку по точке (.) и пробелу, а затем извлекает вторую часть, содержащую имя.
#Возвращаемое значение: Извлеченное имя.

female_passengers = titanic_data[titanic_data['Sex'] == 'female'].copy()
female_passengers['First Name'] = female_passengers['Name'].apply(extract_first_name)
most_common_female_name = female_passengers['First Name'].value_counts().idxmax()
most_common_female_name = female_passengers['First Name'].mode()[0]
#Создает новый столбец First Name в датафрейме female_passengers, который содержит извлеченные имена для женских пассажиров.
#Внутренняя логика:
#Применяет функцию extract_first_name к столбцу Name датафрейма female_passengers.
#Результат: Датафрейм female_passengers теперь содержит дополнительный столбец First Name с извлеченными именами.
#Находит самое частое женское имя среди извлеченных имен в столбце First Name.
#Использует метод .value_counts() для подсчета уникальных значений.
#Метод .idxmax() возвращает индекс с наибольшим значением (частотой).
#Метод .mode()[0] также возвращает самое частое значение (режим) в случае, если их несколько.



# Вывод ответов на вопросы
print(f"1. {male_count} {female_count}")
print(f"2. {survival_rate:.2f}")
print(f"3. {first_class_percentage:.2f}")
print(f"4. {average_age:.2f} {median_age:.2f}")
print(f"5. {correlation:.2f}")
print(f"6. Самое популярное женское имя: {most_common_female_name}")

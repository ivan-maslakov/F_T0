import pandas as pd
import numpy as np


# 1. Просмотр строк
df = pd.read_csv('Titanic.csv')
#print(df.columns)

# Первые 10 строк
print("Первые 10 строк:\n", df.head(10))

# Последние 10 строк
print("\nПоследние 10 строк:\n", df.tail(10))

# Случайные 5 строк (воспроизводимость с помощью seed)
np.random.seed(42)
random_rows = df.sample(5)
print("\nСлучайные 5 строк:\n", random_rows)

# 2. Средний и медианный возраст
print("\nСредний возраст:", df['Age'].mean())
print("Медианный возраст:", df['Age'].median())

print("\n--- Средний возраст по полу ---")
print(df.groupby('Gender')['Age'].mean())
print("\n--- Медианный возраст по полу ---")
print(df.groupby('Gender')['Age'].median())

# 3. Число пассажиров по возрастным группам
age_group1 = df[df['Age'] <= 25].shape[0]
age_group2 = df[(df['Age'] > 25) & (df['Age'] <= 35)].shape[0]
age_group3 = df[df['Age'] > 35].shape[0]

print("\nПассажиров до 25 лет:", age_group1)
print("Пассажиров от 25 до 35 лет:", age_group2)
print("Пассажиров старше 35 лет:", age_group3)

# 4. Пассажиры с одинаковыми именами
print("\nЕсть ли пассажиры с одинаковыми именами:", df['Name'].duplicated().any())

# 5. Аномалии
print("\nСтатистики:\n", df.describe())
# Можно заметить, что максимальное значение Fare значительно превышает 75-й процентиль,
# что может указывать на выбросы.

# Обработка выбросов в Fare (замена на медиану)
median_fare = df['Fare'].median()
df['Fare'] = np.where(df['Fare'] > 3 * median_fare, median_fare, df['Fare'])

# 6. Процент спасшихся по возрастным группам
bins = [0, 18, 30, 50, 100]
labels = ['Дети', 'Молодые', 'Взрослые', 'Пожилые']
df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)

survival_by_age = df.groupby('AgeGroup')['Survived'].mean() * 100
print("\nПроцент спасшихся по возрастным группам:\n", survival_by_age)
survival_by_age.to_csv('survival_by_age.csv')

# 7. Процент спасшихся по стоимости билета
median_fare = df['Fare'].median()
df['FareCategory'] = np.where(df['Fare'] > median_fare, 'High', 'Low')

survival_by_fare = df.groupby('FareCategory')['Survived'].mean() * 100
print("\nПроцент спасшихся по стоимости билета:\n", survival_by_fare)
survival_by_fare.to_csv('survival_by_fare.csv')

# 8. Пассажиры с определенным возрастом и стоимостью билета
filtered_passengers = df[((df['Age'] == 22) | (df['Age'] == 26) | (df['Age'] == 32)) &
                        (df['Fare'] >= 30) & (df['Fare'] <= 300)]

print("\nЧисло пассажиров с возрастом 22, 26 или 32 года и билетом от 30 до 300 долларов:",
      filtered_passengers.shape[0])


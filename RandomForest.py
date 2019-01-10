import numpy as np
import urllib
# заносим в переменную url ссылку на данные
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data"
# загружаем данные
raw_data = urllib.request.urlopen(url)
dataset = np.loadtxt(raw_data, delimiter=",")
# разделим данные на целевую переменную y и матрицу признаков X
X = dataset[:,0:10]
y = dataset[:,10]
# загрузим библиотеки для использования случайного леса
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
# Разбиваем данные на обучающую и тестовую выборку. Обучаем данные
forest = RandomForestClassifier()
trainX, testX, trainY, testY = train_test_split( X, y, test_size = 0.3)
forest.fit(trainX, trainY)
# Выводим прогноз и его точность на тестовой выборке
print('Accuracy: \n', forest.score(testX, testY))
pred = forest.predict(testX)
print(pred)

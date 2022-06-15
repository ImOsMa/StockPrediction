import tensorflow as tf
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split

from data_fetcher import get_stock_data_all, get_X_Y_all
from constantas import *
from scaller import *


warnings.filterwarnings('ignore')

X_data = {
    'x': [],
    'x_all': [],
    'x_1': [],
    'x_2': [],
    'x_3': [],
    'x_4': [],
    'x_5': []
}

y, new_y, scal_y = [], [], []
scal_x_dict = {
    'scal_x': [],
    'scal_x_all': [],
    'scal_x_1': [],
    'scal_x_2': [],
    'scal_x_3': [],
    'scal_x_4': [],
    'scal_x_5': [],
}

new_x_dict = {
    'new_x': [],
    'new_x_all': [],
    'new_x_1': [],
    'new_x_2': [],
    'new_x_3': [],
    'new_x_4': [],
    'new_x_5': [],
}

get_stock_data_all(TICKER, START, END, train_file_all)
get_stock_data_all(TICKER, START_TEST, END_TEST, test_file_all)
get_stock_data_all(TICKER, START, END_TEST, scal_file_all)
data = pd.read_csv(train_file_all, encoding='utf-8')
get_X_Y_all(data, seq_len, X_data, y, '')
new_data = pd.read_csv(test_file_all, encoding='utf-8')
get_X_Y_all(new_data, seq_len, new_x_dict, new_y, 'new')
scal_data = pd.read_csv(scal_file_all, encoding='utf-8')
get_X_Y_all(scal_data, seq_len, scal_x_dict, scal_y, 'scal')

for k, v in X_data:
    X_data[k] = np.array(v)
y = np.array(y)

for k, v in scal_x_dict:
    scal_x_dict[k] = np.array(v)
scal_y = np.array(scal_y)

for k, v in new_x_dict:
    new_x_dict[k] = np.array(v)
new_y = np.array(new_y)



# Произведем скалинг данных

X = []
for k, v in X_data.values():
    X.append(v)

NEW_X = []
for k, v in NEW_X.values():
    NEW_X.append(v)
data_scalling(X, NEW_X)
x_pca, x_all_pca, scaler_y = additional_pca_new_scalling(scal_y, y, X_data, new_x_dict, scal_x_dict)


# Разобьем тренировочные данные на обучение и проверку в соотношении 9 к 1 и перемешаем их.
X_train, X_valid, y_train, y_valid = train_test_split(X_data.get('x'), y, test_size=0.1, random_state=42, shuffle=True)
X_train_all, X_valid_all, y_train_all, y_valid_all = train_test_split(X_data.get('x_all'), y, test_size=0.1,
                                                                      random_state=42,
                                                                      shuffle=True)
X_train_1, X_valid_1, y_train_1, y_valid_1 = train_test_split(X_data.get('x_1'), y, test_size=0.1, random_state=42,
                                                              shuffle=True)
X_train_2, X_valid_2, y_train_2, y_valid_2 = train_test_split(X_data.get('x_2'), y, test_size=0.1, random_state=42,
                                                              shuffle=True)
X_train_3, X_valid_3, y_train_3, y_valid_3 = train_test_split(X_data.get('x_3'), y, test_size=0.1, random_state=42,
                                                              shuffle=True)
X_train_4, X_valid_4, y_train_4, y_valid_4 = train_test_split(X_data.get('x_4'), y, test_size=0.1, random_state=42,
                                                              shuffle=True)
X_train_5, X_valid_5, y_train_5, y_valid_5 = train_test_split(X_data.get('x_5'), y, test_size=0.1, random_state=42,
                                                              shuffle=True)
X_train_6, X_valid_6, y_train_6, y_valid_6 = train_test_split(X_data.get('x_6'), y, test_size=0.1, random_state=42,
                                                              shuffle=True)
X_train_pca, X_valid_pca, y_train_pca, y_valid_pca = train_test_split(x_pca, y, test_size=0.1, random_state=42,
                                                                      shuffle=True)
X_train_all_pca, X_valid_all_pca, y_train_all_pca, y_valid_all_pca = train_test_split(x_all_pca, y, test_size=0.1,
                                                                                      random_state=42, shuffle=True)

print("Shape1:", X_train.shape)
print("Shape2:", X_train_all.shape)
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(100, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(100, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(1, activation=tf.nn.relu))



model.compile(optimizer="adam", loss="mean_squared_error")

model.fit(X_train, y_train, epochs=100)

print(model.evaluate(X_valid, y_valid))

#Предскажем цены с помощью обученной модели и произведем обратный скалинг.
y1 = scaler_y.inverse_transform(model.predict(X_valid))
y2 = scaler_y.inverse_transform(model.predict(new_x_dict.get('new_x')))
y3 = scaler_y.inverse_transform(y_valid)

print(y2)

#Запишем данные в файлы, для дальнейшего сравнения и построения графиков в другой части программы.

# np.save('MLP_y_pred', y1)
# np.save('MLP_new_y', y2)
# np.save('MLP_y', y3)
#
# X_train_all = np.array([x.flatten() for x in X_train_all])
# X_valid_all = np.array([x.flatten() for x in X_valid_all])
# new_x_all = np.array([x.flatten() for x in new_x_all])
# X_train_1 = np.array([x.flatten() for x in X_train_1])
# X_valid_1 = np.array([x.flatten() for x in X_valid_1])
# new_x_1 = np.array([x.flatten() for x in new_x_1])
# X_train_2 = np.array([x.flatten() for x in X_train_2])
# X_valid_2 = np.array([x.flatten() for x in X_valid_2])
# new_x_2 = np.array([x.flatten() for x in new_x_2])
# X_train_3 = np.array([x.flatten() for x in X_train_3])
# X_valid_3 = np.array([x.flatten() for x in X_valid_3])
# new_x_3 = np.array([x.flatten() for x in new_x_3])
# X_train_4 = np.array([x.flatten() for x in X_train_4])
# X_valid_4 = np.array([x.flatten() for x in X_valid_4])
# new_x_4 = np.array([x.flatten() for x in new_x_4])
# X_train_5 = np.array([x.flatten() for x in X_train_5])
# X_valid_5 = np.array([x.flatten() for x in X_valid_5])
# new_x_5 = np.array([x.flatten() for x in new_x_5])
# X_train_6 = np.array([x.flatten() for x in X_train_6])
# X_valid_6 = np.array([x.flatten() for x in X_valid_6])
# new_x_6 = np.array([x.flatten() for x in new_x_6])
# X_train_pca = np.array([x.flatten() for x in X_train_pca])
# X_valid_pca = np.array([x.flatten() for x in X_valid_pca])
# new_x_pca = np.array([x.flatten() for x in new_x_pca])
# X_train_all_pca = np.array([x.flatten() for x in X_train_all_pca])
# X_valid_all_pca = np.array([x.flatten() for x in X_valid_all_pca])
# new_x_all_pca = np.array([x.flatten() for x in new_x_all_pca])
#
#
# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Dense(100, activation=tf.nn.relu))
# model.add(tf.keras.layers.Dense(100, activation=tf.nn.relu))
# model.add(tf.keras.layers.Dense(1, activation=tf.nn.relu))
# model.compile(optimizer="adam", loss="mean_squared_error")
#
# model.fit(X_train_all, y_train_all, epochs=100)
# print(model.evaluate(X_valid_all, y_valid_all))
# y1 = scaler_y.inverse_transform(model.predict(X_valid_all))
# y2 = scaler_y.inverse_transform(model.predict(new_x_all))
# print(y2)
# np.save('MLP_y_pred_all', y1)
# np.save('MLP_new_y_all', y2)
#
# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Dense(100, activation=tf.nn.relu))
# model.add(tf.keras.layers.Dense(100, activation=tf.nn.relu))
# model.add(tf.keras.layers.Dense(1, activation=tf.nn.relu))
# model.compile(optimizer="adam", loss="mean_squared_error")
#
# model.fit(X_train_1, y_train_1, epochs=100)
# print(model.evaluate(X_valid_1, y_valid_1))
# y1 = scaler_y.inverse_transform(model.predict(X_valid_1))
# y2 = scaler_y.inverse_transform(model.predict(new_x_1))
# print(y2)
# np.save('MLP_y_pred_1', y1)
# np.save('MLP_new_y_1', y2)
#
# model.fit(X_train_2, y_train_2, epochs=100)
# print(model.evaluate(X_valid_2, y_valid_2))
# y1 = scaler_y.inverse_transform(model.predict(X_valid_2))
# y2 = scaler_y.inverse_transform(model.predict(new_x_2))
# print(y2)
# np.save('MLP_y_pred_2', y1)
# np.save('MLP_new_y_2', y2)
#
# model.fit(X_train_3, y_train_3, epochs=100)
# print(model.evaluate(X_valid_3, y_valid_3))
# y1 = scaler_y.inverse_transform(model.predict(X_valid_3))
# y2 = scaler_y.inverse_transform(model.predict(new_x_3))
# print(y2)
# np.save('MLP_y_pred_3', y1)
# np.save('MLP_new_y_3', y2)
#
# model.fit(X_train_4, y_train_4, epochs=100)
# print(model.evaluate(X_valid_4, y_valid_4))
# y1 = scaler_y.inverse_transform(model.predict(X_valid_4))
# y2 = scaler_y.inverse_transform(model.predict(new_x_4))
# print(y2)
# np.save('MLP_y_pred_4', y1)
# np.save('MLP_new_y_4', y2)
#
# model.fit(X_train_5, y_train_5, epochs=100)
# print(model.evaluate(X_valid_5, y_valid_5))
# y1 = scaler_y.inverse_transform(model.predict(X_valid_5))
# y2 = scaler_y.inverse_transform(model.predict(new_x_5))
# print(y2)
# np.save('MLP_y_pred_5', y1)
# np.save('MLP_new_y_5', y2)
#
# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Dense(100, activation=tf.nn.relu))
# model.add(tf.keras.layers.Dense(100, activation=tf.nn.relu))
# model.add(tf.keras.layers.Dense(1, activation=tf.nn.relu))
# model.compile(optimizer="adam", loss="mean_squared_error")
#
# model.fit(X_train_all_pca, y_train_all_pca, epochs=100)
# print(model.evaluate(X_valid_all_pca, y_valid_all_pca))
# y1 = scaler_y.inverse_transform(model.predict(X_valid_all_pca))
# y2 = scaler_y.inverse_transform(model.predict(new_x_all_pca))
# print(y2)
# np.save('MLP_y_pred_all_pca', y1)
# np.save('MLP_new_y_all_pca', y2)
#
# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Dense(100, activation=tf.nn.relu))
# model.add(tf.keras.layers.Dense(100, activation=tf.nn.relu))
# model.add(tf.keras.layers.Dense(1, activation=tf.nn.relu))
# model.compile(optimizer="adam", loss="mean_squared_error")
#
# model.fit(X_train_pca, y_train_pca, epochs=100)
# print(model.evaluate(X_valid_pca, y_valid_pca))
# y1 = scaler_y.inverse_transform(model.predict(X_valid_pca))
# y2 = scaler_y.inverse_transform(model.predict(new_x_pca))
# print(y2)
# np.save('MLP_y_pred_pca', y1)
# np.save('MLP_new_y_pca', y2)
#
# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Dense(100, activation=tf.nn.relu))
# model.add(tf.keras.layers.Dense(100, activation=tf.nn.relu))
# model.add(tf.keras.layers.Dense(1, activation=tf.nn.relu))
# model.compile(optimizer="adam", loss="mean_squared_error")
#
# model.fit(X_train_6, y_train_6, epochs=100)
# print(model.evaluate(X_valid_6, y_valid_6))
# y1 = scaler_y.inverse_transform(model.predict(X_valid_6))
# y2 = scaler_y.inverse_transform(model.predict(new_x_6))
# print(y2)
# np.save('MLP_y_pred_6', y1)
# np.save('MLP_new_y_6', y2)

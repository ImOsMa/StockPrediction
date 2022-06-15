import yfinance as yf
import time
import numpy as np


def get_stock_data_all(ticker, start_date, end_date, file):
    yf.pdr_override()
    """
    Получает исторические данные по дневным ценам акций между датами
    :param ticker: компания или компании, чьи данные должны быть извлечены
    :type ticker: string or list of strings
    :param start_date: начальная дата получения цен на акции
    :type start_date: string of date "YYYY-mm-dd"
    :param end_date: конечная дата получения цен на акции
    :type end_date: string of date "YYYY-mm-dd"
    :param file: имя возвращаемого файла с данными
    :return: файл формата csv
    """
    i = 1
    all_data = 0
    while i > 0:
        try:
            all_data = yf.download(ticker, start_date, end_date)
            i = 0
        except ValueError:
            i += 1
            if i < 5:
                print("ValueError, trying again")
                time.sleep(10)
            else:
                print("Tried 5 times, Yahoo error. Trying after 2 minutes")
                i = 1
                time.sleep(120)
    print("download ok!")
    all_data.to_csv(file)


def get_X_Y_all(data, seq_len, X_dict, list_y, x_type=''):
    """
    Преобразует данные, разбивая на признаки и ответы.
    :param data: исходный массив данных
    :param seq_len: количество признаков
    :param list_x: список, в который добавляются признаки
    :param list_y: список, в который добавляются ответы
    """

    x_str = x_type + '_'
    for i in range(len(data) - seq_len):
        X_dict[x_str + 'x'].append(np.array(data.iloc[i: i + seq_len, 5]))
        list_y.append(np.array([data.iloc[i + seq_len, 5]], np.float64))
        X_dict[x_str + 'x_all'].append(np.array(data.iloc[i: i + seq_len, lambda df: [5, 4, 2, 3, 1, 6]]))
        X_dict[x_str + 'x_1'].append(np.array(data.iloc[i: i + seq_len, lambda df: [5, 2, 3, 1, 6]]))
        X_dict[x_str + 'x_2'].append(np.array(data.iloc[i: i + seq_len, lambda df: [5, 4, 3, 1, 6]]))
        X_dict[x_str + 'x_3'].append(np.array(data.iloc[i: i + seq_len, lambda df: [5, 4, 2, 1, 6]]))
        X_dict[x_str + 'x_4'].append(np.array(data.iloc[i: i + seq_len, lambda df: [5, 4, 2, 3, 6]]))
        X_dict[x_str + 'x_5'].append(np.array(data.iloc[i: i + seq_len, lambda df: [5, 4, 2, 3, 1]]))
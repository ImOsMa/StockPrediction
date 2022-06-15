"""
start и end - период, на котором будет проходить обучение.
start_test и end_test - период, на котором будет проходить предсказание.
seq_len - количество предыдущих дневных цен (признаков), на основе которых делается предсказание цены дня.
"""
START = "2003-01-01"
END = "2019-04-01"
START_TEST = "2019-03-19"
END_TEST = "2019-05-01"
TICKER = "BMW.DE"
train_file_all = "stock_prices_all.csv"
test_file_all = "test_all.csv"
scal_file_all = "scal_all.csv"
seq_len = 10

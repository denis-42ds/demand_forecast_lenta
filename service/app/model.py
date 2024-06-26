import pandas as pd
import pickle

# Загрузка модели
with open('model.pcl', 'rb') as f:
    model = pickle.load(f)

# Загрузка данных из файла shablon.xlsx
data = pd.read_csv('shablon.csv')
holiday = pd.read_csv('holidays_covid_calendar.csv')

holiday = holiday[['date', 'holiday']]
holiday['date'] = pd.to_datetime(holiday['date'], format='%d.%m.%Y')

data = data.merge(holiday, on='date', how='left')



def create_features(data, window, max_lag):

  try:
        data['day'] = data['date'].dt.day
        data['month'] = data['date'].dt.month
        data['year'] = data['date'].dt.year
        data['weekday'] = data['date'].dt.weekday

# Вычисляем скользящее среднее значение pr_sales_in_units для каждой группы
        data['rolling_average'] = (
            grouped_data['pr_sales_in_units']
            .rolling(window=window, min_periods=1)
            .mean()
            .reset_index(drop=True)
            )

        # Добавляем смещение на 14 дней к столбцу rolling_average
        #data['rolling_average'] = data.groupby(['st_id', 'pr_sku_id'])['rolling_average'].shift(max_lag) код для одного столбца

        for lag in range(1, max_lag + 1):
            data['lag_{}'.format(lag)] = data.groupby(['st_id', 'pr_sku_id'])['rolling_average'].shift(lag, fill_value=0)

data = create_features(data, 3, 14)

input_data = data.copy()  

predictions = model.predict(input_data)


results = predictions

data.to_csv('results.csv', index=False)

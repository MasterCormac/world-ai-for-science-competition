import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Concatenate, Flatten, Input, Lambda, LSTM
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import random
import datetime

# 1. Data Preprocessing
rng_seed = 2222
random.seed(rng_seed)
np.random.seed(rng_seed)
tf.random.set_seed(rng_seed)

e_price = pd.read_csv("data/electricity price.csv")
unit = pd.read_csv("data/unit.csv", encoding="gbk")

# add. data
extra_f = pd.DataFrame
try:
    extra_f = pd.read_csv("/app/tcdata/round2_test_data.csv")
except Exception as _e:
    extra_f = pd.read_csv("../tcdata/round2_test_data.csv")

e_price = pd.concat([e_price, extra_f], ignore_index=True)

# coal price
coal = pd.read_csv("data/comp price.csv")
coal = coal.drop(columns=['5500 rate', '5000 rate', '4500 rate'])

# power plant generation costs
unit['cost'] = unit[' coal consumption (g coal/KWh)'] / (1 - 0.01 * unit[' power consumption rate (%)'])
unit = unit.rename(columns={'Capacity（MW）': 'capacity', ' utilization hour (h)': 'u_hour'})

# Expand weekly coal price to daily
coal['Week_Start'] = coal['begin day']
coal['Week_Start'] = pd.to_datetime(coal['Week_Start'])
e_price['day'] = pd.to_datetime(e_price['day'])
coal_expanded = coal.set_index('Week_Start').reindex(e_price['day'], method='bfill')
coal_expanded = coal_expanded.reset_index()

# Merge
e_price['55p'] = coal_expanded['5500 price'] / 660.0
e_price['50p'] = coal_expanded['5000 price'] / 660.0
e_price['45p'] = coal_expanded['4500 price'] / 660.0
e_price['cost_rate'] = (e_price['55p'] + e_price['50p'] + e_price['45p']) / 3.0

print(e_price)

# Generate predicted demand (used for pricing models in power plants)
data = e_price[['demand']]
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)
look_back = 30

x_train = []
y_train = []
x_pred = []
for i in range(len(scaled_data) - look_back):
    x_train.append(scaled_data[i:(i + look_back)])
    y_train.append(scaled_data[i + look_back])

x_train = np.array(x_train)
y_train = np.array(y_train)

# Build LSTM
inputs = Input(shape=(look_back, 1))
lstm = LSTM(30, activation='relu')(inputs)
forget = Dropout(0.1)(lstm)
outputs = Dense(1)(forget)
model = Model(inputs=inputs, outputs=outputs)
model.compile(loss='mse', optimizer='adam')

model.fit(x_train, y_train, epochs=20)

# Predict demand
pred = model.predict(x_train)
predt = scaler.inverse_transform(pred)

# Update predicted demand
e_price['demand_hat'] = e_price['demand']
e_price.loc[30:, 'demand_hat'] = predt.flatten()
print(e_price)

# 2. Price Settlement Model: 
# Calculate the clearing price 
# based on the bids from each plant and their generation capacity

def clamp(input_, min_, max_):
    return max(min_, min(max_, input_))

# Generate bidding data f==
cost_price_rate = 1.091058
test_size = e_price.shape[0] - 3
unit_size = unit.shape[0]

X = []
Y = []

# bidding price with normal noise
def generate_input(demand_, center):
    res = []
    for index in range(unit_size):
        price = unit.at[index, 'cost'] * cost_price_rate + random.gauss(center, clamp(random.gauss(80, 10), 60, 100))
        price = clamp(price, 0, 1500)
        capacity = unit.at[index, 'capacity']
        res.append([price, capacity, demand_ / 100.0])
    return res


for i in range(test_size):
    flag = False
    demand = e_price.at[i, 'demand']
    dt = generate_input(demand, random.gauss(0, 50))
    X.append(dt)
    dt.sort(key=lambda o: o[0])
    capacity_sum = 0
    for item in dt:
        capacity_sum += item[1]
        if capacity_sum >= demand:
            Y.append(item[0])
            flag = True
            break
    if not flag:
        Y.append(1500)
    if i % 800 == 0:
        print(i / 800)

# Preprocess training data
X = np.array(X)
X = np.reshape(X, (-1, unit_size * 3))
scalerX = StandardScaler()
scaled_X = scalerX.fit_transform(X)
Y = np.array(Y)
Y = np.reshape(Y, (-1, 1))
scalerY = StandardScaler()
scaled_Y = scalerY.fit_transform(Y)

# test-train split
test_X = scaled_X[(test_size - 10):]
train_X = scaled_X[:(test_size - 10)]
test_Y = scaled_Y[(test_size - 10):]
train_Y = scaled_Y[:(test_size - 10)]

# Build price settlement model
price_model = Sequential([
    Dense(1024, activation='relu', input_shape=(unit_size * 3,), name='price_model_d1'),
    Dense(512, activation='relu', name='price_model_d2'),
    Dropout(0.15, name='price_model_drop'),
    Dense(1)
])
price_model.compile(optimizer='adam', loss='mse')
price_model.fit(train_X, train_Y, epochs=30)

# Predict prices
pred_Y = price_model.predict(test_X)
pred_Y = scalerY.inverse_transform(pred_Y)
print(pred_Y)

# Compute MSE
test_Y = scalerY.inverse_transform(test_Y)
print(test_Y)
print(tf.reduce_mean(tf.square(test_Y - pred_Y)))


# 3. Power plant bidding model: 
# Simulate the bidding of each plant

ans_eprice = e_price[e_price.isnull().any(axis=1)]
train_eprice = e_price.dropna(axis=0, how='any')

# Freeze the training of the price settlement model
price_model.trainable = False


# generate test and train
def create_model():
    inputs = Input(shape=(3,))
    x1 = Dense(16, activation='relu')(inputs)
    x2 = Dense(16, activation='relu')(x1)
    x3 = Dropout(0.05)(x2)
    outputs = Dense(1)(x3)
    return Model(inputs=inputs, outputs=outputs)

units = []
unit_cnt = unit_size
Y = np.array([cprice for cprice in train_eprice['clearing price (CNY/MWh)']])
X = []
XP = []  # 用于计算答案的预测数据


# Create bidding models for each power plant
for i in range(unit_cnt):
    now_model = create_model()
    units.append(now_model)

# Generate training data
for demand, demand_h in zip(train_eprice['demand'], train_eprice['demand_hat']):
    X.append([[unit.at[i, 'cost'], unit.at[i, 'capacity'], demand_h, unit.at[i, 'capacity'], demand] for i in range(unit_cnt)])

# Generate test data
for demand, demand_h in zip(ans_eprice['demand'], ans_eprice['demand_hat']):
    XP.append([[unit.at[i, 'cost'], unit.at[i, 'capacity'], demand_h, unit.at[i, 'capacity'], demand] for i in range(unit_cnt)])

X = np.array(X)
XP = np.array(XP)
print(X.shape)

# Use LSTM to simulate power plant bidding models
input_unit = Input(shape=(unit_size, 3))
first_half = LSTM(30, activation='relu')(input_unit)
concat = Concatenate()(first_half)
model = Model(inputs=input_unit, outputs=concat)

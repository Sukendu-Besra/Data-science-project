import pandas as pd

df = pd.read_csv('/content/traffic.csv')
df

df['DateTime'] = pd.to_datetime(df['DateTime'])

df = df.sort_values(by='DateTime').reset_index(drop=True)

print(df.head())

pivot_df = df.pivot(index='DateTime', columns='Junction', values='Vehicles')

pivot_df = pivot_df.fillna(0)

def create_lag_features(data):
    df_lag = data.copy()
    for i in range(1, 2):
        df_shifted = data.shift(i)
        df_shifted.columns = [f"{col}(H-{i})" for col in data.columns]
        df_lag = pd.concat([df_lag, df_shifted], axis=1)
    return df_lag.dropna()

Xy = create_lag_features(pivot_df)
Xy

from sklearn.preprocessing import MinMaxScaler

X = Xy.iloc[:, len(pivot_df.columns):]
y = Xy.iloc[:, :len(pivot_df.columns)]

X_train = X[X.index < '2017-04-01']
X_test = X[X.index >= '2017-04-01']
y_train = y[y.index < '2017-04-01']
y_test = y[y.index >= '2017-04-01']

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_scaled
X_test_scaled

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

model = LinearRegression()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
print("MSE:", mse)

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(y_test.index, y_test.iloc[:, 0], label='Actual')
plt.plot(y_test.index, y_pred[:, 0], label='Predicted', linestyle='--')
plt.title('Traffic Prediction at Junction 1')
plt.xlabel('Time')
plt.ylabel('Vehicles')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(y_test.index, y_test.iloc[:, 1], label='Actual', color='blue')
plt.plot(y_test.index, y_pred[:, 1], label='Predicted', linestyle='--', color='orange')
plt.title('Traffic Prediction at Junction 2')
plt.xlabel('Time')
plt.ylabel('Vehicles')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(y_test.index, y_test.iloc[:, 2], label='Actual', color='green')
plt.plot(y_test.index, y_pred[:, 2], label='Predicted', linestyle='--', color='red')
plt.title('Traffic Prediction at Junction 3')
plt.xlabel('Time')
plt.ylabel('Vehicles')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(y_test.index, y_test.iloc[:, 3], label='Actual', color='purple')
plt.plot(y_test.index, y_pred[:, 3], label='Predicted', linestyle='--', color='brown')
plt.title('Traffic Prediction at Junction 4')
plt.xlabel('Time')
plt.ylabel('Vehicles')
plt.legend()
plt.grid(True)
plt.show()

fig, axs = plt.subplots(4, 1, figsize=(15, 12), sharex=True)

junction_names = ['Junction 1', 'Junction 2', 'Junction 3', 'Junction 4']
colors = ['blue', 'orange', 'green', 'red']

for i in range(4):
    axs[i].plot(y_test.index, y_test.iloc[:, i], label='Actual', color=colors[i])
    axs[i].plot(y_test.index, y_pred[:, i], label='Predicted', linestyle='--', color='gray')
    axs[i].set_title(f'Traffic Prediction at {junction_names[i]}')
    axs[i].set_ylabel('Vehicles')
    axs[i].legend()
    axs[i].grid(True)

plt.xlabel('Time')
plt.tight_layout()
plt.show()

import numpy as np

sample_hours = 10
x = np.arange(sample_hours)
width = 0.35

fig, axs = plt.subplots(2, 2, figsize=(14, 8))
axs = axs.flatten()

for i in range(4):
    axs[i].bar(x - width/2, y_test.iloc[:sample_hours, i], width, label='Actual')
    axs[i].bar(x + width/2, y_pred[:sample_hours, i], width, label='Predicted')
    axs[i].set_title(f'Junction {i+1}')
    axs[i].set_ylabel('Vehicles')
    axs[i].set_xticks(x)
    axs[i].set_xticklabels(y_test.index[:sample_hours].strftime('%H:%M'), rotation=45)
    axs[i].legend()
    axs[i].grid(True)

plt.suptitle("Bar Comparison of Actual vs Predicted Traffic")
plt.tight_layout()
plt.show()

fig, axs = plt.subplots(2, 2, figsize=(12, 10))
axs = axs.flatten()

for i in range(4):
    axs[i].scatter(y_test.iloc[:, i], y_pred[:, i], alpha=0.5, c='teal')
    axs[i].plot([y_test.iloc[:, i].min(), y_test.iloc[:, i].max()],
                [y_test.iloc[:, i].min(), y_test.iloc[:, i].max()], 'r--')
    axs[i].set_title(f'Scatter: Actual vs Predicted (Junction {i+1})')
    axs[i].set_xlabel('Actual Vehicles')
    axs[i].set_ylabel('Predicted Vehicles')
    axs[i].grid(True)

plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
from ipywidgets import interact
import seaborn as sns

def plot_scatter(junction):
    j_idx = junction - 1
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test.iloc[:, j_idx], y=y_pred[:, j_idx], alpha=0.6)
    plt.plot([y_test.iloc[:, j_idx].min(), y_test.iloc[:, j_idx].max()],
             [y_test.iloc[:, j_idx].min(), y_test.iloc[:, j_idx].max()], 'r--')
    plt.xlabel('Actual Vehicles')
    plt.ylabel('Predicted Vehicles')
    plt.title(f'Junction {junction}: Actual vs Predicted')
    plt.grid(True)
    plt.show()

interact(plot_scatter, junction=[1, 2, 3, 4])

import pandas as pd
import seaborn as sns

df_compare = pd.DataFrame({
    'J1_Actual': y_test.iloc[:, 0],
    'J1_Pred': y_pred[:, 0],
    'J2_Actual': y_test.iloc[:, 1],
    'J2_Pred': y_pred[:, 1],
    'J3_Actual': y_test.iloc[:, 2],
    'J3_Pred': y_pred[:, 2],
    'J4_Actual': y_test.iloc[:, 3],
    'J4_Pred': y_pred[:, 3],
})

sns.pairplot(df_compare, plot_kws={'alpha': 0.6})

start_idx = 0
end_idx = 100

plt.figure(figsize=(14, 6))
plt.plot(y_test.index[start_idx:end_idx], y_test.iloc[start_idx:end_idx, 2], label='Actual', color='green')
plt.plot(y_test.index[start_idx:end_idx], y_pred[start_idx:end_idx, 2], label='Predicted', linestyle='--', color='red')
plt.title('Zoomed: Junction 3 Traffic')
plt.xlabel('Time')
plt.ylabel('Vehicles')
plt.legend()
plt.grid(True)
plt.show()

start_idx = 0
end_idx = 100

junction_names = ['Junction 1', 'Junction 2', 'Junction 3', 'Junction 4']
colors = ['blue', 'orange', 'green', 'red']

plt.figure(figsize=(14, 8))

for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.plot(y_test.index[start_idx:end_idx], y_test.iloc[start_idx:end_idx, i],
             label='Actual', color=colors[i])
    plt.plot(y_test.index[start_idx:end_idx], y_pred[start_idx:end_idx, i],
             label='Predicted', linestyle='--', color='black')
    plt.title(f'Zoomed: {junction_names[i]} Traffic')
    plt.xlabel('Time')
    plt.ylabel('Vehicles')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()

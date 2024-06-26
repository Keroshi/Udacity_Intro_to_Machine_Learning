from sklearn.preprocessing import MinMaxScaler
import numpy as np

data = np.array([[115], [140], [175]])
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)
print(scaled_data)

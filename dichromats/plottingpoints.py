import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

filename = '/Users/cameronzhang/Downloads/1994_participant_data - cr_testing.csv'
df = pd.read_csv(filename)

x_point = np.array([])
y_point = np.array([])

for idx in df.index:
    X = df.at[idx, 'x']
    Y = df.at[idx, 'y']
    x_point = np.append(x_point, X)
    y_point = np.append(y_point, Y)
    
plt.xlim(-0.002, 0.45)
plt.ylim(0, 0.65)
plt.scatter(x_point, y_point, s=5)
plt.show()
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Read the data
file_name = '../data/Two_class/data.csv'
data = pd.read_csv(file_name)

model_location = '../model/Two_class/model.csv'
model = pd.read_csv(model_location)
x_ax = np.linspace(data.iloc[:, 1].min(), data.iloc[:, 1].max(), 100)
y_ax = (-x_ax * model.iloc[0, 0] + model.iloc[0,1])/model.iloc[1,0]
# Plot the data
plt.scatter(data.iloc[:,1 ], data.iloc[:, 2], c=data.iloc[:, 3], cmap='coolwarm')
plt.plot(x_ax, y_ax, color='black')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Data')
plt.savefig('../Plots/Two_class.png')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

path = r'save'
data1 = pd.read_csv('save/nn_LEGO_cnn_10.csv')

x = np.linspace(0, 9, num=10)
y1_l = data1.loc[:, 'loss']
y1_a = data1.loc[:, 'accuracy']


plt.plot(x, y1_l, color='b', label='d1', marker='o', linewidth=1)
plt.title('loss graph')
plt.xlabel('FL training round')
plt.ylabel('Training loss')
plt.legend()
plt.show()

plt.plot(x, y1_a, color='b', label='d1', marker='o', linewidth=1)
plt.title('acc graph')
plt.xlabel('FL training round')
plt.ylabel('Testing accuracy')
plt.legend()
plt.show()






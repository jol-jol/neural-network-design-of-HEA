from ase.io import read
from ase.geometry import get_distances
from ase.build import sort

from random import shuffle

import numpy as np

# If variable BE is imported, then directly go to training
if 'BEs' not in vars():
  signatures = []; BEs = []; types = []; surfaces = [];
  with open('data.csv',) as f:
    for l in f.readlines():
      tmp = l[:-1].split(',')
      surfaces.append(tmp[0])
      types.append(tmp[1])
      tmp = list(map(float, tmp[2:]))
      signatures.append(tmp[:-1])
      BEs.append(tmp[-1])
  signatures = np.array(signatures).reshape((len(BEs), -1, 5))
  BEs = np.array(BEs)
  types = np.array(types)
  surfaces = np.array(surfaces)

### Training ###
TRAINING_RATIO = 0.5
training_num = int(TRAINING_RATIO * len(BEs))
import tensorflow as tf
from random import shuffle
rand_idx = list(range(len(signatures))); shuffle(rand_idx)
signatures = signatures[rand_idx]; BEs = BEs[rand_idx]
x_train = signatures[:training_num]; y_train = BEs[:training_num]
x_test = signatures[training_num:]; y_test = BEs[training_num:]

class MyModel(tf.keras.Model):
  def __init__(self, ):
    super(MyModel, self).__init__()
    self.w1 = tf.keras.layers.Dense(6, activation='tanh', use_bias=False)   
    self.w3 = tf.keras.layers.Dense(1, activation='linear', use_bias=False)
  def call(self, inputs):
    x = self.w1(inputs)
    x = self.w3(x)
    return tf.math.reduce_sum(x, axis=1,)

model = MyModel()

model.compile(optimizer='adam', loss='mse', learning_rate=0.0001, \
  metrics=['mae','mse'])

h = model.fit(x_train, y_train, epochs=3000, callbacks=[], )

if 'results' not in vars():
  results = []
y = model.evaluate(x_test, y_test, verbose=2)
results.append(y[2]**0.5)
print(results)

### Ploting results ###
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})
plt.scatter(y_train, model(x_train.astype('float32')).numpy().reshape(-1),
  color='blue', marker='.', alpha=1, edgecolors='None')
plt.scatter(y_test, model(x_test.astype('float32')).numpy().reshape(-1), 
  color='red',  marker='.', alpha=1, edgecolors='None')
plt.legend(['training (%i points)\nMAE=%.2f RMSE=%.2f'%
               (len(x_train), h.history['mae'][-1], h.history['mse'][-1]**0.5), \
            'testing (%i points)\nMAE=%.2f RMSE=%.2f'%
               (len(x_test), y[1], y[2]**0.5)], fontsize=10, loc='upper left')\
            .get_frame().set_edgecolor('k')
plt.plot([-2.5,0.5], [-2.5,0.5], 'k')
plt.plot([-2.5,0.5], [-2.35,0.65], 'k--')
plt.plot([-2.5,0.5], [-2.65,0.35], 'k--')
plt.xlabel(r'DFT-calculated $\Delta E_{\mathrm{OH}}-\Delta E_{\mathrm{OH, Pt(111)}}$ (eV)')
plt.ylabel('Neural network-predicted\n'+
           r'$\Delta E_{\mathrm{OH}}-\Delta E_{\mathrm{OH, Pt(111)}}$ (eV)')
plt.xlim([-2.5, 0.5]); plt.ylim([-2.5,0.5])
plt.box(on=True)
plt.tick_params(direction='in', right=True, top=True)
plt.show()


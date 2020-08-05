# if model training has not been performed, perform it first by executing
# "main.py" file
if 'model' not in vars():
  exec(open('main.py').read())

# elements_dict maps atomic number of an element (Ru, Rh, Ir, Pt or Pd)
# to the first 3 features used in the neural network model (period number,
# group number, and electronegativity). Note that these features are 
# referenced to Ru (a simplified normalization)
elements_dict = {44: [0,0,0], 45: [0,1,0], 46: [0,2,3],\
                 77: [1,1,1.5], 78: [1,2,4.5],}

coord_nums_dict = {}; gen_coord_nums_dict = {};
# 'coord_nums.csv' file contains the coordination environment information
# of different active sites on different crystal surfaces, as discussed
# in the paper
with open('coord_nums.csv') as f:
  for l in f.readlines():
    items = l.split(',')
    label = items[0]
    coord_nums_dict[label] = list(map(int, items[1:]))
    gen_coord_nums_dict[label] = sum(coord_nums_dict[label][2:])/12.
    # normalized by 12, which is the maximal coord. num. in fcc structure
gen_coord_nums = np.array(list(gen_coord_nums_dict.values()))
reorder_idx = gen_coord_nums.argsort()

from random import shuffle
from numpy import array, arange
elements = list(elements_dict.values()) * 5
results = {}
results_averages = {}
for envir in coord_nums_dict:
  coord_nums = coord_nums_dict[envir]
  new_results = []
  for i in range(10000):
    shuffle(elements)
    new_result = []
    new_result += [elements[0] + [coord_nums[0], 1, ]]
    new_result += [elements[1] + [coord_nums[1], 1, ]]
    for j in range(2, len(coord_nums)):
      new_result += [elements[j] + [coord_nums[j], 0, ]]
    for j in range(len(coord_nums), len(elements)):
      new_result += [[0,0,0,0,0,]] # [elements[j] + [11, 0, ]]
    new_results += [new_result]

  new_results = model(array(new_results).astype('float32'))
  new_results = new_results.numpy().reshape(-1).tolist()
  discretized_results = {}
  for i in range(int(min(new_results)*100-1), int(max(new_results)*100+1)):
    discretized_results[i] = 0
  for i in new_results:
    discretized_results[int(i*100-1)] += 1

  results[envir] = []
  for i in discretized_results:
    results[envir].append([i/100, discretized_results[i]])
  results[envir] = array(results[envir])
  results_averages[envir] = np.array(new_results).mean()

  print('%s %f' % (envir, np.array(new_results).std()))

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.figure(figsize = (1, len(coord_nums_dict), ))
gs = gridspec.GridSpec(1, len(coord_nums_dict)+4,  \
                       width_ratios=[0.75,]*len(results) + [0.2, 6, 0.2, 2])
gs.update(wspace=0.0, hspace=0.1)
for i in range(len(coord_nums_dict)):
  ax = plt.subplot(gs[i], xlim=(-0.1,1.2), ylim=(-2.0, 0.5),)
  envir = list(coord_nums_dict.keys())[reorder_idx[i]]
  plt.fill(results[envir][:,1]/625, results[envir][:,0], color=[0.5,0.5,1])
  plt.scatter([0., ], [results_averages[envir],], color='black', marker='x', \
              zorder=3)
  plt.text(0.6, 0.4, envir[:-6]+'\n'+envir[-6:], \
           horizontalalignment='center', verticalalignment='top', fontsize=10)
  if i == 0:
    ax.set_ylabel('Neural network-predicted \n' +
      r'$\Delta E_{\mathrm{OH}}-\Delta E_{\mathrm{OH, Pt(111)}}$ (eV)')
  ax.set_xticklabels([])
  ax.tick_params(bottom=False)
  if i > 0:
    ax.set_yticklabels([])
    ax.spines['left'].set_color('white')
    ax.tick_params(left=False)
  if i < len(results) - 1:
    ax.spines['right'].set_color('white')
  if i == int(len(results)/2):
    ax.set_xlabel('\nRelative frequency')
  ax.tick_params(direction='in', )
ax.tick_params(right=True)
ax.legend(['Frequency distribution', 'Mean of distribution'], \
          fancybox=False, edgecolor='black', loc='lower right', fontsize=12)

ax = plt.subplot(gs[-3], ylim=(-2.0, 0.5),)
ax.scatter(list(gen_coord_nums_dict.values()), \
           list(results_averages.values()), \
           zorder=3, color='black', marker='x',)
ax.set_xlabel('Total CN of nearest neighbours')
ax.tick_params(direction='in', right=True, top=True)
ax.set_yticklabels([])

from numpy import polyfit
results_averages = np.array(list(results_averages.values()))
a, b = polyfit(list(gen_coord_nums_dict.values()), \
               results_averages, deg=1)
plt.plot([gen_coord_nums[reorder_idx[0]], gen_coord_nums[reorder_idx[-1]]], \
  [a*gen_coord_nums[reorder_idx[0]]+b, a*gen_coord_nums[reorder_idx[-1]]+b],
  color='blue', zorder=1)
R_2 = 1 - sum((gen_coord_nums*a+b-results_averages)**2) / \
          sum((results_averages-results_averages.mean())**2)
MAE = abs(gen_coord_nums*a+b-results_averages).mean()
RMSE = (((gen_coord_nums*a+b-results_averages)**2).mean())**0.5
ax.text(9.5, -0.5, '$R^2$: %.2f\nMAE: %.2f eV\nRMSE: %.2f eV\n'%(R_2,MAE,RMSE),
        horizontalalignment='right', verticalalignment='bottom', fontsize=10)
ax.legend(['Linear fit', 'Mean of distribution'], \
          fancybox=False, edgecolor='black', loc='lower right', fontsize=12)

ax = plt.subplot(gs[-1], xlim=(-2., 1), ylim=(-2.0, 0.5),)
ax.plot([-2+0.8, 0.1+0.8, 1.72-0.5-0.8], [-2, 0.1, 0.5], 
         zorder=3, color='blue', )
ax.scatter(results_averages+0.8, results_averages, \
           zorder=3, color='black', marker='x',)
ax.set_xlabel('Limiting\npotential (V)')
ax.tick_params(direction='in', right=True, top=True)
ax.set_yticklabels([])

  
plt.show()
#print(model(inp))

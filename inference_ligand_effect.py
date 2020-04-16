if 'model' not in vars():
  exec(open('main.py').read())

elements_dict = {44: [1,0,0,0,0], 45: [0,1,0,0,0], 46: [0,0,1,0,0],\
                 77: [0,0,0,1,0], 78: [0,0,0,0,1],}
elements_dict = {44: [0,0,0], 45: [0,1,0], 46: [0,2,3],\
                 77: [1,1,1.5], 78: [1,2,4.5],}

envir = '8-8 (100)'
coord_nums = [8,8,8,8,8,8,12,12,12,12,12,12]

from numpy import array, arange
colors = [[1,0,0], [0,1,0], [0,0,1], [1,1,0], [0,1,1]]
colors = array(colors)

from random import shuffle
elements = list(elements_dict.values()) * 5
inputs = []; results = [];
for i in range(10000):
  shuffle(elements)
  new_input = []
  new_input += [elements[0] + [coord_nums[0], 1, ]]
  new_input += [elements[1] + [coord_nums[1], 1, ]]
  for j in range(2, len(coord_nums)):
    new_input += [elements[j] + [coord_nums[j], 0, ]]
  for j in range(len(coord_nums), len(elements)):
    new_input += [[0,0,0,0,0,]] # [elements[j] + [11, 0, ]]
  inputs += [new_input]
inputs = array(inputs)

results = model(array(inputs).astype('float32'))
results = results.numpy().reshape(-1)

def plot_density(ax, data, color):
  data = data.tolist()
  discretized_data = {}
  for i in range(int(min(data)*250-2), int(max(data)*250+2)):
    discretized_data[i] = 0
  for i in data:
    discretized_data[int(i*250-1)] += 1
  x = array(list(discretized_data.keys()))/250
  y = array(list(discretized_data.values()))/230
  ax.fill(x, y, color=color, alpha=0.5)

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
plt.figure(figsize = (4, 1, ))
gs = gridspec.GridSpec(4, 1, height_ratios=[3, 1, 1, 1])
gs.update(wspace=0.0, hspace=0.0)

ax = plt.subplot(gs[0], xlim=(-1.8, -0.5), ylim=(0,1.19),)
ax.set_xticklabels([])
ax.tick_params(direction='in', right=True, top=True)
plot_density(ax, results, [0,0,0])

ax = plt.subplot(gs[1], xlim=(-1.8, -0.5), ylim=(0,0.39),)
ax.set_xticklabels([])
ax.tick_params(direction='in', right=True, top=True)
ax.set_ylabel('Relative frequency')
plot_density(ax, results[np.all(np.all(inputs[:,0:2,0:2] == [[0, 0], [0, 0]], 
             axis=1), axis=1)], colors[0])
plot_density(ax, results[np.all(np.all(inputs[:,0:2,0:2] == [[0, 1], [0, 1]], 
             axis=1), axis=1)], colors[1])
plot_density(ax, results[np.all(np.all(inputs[:,0:2,0:2] == [[0, 2], [0, 2]], 
             axis=1), axis=1)], colors[2])
plot_density(ax, results[np.all(np.all(inputs[:,0:2,0:2] == [[0, 0], [0, 1]], 
             axis=1), axis=1)], (colors[0]+colors[1])/2)
plot_density(ax, results[np.all(np.all(inputs[:,0:2,0:2] == [[0, 0], [0, 2]], 
             axis=1), axis=1)], (colors[0]+colors[2])/2)
plot_density(ax, results[np.all(np.all(inputs[:,0:2,0:2] == [[0, 1], [0, 2]], 
             axis=1), axis=1)], (colors[1]+colors[2])/2)
ax.plot([results[np.all(np.all(inputs[:,0:2,0:2] == [[0, 0], [0, 0]], 
         axis=1), axis=1)].mean()]*2, [0, 1], '--', color=colors[0], alpha=0.5)
ax.plot([results[np.all(np.all(inputs[:,0:2,0:2] == [[0, 1], [0, 1]], 
         axis=1), axis=1)].mean()]*2, [0, 1], '--', color=colors[1], alpha=0.5)
ax.plot([results[np.all(np.all(inputs[:,0:2,0:2] == [[0, 2], [0, 2]], 
         axis=1), axis=1)].mean()]*2, [0, 1], '--', color=colors[2], alpha=0.5)
ax.plot([results[np.all(np.all(inputs[:,0:2,0:2] == [[0, 0], [0, 1]], 
         axis=1), axis=1)].mean()]*2, [0, 1], '--', color=(colors[0]+colors[1])/2, alpha=0.5)
ax.plot([results[np.all(np.all(inputs[:,0:2,0:2] == [[0, 0], [0, 2]], 
         axis=1), axis=1)].mean()]*2, [0, 1], '--', color=(colors[0]+colors[2])/2, alpha=0.5)
ax.plot([results[np.all(np.all(inputs[:,0:2,0:2] == [[0, 1], [0, 2]], 
         axis=1), axis=1)].mean()]*2, [0, 1], '--', color=(colors[1]+colors[2])/2, alpha=0.5)
ax.text(results[np.all(np.all(inputs[:,0:2,0:2] == [[0, 0], [0, 0]], 
        axis=1), axis=1)].mean(), 0.37, ' Ru \n Ru ', color=colors[0],
        horizontalalignment='right', verticalalignment='top')
ax.text(results[np.all(np.all(inputs[:,0:2,0:2] == [[0, 1], [0, 1]], 
        axis=1), axis=1)].mean(), 0.37, ' Rh \n Rh ', color=colors[1],
        horizontalalignment='right', verticalalignment='top')
ax.text(results[np.all(np.all(inputs[:,0:2,0:2] == [[0, 2], [0, 2]], 
        axis=1), axis=1)].mean(), 0.37, ' Pd \n Pd ', color=colors[2],
        horizontalalignment='left', verticalalignment='top')
ax.text(results[np.all(np.all(inputs[:,0:2,0:2] == [[0, 0], [0, 1]], 
        axis=1), axis=1)].mean(), 0.37, ' Ru \n Rh ', color=(colors[0]+colors[1])/2,
        horizontalalignment='right', verticalalignment='top')
ax.text(results[np.all(np.all(inputs[:,0:2,0:2] == [[0, 0], [0, 2]], 
        axis=1), axis=1)].mean(), 0.37, ' Ru \n Pd ', color=(colors[0]+colors[2])/2,
        horizontalalignment='left', verticalalignment='top')
ax.text(results[np.all(np.all(inputs[:,0:2,0:2] == [[0, 1], [0, 2]], 
        axis=1), axis=1)].mean(), 0.37, ' Rh \n Pd ', color=(colors[1]+colors[2])/2,
        horizontalalignment='left', verticalalignment='top')

ax = plt.subplot(gs[2], xlim=(-1.8, -0.5), ylim=(0,0.39),)
ax.set_xticklabels([])
ax.tick_params(direction='in', right=True, top=True)
plot_density(ax, results[np.all(np.all(inputs[:,0:2,0:2] == [[1, 1], [1, 1]], 
             axis=1), axis=1)], colors[3])
plot_density(ax, results[np.all(np.all(inputs[:,0:2,0:2] == [[1, 2], [1, 2]], 
             axis=1), axis=1)], colors[4])
plot_density(ax, results[np.all(np.all(inputs[:,0:2,0:2] == [[1, 1], [1, 2]], 
             axis=1), axis=1)], (colors[3]+colors[4])/2)
plot_density(ax, results[np.all(np.all(inputs[:,0:2,0:2] == [[0, 0], [1, 1]], 
             axis=1), axis=1)], (colors[0]+colors[3])/2)
plot_density(ax, results[np.all(np.all(inputs[:,0:2,0:2] == [[0, 0], [1, 2]], 
             axis=1), axis=1)], (colors[0]+colors[4])/2)
ax.plot([results[np.all(np.all(inputs[:,0:2,0:2] == [[1, 1], [1, 1]], 
         axis=1), axis=1)].mean()]*2, [0, 1], '--', color=colors[3], alpha=0.5)
ax.plot([results[np.all(np.all(inputs[:,0:2,0:2] == [[1, 2], [1, 2]], 
         axis=1), axis=1)].mean()]*2, [0, 1], '--', color=colors[4], alpha=0.5)
ax.plot([results[np.all(np.all(inputs[:,0:2,0:2] == [[1, 1], [1, 2]], 
         axis=1), axis=1)].mean()]*2, [0, 1], '--', color=(colors[3]+colors[4])/2, alpha=0.5)
ax.plot([results[np.all(np.all(inputs[:,0:2,0:2] == [[0, 0], [1, 1]], 
         axis=1), axis=1)].mean()]*2, [0, 1], '--', color=(colors[0]+colors[3])/2, alpha=0.5)
ax.plot([results[np.all(np.all(inputs[:,0:2,0:2] == [[0, 0], [1, 2]], 
         axis=1), axis=1)].mean()]*2, [0, 1], '--', color=(colors[0]+colors[4])/2, alpha=0.5)
ax.text(results[np.all(np.all(inputs[:,0:2,0:2] == [[1, 1], [1, 1]], 
        axis=1), axis=1)].mean(), 0.37, ' Ir \n Ir ', color=colors[3],
        horizontalalignment='right', verticalalignment='top')
ax.text(results[np.all(np.all(inputs[:,0:2,0:2] == [[1, 2], [1, 2]], 
        axis=1), axis=1)].mean(), 0.37, ' Pt \n Pt ', color=colors[4],
        horizontalalignment='left', verticalalignment='top')
ax.text(results[np.all(np.all(inputs[:,0:2,0:2] == [[1, 1], [1, 2]], 
        axis=1), axis=1)].mean(), 0.37, ' Ir \n Pt ', color=(colors[3]+colors[4])/2,
        horizontalalignment='left', verticalalignment='top')
ax.text(results[np.all(np.all(inputs[:,0:2,0:2] == [[0, 0], [1, 1]], 
        axis=1), axis=1)].mean(), 0.37, ' Ru \n Ir ', color=(colors[0]+colors[3])/2,
        horizontalalignment='right', verticalalignment='top')
ax.text(results[np.all(np.all(inputs[:,0:2,0:2] == [[0, 0], [1, 2]], 
        axis=1), axis=1)].mean(), 0.37, ' Ru \n Pt ', color=(colors[0]+colors[4])/2,
        horizontalalignment='left', verticalalignment='top')

ax = plt.subplot(gs[3], xlim=(-1.8, -0.5), ylim=(0,0.39),)
ax.tick_params(direction='in', right=True, top=True)
ax.set_xlabel('Neural network-predicted ' + 
              r'$\Delta E_{\mathrm{OH}}-\Delta E_{\mathrm{OH, Pt(111)}}$ (eV)')
plot_density(ax, results[np.all(np.all(inputs[:,0:2,0:2] == [[0, 1], [1, 1]], 
             axis=1), axis=1)], (colors[1]+colors[3])/2)
plot_density(ax, results[np.all(np.all(inputs[:,0:2,0:2] == [[0, 2], [1, 2]], 
             axis=1), axis=1)], (colors[2]+colors[4])/2)
plot_density(ax, results[np.all(np.all(inputs[:,0:2,0:2] == [[0, 2], [1, 1]], 
             axis=1), axis=1)], (colors[2]+colors[3])/2)
plot_density(ax, results[np.all(np.all(inputs[:,0:2,0:2] == [[0, 1], [1, 2]], 
             axis=1), axis=1)], (colors[1]+colors[4])/2)
ax.plot([results[np.all(np.all(inputs[:,0:2,0:2] == [[0, 1], [1, 1]], 
         axis=1), axis=1)].mean()]*2, [0, 1], '--', color=(colors[1]+colors[3])/2, alpha=0.5)
ax.plot([results[np.all(np.all(inputs[:,0:2,0:2] == [[0, 2], [1, 2]], 
         axis=1), axis=1)].mean()]*2, [0, 1], '--', color=(colors[2]+colors[4])/2, alpha=0.5)
ax.plot([results[np.all(np.all(inputs[:,0:2,0:2] == [[0, 2], [1, 1]], 
         axis=1), axis=1)].mean()]*2, [0, 1], '--', color=(colors[2]+colors[3])/2, alpha=0.5)
ax.plot([results[np.all(np.all(inputs[:,0:2,0:2] == [[0, 1], [1, 2]], 
         axis=1), axis=1)].mean()]*2, [0, 1], '--', color=(colors[1]+colors[4])/2, alpha=0.5)
ax.text(results[np.all(np.all(inputs[:,0:2,0:2] == [[0, 1], [1, 1]], 
        axis=1), axis=1)].mean(), 0.37, ' Rh \n Ir ', color=(colors[1]+colors[3])/2,
        horizontalalignment='right', verticalalignment='top')
ax.text(results[np.all(np.all(inputs[:,0:2,0:2] == [[0, 2], [1, 2]], 
        axis=1), axis=1)].mean(), 0.37, ' Pd \n Pt ', color=(colors[2]+colors[4])/2,
        horizontalalignment='left', verticalalignment='top')
ax.text(results[np.all(np.all(inputs[:,0:2,0:2] == [[0, 2], [1, 1]], 
        axis=1), axis=1)].mean(), 0.37, ' Pd \n Ir ', color=(colors[2]+colors[3])/2,
        horizontalalignment='right', verticalalignment='top')
ax.text(results[np.all(np.all(inputs[:,0:2,0:2] == [[0, 1], [1, 2]], 
        axis=1), axis=1)].mean(), 0.37, ' Rh \n Pt ', color=(colors[1]+colors[4])/2,
        horizontalalignment='left', verticalalignment='top')

print('%s %f' % (envir, np.array(results).std()))

for i in elements_dict:
  elements = elements_dict[i]
  new_input = [elements + [coord_nums[0], 1, ]]
  new_input += [elements + [coord_nums[1], 1, ]]
  for j in range(2, len(coord_nums)):
    new_input += [elements + [coord_nums[j], 0, ]]
  inputs = array([new_input])

  results = model(array(inputs).astype('float32'))
  results = results.numpy().reshape(-1)
  print(results)

plt.show()


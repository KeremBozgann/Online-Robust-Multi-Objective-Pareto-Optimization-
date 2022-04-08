import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib as mpl
import scipy.stats as stats
import matplotlib.patches as mpatches


#Pareto front
D= 0.5
U= 0.2
obj1= np.array([9, 6.67, 4.33, 2, 8, 5.03, 3, 3, 6.33, 1.33])
obj2= np.array([2, 4.33, 6.67, 9, 1, 2.96, 5.43, 3, 1.42, 7.41])
anots= [f"Kol {i+1}" for i in range(10)]
pareto_obj1= np.array([9, 6.67, 4.33, 2])
pareto_obj2= np.array([2, 4.33, 6.67, 9])

fig, ax= plt.subplots()

# removing the default axis on all sides:
for side in ['bottom','right','top','left']:
    ax.spines[side].set_visible(False)

# removing the axis ticks
plt.xticks([]) # labels
plt.yticks([])
ax.xaxis.set_ticks_position('none') # tick markers
ax.yaxis.set_ticks_position('none')


ax.scatter(obj1, obj2, color= 'tab:red', marker= '^',
           label='Pareto optimal olmayan kollar')
ax.scatter(pareto_obj1, pareto_obj2, color= 'tab:green', marker= '^',
           label='Pareto optimal kollar')

for i in range(len(obj1)):
    rect = mpatches.Rectangle((obj1[i]- D,
                               obj2[i] - D),
                              2 * D,
                              2 * D,
                              fill=False,
                              color="tab:purple",
                              linewidth=2)
    plt.gca().add_patch(rect)

for i in range(len(obj1)):
    rect = mpatches.Rectangle((obj1[i]- D - U,
                               obj2[i] - D- U),
                              2 * (D+U),
                              2 * (D+U),
                              fill=False,
                              color="tab:orange",
                              linewidth=2)
    plt.gca().add_patch(rect)
#
# ax.scatter([obj1[2]-D, obj1[3]-D], [obj2[2]-D, obj2[3]-D],
#            color= 'tab:green', marker= 'x')
# ax.scatter([obj1[9]+D], [obj2[9]+D],
#            color= 'tab:red', marker= 'x')

for i in range(10):
    ax.annotate(anots[i], xy=(obj1[i], obj2[i]),
                    xytext=(obj1[i]+0.15, obj2[i]+0.15))
xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()

dps = fig.dpi_scale_trans.inverted()
bbox = ax.get_window_extent().transformed(dps)
width, height = bbox.width, bbox.height

# manual arrowhead width and length
hw = 1. / 20. * (ymax - ymin)
hl = 1. / 20. * (xmax - xmin)
lw = 1.  # axis line width
ohg = 0.3  # arrow overhang

# compute matching arrowhead length and width
yhw = hw / (ymax - ymin) * (xmax - xmin) * height / width
yhl = hl / (xmax - xmin) * (ymax - ymin) * width / height

# draw x and y axis
ax.arrow(xmin, 0, xmax - xmin, 0., fc='k', ec='k', lw=lw,
         head_width=hw, head_length=hl, overhang=ohg,
         length_includes_head=True, clip_on=False)

ax.arrow(0, ymin, 0., ymax - ymin, fc='k', ec='k', lw=lw,
         head_width=yhw, head_length=yhl, overhang=ohg,
         length_includes_head=True, clip_on=False)


ax.set_xlabel('Hedef 1')
ax.set_ylabel('Hedef 2')
ax.legend()

plt.show()
plt.savefig('istatistiksel ve onlenemez.pdf')



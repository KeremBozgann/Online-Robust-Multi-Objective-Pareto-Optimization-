from simglucose_test import pareto, plot_pareto
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axisartist.axislines import SubplotZero

matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

plt.ioff()

func= lambda x: 1/np.cos(x)

a =-1
b = 0
c= 0

X = np.linspace(1, 10, 10)
Y = a * np.power(X, 2) + b * X + c

x_dom = [1.6, 2.9, 6.8, 7.10, 5.59, 2.78, 3.86, 3.94, 5.06]
y_dom = [-24, -25.7, -74.8, -89, -64.7, -47.3, -40.5, -65.3, -47.9]

point_size= 60

fig, ax= plt.subplots()
ax.scatter(X, Y, color= 'tab:red', s= point_size)

xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()

ax.scatter(X[-1], Y[-1], c= 'white', edgecolor= 'white', s= point_size, linewidth= 2)
ax.scatter(x_dom, y_dom, color= 'tab:orange', edgecolor= 'tab:orange', s= point_size)

# for direction in ["yzero", "xzero"]:
#     # adds arrows at the ends of each axis
#     ax.axis[direction].set_axisline_style("-|>")
#
#     # adds X and Y-axis from the origin
#     ax.axis[direction].set_visible(True)

for side in ['bottom','right','top','left']:
    ax.spines[side].set_visible(False)

# for direction in ["left", "right", "bottom", "top"]:
#     # hides borders
#     ax.axis[direction].set_visible(False)


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
ax.arrow(0, ymin+ 5, xmax - xmin, 0., fc='k', ec='k', lw=lw,
         head_width=hw, head_length=hl, overhang=ohg,
         length_includes_head=True, clip_on=False)

ax.arrow(0, ymin+ 5, 0., ymax - ymin, fc='k', ec='k', lw=lw,
         head_width=yhw, head_length=yhl, overhang=ohg,
         length_includes_head=True, clip_on=False)



ax.set_ylim(ymax=1)
ax.set_xlim(xmin=0)

ax.set_xticks([])
ax.set_yticks([])

ax.xaxis.set_label_coords(.45, .05)
ax.yaxis.set_label_coords(-.01, .55)

plt.xlabel(r'\textbf{Objective 1}', fontsize=12)
plt.ylabel(r'\textbf{Objective 2}', fontsize=12)

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['xtick.major.pad'] = '5'
plt.rcParams['ytick.major.pad'] = '5'

plt.savefig('test.pdf', bbox_inches='tight', pad_inches= 0.1)
plt.show()
# plt.savefig('test.pdf')
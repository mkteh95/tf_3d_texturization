import binvox_rw
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import HTML

def init():
    return (fig,)

def animate(pos):
    ax.view_init(azim=pos)
    return (fig,)

with open('3Dmodels/model.binvox', 'rb') as f:
    model = binvox_rw.read_as_3d_array(f)
    
voxels = model.data

filled = np.where(voxels)
palette = np.random.uniform(0, 1, voxels.shape + (3,))

colors = np.zeros(palette.shape)
colors[filled] = palette[filled]

edges = np.clip(2 * colors - 0.5, 0, 1)

fig = plt.figure()
# fig.set_size_inches(13.75, 13.75, 13.75)
ax = Axes3D(fig)
ax.voxels(voxels, facecolors=colors, edgecolors=edges, linewidth=0.5)
# ax1 = fig.add_subplot(221, projection='3d')
# ax1.voxels(voxels, facecolors=colors, edgecolors=edges, linewidth=0.5)

# ax2 = fig.add_subplot(222, projection='3d')
# ax2.voxels(voxels.transpose(0,2,1), facecolors=colors.transpose(0,2,1,3), edgecolors=edges.transpose(0,2,1,3), linewidth=0.5)

# ax3 = fig.add_subplot(223, projection='3d')
# ax3.voxels(voxels.transpose(1,2,0), facecolors=colors.transpose(1,2,0,3), edgecolors=edges.transpose(1,2,0,3), linewidth=0.5)

# ax4 = fig.add_subplot(224, projection='3d')
# ax4.voxels(voxels.transpose(2,1,0), facecolors=colors.transpose(2,1,0,3), edgecolors=edges.transpose(2,1,0,3), linewidth=0.5)

for i in range(0, 360):
    ax.view_init(azim=i)
    plt.show()
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from matplotlib import cm
from matplotlib import animation, rc
from IPython.display import HTML

import numpy as np

w_min = -7
w_max = 5

b_min = -5
b_max = 5

def plot_3d_view(sn,  X, Y, plot_3d = False, anim =False):
    if plot_3d==True and anim ==False: 
        
        W = np.linspace(w_min, w_max, 256)
        b = np.linspace(b_min, b_max, 256)
        WW, BB = np.meshgrid(W, b)
        Z = sn.error(X, Y, WW, BB)
        fig = plt.figure(dpi=100)
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(WW, BB, Z, rstride=3, cstride=3, alpha=0.5, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        cset = ax.contourf(WW, BB, Z, 25, zdir='z', offset=-1, alpha=0.6, cmap=cm.coolwarm)
        ax.set_xlabel('w')
        ax.set_xlim(w_min - 1, w_max + 1)
        ax.set_ylabel('b')
        ax.set_ylim(b_min - 1, b_max + 1)
        ax.set_zlabel('error')
        ax.set_zlim(-1, np.max(Z))
        ax.view_init (elev=25, azim=-75) # azim = -20
        ax.dist=12  
        title = ax.set_title('Epoch 0')
        return ax, title,fig

def plot_2d_view(sn, X,Y,plot_2d =False, anim = False):
    if plot_2d:
        W = np.linspace(w_min, w_max, 256)
        b = np.linspace(b_min, b_max, 256)
        WW, BB = np.meshgrid(W, b)
        Z = sn.error(X, Y, WW, BB)
        fig = plt.figure(dpi = 100)
        ax = plt.subplot(111)
        ax.set_xlabel('w')
        ax.set_xlim(w_min -1, w_max+1)
        ax.set_ylabel('b')
        ax.set_ylim(b_min -1, b_max+1)
        title = ax.set_title('Epoch 0')
        cset = plt.contourf(WW, BB, Z, 25, alpha=0.6, cmap=cm.bwr)
        plt.show()
        return ax, title, fig



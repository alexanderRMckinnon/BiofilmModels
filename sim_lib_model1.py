import numpy as np
from sim_lib import *

class CellDeathModelOne(SpatiotemporalModel):
    def __init__(self):
        SpatiotemporalModel.__init__(self, in_filename="CellDeathModelOne.gif", in_t_max=2, in_gif_t_max=1)
        
    def initialise(self):
        x_min, x_max, x_numpoints = -1, 1, 100
        x_length = x_max - x_min
        self.dx = x_length/x_numpoints
        self.u, self.w = np.random.random([x_numpoints, x_numpoints]), np.random.random([x_numpoints, x_numpoints])
        self.t = 0
        self.dt = (self.dx**2)/2
        self.steps = int(self.t_max/(self.frames*self.dt))
        self.min_u, self.max_u = 0, 1
        self.min_w, self.max_w = 0, 1
        
    def update(self):
        for _ in range(self.steps):
            self.t += self.dt
            self._update()
            
    def _update(self):
        self.u += self.u*(1-self.u)
        self.w += 0
    
    def initialise_figure(self):
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=np.array([16, 9]))
        return fig, ax
    
    def draw(self, ax):
        ax[0].clear()
        ax[1].clear()
        ax[0].get_yaxis().set_visible(False)
        ax[1].get_yaxis().set_visible(False)
        ax[0].get_xaxis().set_visible(False)
        ax[1].get_xaxis().set_visible(False)
        ax[0].imshow(self.u, cmap='Reds', interpolation='lanczos', vmin=self.min_u, vmax = self.max_u)
        ax[1].imshow(self.w, cmap='Blues', interpolation='lanczos', vmin=self.min_w, vmax = self.max_w)
        ax[0].grid(b=False)
        ax[1].grid(b=False)
        ax[0].set_title("u, t = {:.1f}".format(self.t))
        ax[1].set_title("w, t = {:.1f}".format(self.t))
        
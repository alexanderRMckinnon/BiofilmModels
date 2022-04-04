import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib import animation
from matplotlib.animation import FuncAnimation, PillowWriter 
from IPython.display import Image

class SpatiotemporalModel:
    def __init__(self):
        self.dpi = 100 # Dots Per square Inch, sharpness of gif
        self.fps = 10 # Frame rate of gif, effects gif length but not simulation length
        self.fig_dim = np.array([16, 9])/2 # Dimensions of gif, keep the ratio but change the scaling factor
        self.filename = 'simulation.gif' # Name of gif file
        
        self.t_max = 5
        self.gif_t_max = 2
#         self.frames = self.gif_t_max*self.fps
        self.frames = 10
        
        self.dx = 10/1000
        self.CLF = 1
        self.dt = self.CLF*(self.dx**2)/(2*abs(1))
        self.dt = self.t_max
        
#         self.dx = 0.01
#         self.dt = 0.9* self.dx**2/2
        

    def initialise(self):
        raise NotImplementedError()
    def initialise_figure(self):
        fig, ax = plt.subplots(figsize = (self.fig_dim[0], self.fig_dim[1]))
        return fig, ax
    def update(self):
        raise NotImplementedError()
    def draw(self, ax):
        raise NotImplementedError()
    def plot_simulation(self):
        fig, ax = self.initialise_figure()
        
        def step(t):
            self.update()
            self.draw(ax)
        
        anim = animation.FuncAnimation(fig, step, frames = np.arange(self.frames))        
        anim.save(filename = self.filename, fps = self.fps, dpi = self.dpi)
        plt.close()
        
        with open(self.filename,'rb') as file:
            display(Image(file.read()))
            
            
            
class OneDimDiffusion(SpatiotemporalModel):
    def __init__(self):
        SpatiotemporalModel.__init__(self)
        self.D = 1
        self.num_x_points = 1000
        self.t = 0
        self.X = np.linspace(-5, 5, self.num_x_points)
        self.Y = np.exp(-self.X**2)
        
    def update(self):
        for _ in range(int(0.1/(0.9*self.dt))):
#         for _ in range(100):
            self.t += self.dt
            self._update()
    def _update(self):
        La = laplacian1D(self.Y, self.dx)
        delta_Y = self.dt*self.D*La
        self.Y += delta_Y
    def draw(self, ax):
        ax.clear()
        ax.plot(self.X,self.Y, color="r")
        ax.set_ylim(0,1)
        ax.set_xlim(-5,5)
        ax.set_title("t = {:.2f}".format(self.t))
        
def laplacian1D(Y, dx):
    return (-2*Y + np.roll(Y,1,axis=0) + np.roll(Y,-1,axis=0)) / (dx ** 2)
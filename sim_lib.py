import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib import animation
from matplotlib.animation import FuncAnimation, PillowWriter 
from IPython.display import Image

class SpatiotemporalModel:
    def __init__(self, in_filename, in_t_max, in_gif_t_max):
        self.filename, self.t_max, self.gif_t_max = in_filename, in_t_max, in_gif_t_max
        
        self.dpi = 100 # Dots Per square Inch, sharpness of gif
        self.fps = 20 # Frame rate of gif, effects gif length but not simulation length
        self.fig_dim = np.array([16, 9])/2 # Dimensions of gif, keep the ratio but change the scaling factor
       
        self.frames = int(self.fps*self.gif_t_max)
        
    def initialise_figure(self):
        fig, ax = plt.subplots(figsize = (self.fig_dim[0], self.fig_dim[1]))
        return fig, ax
    
    def update(self):
        raise NotImplementedError()
    def draw(self, ax):
        raise NotImplementedError()
    def reset_sim(self):
        raise NotImplementedError()
        
    def gif_simulation(self, Y=None):
        fig, ax = self.initialise_figure()
        self.reset_sim(Y)
        def step(t):
            if t<5 or t>self.frames+4:
                self.draw(ax)
            else:
                self.update()
                self.draw(ax)
        anim = animation.FuncAnimation(fig, step, frames = np.arange(self.frames+10))        
        anim.save(filename = self.filename, fps = self.fps, dpi = self.dpi)
        plt.close()
        
        with open(self.filename,'rb') as file:
            display(Image(file.read()))
            
    def histogram_simulation(self, Y=None):
        fig, ax = self.initialise_figure()
        self.reset_sim(Y)
        self.histogram = np.zeros([self.num_x_points, self.frames+1])
        self.histogram[:,0] = self.histogram_get_slice()
        for t in range(1, self.frames+1):
            self.update()
            self.histogram[:,t] = self.histogram_get_slice()
        ax.imshow(self.histogram, cmap='Reds', interpolation='nearest', aspect='auto')
        self.histogram_draw(ax)
#         plt.show()
            
            
class OneDimDiffusion(SpatiotemporalModel):
    def __init__(self):
        SpatiotemporalModel.__init__(self, in_filename="1D_Diffusion_Simulation.gif", in_t_max=5, in_gif_t_max=2)
        self.D = 1
        self.num_x_points = 1000
        self.X = np.linspace(-5, 5, self.num_x_points)
        self.dx = 10/self.num_x_points
        self.t = 0
        self.dt = (self.dx**2)/2
        self.Y = np.exp(-self.X**2)
        self.ylim = [0,1]
        
    def update(self):
        for _ in range(int(self.t_max/(self.frames*self.dt))):
                self.t += self.dt
                self._update()
    def _update(self):
        La = laplacian1D(self.Y, self.dx)
        delta_Y = self.dt*self.D*La
        self.Y += delta_Y
    def draw(self, ax):
        ax.clear()
        ax.plot(self.X,self.Y, color="r")
        ax.set_ylim(self.ylim[0],self.ylim[1])
        ax.get_yaxis().set_visible(False)
        ax.set_xlim(-5,5)
        ax.get_xaxis().set_visible(False)
        ax.set_title("t = {:.1f}".format(self.t))
    def reset_sim(self, Y):
        self.t = 0
        if Y is None:
            self.Y = np.exp(-self.X**2)
        else:
            self.Y = Y
    def histogram_get_slice(self):
        return self.Y
    def histogram_draw(self, ax):
        ax.set_xlabel("time")
        x_axis = list(range(0,int(np.floor(self.t_max))+1))
        ax.set_xticks(np.linspace(0, self.frames, len(x_axis)))
        ax.set_xticklabels(x_axis)
        ax.get_yaxis().set_visible(False)
        ax.set_title("Histogram")
def laplacian1D(Y, dx):
    return (-2*Y + np.roll(Y,1,axis=0) + np.roll(Y,-1,axis=0)) / (dx ** 2)
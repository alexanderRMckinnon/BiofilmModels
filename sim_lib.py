import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib import animation
from matplotlib.animation import FuncAnimation, PillowWriter 
from IPython.display import Image, display, Latex
import matplotlib.pylab as pl

class SpatiotemporalModel:
    def __init__(self, in_filename, in_t_max, in_gif_t_max):
        self.filename, self.t_max, self.gif_t_max = in_filename, in_t_max, in_gif_t_max
        
        self.dpi = 100 # Dots Per square Inch, sharpness of gif
        self.fps = 20 # Frame rate of gif, effects gif length but not simulation length
        self.fig_dim = np.array([16, 9])/2 # Dimensions of gif, keep the ratio but change the scaling factor
       
        self.frames = int(self.fps*self.gif_t_max)
        self.line_frames = 10
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
            if t<5 or t>self.frames+5:
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

    def line_plot_simulation(self, Y=None):
        fig, ax = self.initialise_figure()
        self.reset_sim(Y)
        cmap = pl.cm.Reds(np.linspace(1,0.3, self.line_frames))
        self.draw_line_plot(ax, cmap[0])
        line_plot_true = np.around(np.linspace(0, self.frames, self.line_frames)).astype(int)
        count = 1
        for t in range(1, self.frames+1):
            self.update()
#             print(t)
            if t in line_plot_true:
#                 print("!")
                self.draw_line_plot(ax, cmap[count])
                count = count + 1
        self.draw_line_plot_final(ax)
       
        
class TemporalFitzHuNagReaction(SpatiotemporalModel):
    def __init__(self):
        SpatiotemporalModel.__init__(self, in_filename="Temporal_FitzHuNag.gif", in_t_max=5, in_gif_t_max=2)
        self.dt = 0.0001
        self.t = 0
        self.v, self.w = 0.1, 0.7
        self.X, self.Y_v, self.Y_w = [], [], []
        self.alpha, self.beta = 0.2, 5
    def v_Reaction(self, v, w, alpha):
        return v - v**3 - w + alpha
    def w_Reaction(self, v, w, beta):
        return (v - w)*beta
    def update(self):
        for _ in range(int(self.t_max/((self.frames+1)*self.dt))):
            self.t += self.dt
            self._update()
    def _update(self):     
        self.v += self.dt * self.v_Reaction(self.v, self.w, self.alpha) 
        self.w += self.dt * self.w_Reaction(self.v, self.w, self.beta)
    def draw(self, ax):
        ax.clear()
        
        self.X.append(self.t)
        self.Y_v.append(self.v)
        self.Y_w.append(self.w)

        ax.plot(self.X,self.Y_v, color="r", label="v")
        ax.plot(self.X,self.Y_w, color="b", label="w")
        ax.legend()
        
        ax.set_ylim(0,1)
        ax.set_xlim(0,5)
        ax.set_xlabel("t")
        ax.set_ylabel("Concentrations")
    def reset_sim(self, Y):
        self.t = 0
        self.v, self.w = 0.1, 0.7
        self.X, self.Y_v, self.Y_w = [], [], []

        
class OneDimFitzHuNagReaction(TemporalFitzHuNagReaction):
    def __init__(self):
        TemporalFitzHuNagReaction.__init__(self)
        self.Da, self.Db = 1, 100
        self.dx = 1
        self.v, self.w =  np.random.normal(loc=0, scale=0.05, size=1000), 
    np.random.normal(loc=0, scale=0.05, size=1000)
        self.dt = 0.1
    def _update(self):
        self.v += self.dt*(self.Da*laplacian1D(self.v, self.dx) + self.v_Reaction(self.v, self.w, self.alpha))
        self.w += self.dt*(self.Db*laplacian1D(self.w, self.dx) + self.w_Reaction(self.v, self.w, self.beta))
    def draw(self, ax):
        ax.clear()
        ax.plot(self.v, color="r", label="v")
        ax.plot(self.w, color="b", label="w")
        ax.legend()
        ax.set_ylim(-1,1)
        ax.set_title("t = {:.1f}".format(self.t))
    def reset_sim(self, Y):
        self.t = 0
        self.v, self.w = np.random.normal(loc=0, scale=0.05, size=1000),  np.random.normal(loc=0, scale=0.05, size=1000)
        
            
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
    def draw_line_plot(self, ax, cmap):
        ax.plot(self.X,self.Y, color=cmap, label = "t={:.1f}".format(self.t))
    def draw_line_plot_final(self, ax):
        ax.set_ylim(self.ylim[0],self.ylim[1])
        ax.get_yaxis().set_visible(False)
        ax.set_xlim(-5,5)
        ax.get_xaxis().set_visible(False)
        ax.legend()
        ax.set_title("Diffusion 1D Model")
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
    def get_equation(self):
        display(Latex(r'$$\frac{\partial Y}{\partial t} = D \frac{\partial^2 Y}{\partial^2 x}$$'))
def laplacian1D(Y, dx):
    return (-2*Y + np.roll(Y,1,axis=0) + np.roll(Y,-1,axis=0)) / (dx ** 2)
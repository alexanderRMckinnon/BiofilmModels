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
    def initialise(self):
        raise NotImplementedError()
        
    def gif_simulation(self):
        fig, ax = self.initialise_figure()
        self.initialise()
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
            
    def histogram_simulation(self):
        fig, ax = self.initialise_figure()
        self.histogram = np.zeros([self.num_x_points, self.frames+1])
        self.histogram[:,0] = self.histogram_get_slice()
        for t in range(1, self.frames+1):
            self.update()
            self.histogram[:,t] = self.histogram_get_slice()
        ax.imshow(self.histogram, cmap='Reds', interpolation='nearest', aspect='auto')
        self.histogram_draw(ax)
        plt.savefig("histogram.jpg")

    def line_plot_simulation(self):
        fig, ax = self.initialise_figure()
        cmap = pl.cm.Reds(np.linspace(1,0.3, self.line_frames))
        self.draw_line_plot(ax, cmap[0])
        line_plot_true = np.around(np.linspace(0, self.frames, self.line_frames)).astype(int)
        count = 1
        for t in range(1, self.frames+1):
            self.update()
            if t in line_plot_true:
                self.draw_line_plot(ax, cmap[count])
                count = count + 1
        self.draw_line_plot_final(ax)
        plt.savefig("line_plot.jpg")
     
    
class ModelCellDeathPattern(SpatiotemporalModel):
    def __init__(self):
        SpatiotemporalModel.__init__(self, in_filename="ModelCellDeathPattern.gif", in_t_max=1, in_gif_t_max=1/20)
        self.d, self.n, self.u_k, self.sigma, self.gamma = 0.01, 4, 1.5, 0.7, 0.05
        self.num_x_points = 300
        self.x_min = -1.5
        self.x_max = 1.5
        self.x_length = self.x_max - self.x_min
        self.dx = self.x_length/self.num_x_points
        self.u, self.du = abs(np.random.normal(loc=0, scale=0.05, size=(self.num_x_points, self.num_x_points))), self.dx
        self.w, self.dw = abs(np.random.normal(loc=0, scale=0.05, size=(self.num_x_points, self.num_x_points))), self.dx
        self.t = 0
        self.dt = (self.dx**2)/2
        self.steps = int(self.t_max/(self.frames*self.dt))
    def update(self):
        for _ in range(self.steps):
                self.t += self.dt
                self._update()
    def _update(self):
        plt.imshow(self.u)
        grad_u, grad_w = grad2D(self.u, self.dx), grad2D(self.w, self.dx)
        self.u += self.dt*( self.f(self.u) + div2D(self.D(self.w)*(grad_u - self.u*grad_w), self.dx))
        self.w += self.g(self.u, self.w) + self.d*laplacian2D(self.w, self.dx)
        plt.imshow(self.u)
        
    def f(self, u):
        return u*(1-u)
    def g(self, u, w):
        return self.sigma*( ( (u/self.u_k)**self.n )/( 1 + ((u/self.u_k)**self.n) ) ) - self.gamma*w
    def D(self, w):
        plt.imshow(w)
        plt.show()
        plt.imshow(np.exp(-w))
        plt.show()
        return np.exp(-w)
    
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
        ax[0].imshow(self.u, cmap='Reds', interpolation='lanczos')
        ax[1].imshow(self.w, cmap='Blues', interpolation='lanczos')
        ax[0].grid(b=False)
        ax[1].grid(b=False)
        ax[0].set_title("u, t = {:.1f}".format(self.t))
        ax[1].set_title("w, t = {:.1f}".format(self.t))
        
    def reset_sim(self, Y):
        self.t = 0
        if Y is None:
            self.u = abs(np.random.normal(loc=0, scale=0.05, size=(self.num_x_points, self.num_x_points)))
            self.w = abs(np.random.normal(loc=0, scale=0.05, size=(self.num_x_points, self.num_x_points)))
        else:
            self.u = Y    
    
class TwoDimFitzHuNagReaction(SpatiotemporalModel):
    def __init__(self):
        SpatiotemporalModel.__init__(self, in_filename="2DFitzHuNagReaction.gif", in_t_max=30, in_gif_t_max=15)
        self.num_x_points, self.num_y_points = 100, 100
        self.v, self.w = np.random.normal(loc=0, scale=0.05, size=(self.num_x_points, self.num_y_points)), np.random.normal(loc=0, scale=0.05, size=(self.num_x_points, self.num_y_points))
        self.D_v, self.D_w, self.a, self.b = 1, 100, -0.005, 10
        self.dx, self.dt = 1, 0.001
        self.steps = int(self.t_max/((self.frames+1)*self.dt))
    def v_Reaction(self, v, w, alpha):
        return v - v**3 - w + alpha
    def w_Reaction(self, v, w, beta):
        return (v - w)*beta
    def update(self):
        for _ in range(self.steps):
            self.t += self.dt
            self._update()
    def _update(self):
        self.v += self.dt*(self.D_v*laplacian2D(self.v, self.dx) + self.v_Reaction(self.v, self.w, self.a))
        self.w += self.dt*(self.D_w*laplacian2D(self.w, self.dx) + self.w_Reaction(self.v, self.w, self.b))
    def reset_sim(self, Y):
        self.t = 0
        self.v, self.w = np.random.normal(loc=0, scale=0.05, size=(self.num_x_points, self.num_y_points)), np.random.normal(loc=0, scale=0.05, size=(self.num_x_points, self.num_y_points))
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
        ax[0].imshow(self.v, cmap='Reds', interpolation='lanczos')
        ax[1].imshow(self.w, cmap='Blues', interpolation='lanczos')
        ax[0].grid(b=False)
        ax[1].grid(b=False)
        ax[0].set_title("v, t = {:.1f}".format(self.t))
        ax[1].set_title("w, t = {:.1f}".format(self.t))
    
class ExtendedTwoDimFitzHuNagReaction(TwoDimFitzHuNagReaction):
    def __init__(self, sim_num):
        TwoDimFitzHuNagReaction.__init__(self)
        parameters = [
            [0.16, 0.08, 0.035, 0.065], 
            [0.14, 0.06, 0.035, 0.065], 
            [0.16, 0.08, 0.06, 0.062], 
            [0.19, 0.05, 0.06, 0.062], 
            [0.16, 0.08, 0.02, 0.055], 
            [0.16, 0.08, 0.05, 0.065], 
            [0.16, 0.08, 0.054, 0.063], 
            [0.16, 0.08, 0.035, 0.06]
        ]
        self.D_v, self.D_w, self.a, self.b = parameters[sim_num][0], parameters[sim_num][1], parameters[sim_num][2], parameters[sim_num][3]
        self.t_max = 20000
        self.gif_t_max = 60
        self.fps = 2
        self.frames = int(self.fps*self.gif_t_max)
        self.steps = int(self.t_max/((self.frames+1)*self.dt))
    def v_Reaction(self, v, w, a):
        return - v*w*w + a*(1-v)
    def w_Reaction(self, v, w, a, b):
        return v*w*w - (a+b)*w
    def _update(self):
        self.v += self.dt*(self.D_v*laplacian2D(self.v, self.dx) + self.v_Reaction(self.v, self.w, self.a))
        self.w += self.dt*(self.D_w*laplacian2D(self.w, self.dx) + self.w_Reaction(self.v, self.w, self.a, self.b))
        
        
class OneDimFitzHuNagReaction(SpatiotemporalModel):
    def __init__(self):
        SpatiotemporalModel.__init__(self, in_filename="1DFitzHuNagReaction.gif", in_t_max=30, in_gif_t_max=15)
        self.num_x_points = 100
        self.D_v, self.D_w = 1, 100
        self.dx = 1
        self.v, self.w =  np.random.normal(loc=0, scale=0.05, size=self.num_x_points), np.random.normal(loc=0, scale=0.05, size=self.num_x_points)
        self.alpha, self.beta = -0.005, 10
        self.dt = 0.001
    def v_Reaction(self, v, w, alpha):
        return v - v**3 - w + alpha
    def w_Reaction(self, v, w, beta):
        return (v - w)*beta
    def update(self):
        for _ in range(int(self.t_max/((self.frames+1)*self.dt))):
            self.t += self.dt
            self._update()
    def _update(self):
        self.v += self.dt*(self.D_v*laplacian1D(self.v, self.dx) + self.v_Reaction(self.v, self.w, self.alpha))
        self.w += self.dt*(self.D_w*laplacian1D(self.w, self.dx) + self.w_Reaction(self.v, self.w, self.beta))
    def draw(self, ax):
        ax.clear()
        ax.plot(self.v, color="r", label="v")
        ax.plot(self.w, color="b", label="w")
        ax.legend()
        ax.set_ylim(-1,1)
        ax.set_xlim(0,100)
        ax.set_title("t = {:.1f}".format(self.t))
        ax.set_xlabel("x")
        ax.set_ylabel("Concentration")
        ax.get_xaxis().set_visible(False)
    def reset_sim(self, Y):
        self.t = 0
        self.v, self.w = np.random.normal(loc=0, scale=0.05, size=self.num_x_points),  np.random.normal(loc=0, scale=0.05, size=self.num_x_points)        
        
        
        
        
        
class TemporalFitzHuNagReaction(SpatiotemporalModel):
    def __init__(self):
        SpatiotemporalModel.__init__(self, in_filename="Temporal_FitzHuNag.gif", in_t_max=5, in_gif_t_max=2)
        self.dt = 0.0001
        self.t = 0
        self.v, self.w = 0.1, 0.7
        self.X, self.Y_v, self.Y_w = [], [], []
        self.alpha, self.beta = 0.2, 5
        self.steps = int(self.t_max/((self.frames+1)*self.dt))
    def v_Reaction(self, v, w, alpha):
        return v - v**3 - w + alpha
    def w_Reaction(self, v, w, beta):
        return (v - w)*beta
    def update(self):
        for _ in range(self.steps):
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
        ax.get_yaxis().set_visible(False)
        ax.set_ylabel("Concentration")
    def reset_sim(self, Y):
        self.t = 0
        self.v, self.w = 0.1, 0.7
        self.X, self.Y_v, self.Y_w = [], [], []

        

        
            
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
        self.steps = int(self.t_max/(self.frames*self.dt))
    def update(self):
        for _ in range(self.steps):
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
        ax.set_xlabel("x")
        ax.set_ylabel("Concentration")
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
def laplacian2D(Y, dx):
    return (-4*Y + np.roll(Y,1,axis=0) + np.roll(Y,-1,axis=0) + np.roll(Y,+1,axis=1) + np.roll(Y,-1,axis=1)
    ) / (dx ** 2)
def grad2D(Y, dx):
    return np.array([ (np.roll(Y, 1, axis=0)-Y)/dx, (np.roll(Y, 1, axis=1)-Y)/dx ])
def div2D(Y, dx):
    return (np.roll(Y[0], 1, axis=0)-Y[0])/dx + (np.roll(Y[1], 1, axis=1)-Y[1])/dx
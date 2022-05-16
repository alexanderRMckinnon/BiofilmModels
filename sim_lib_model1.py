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
        self.fps = 15 # Frame rate of gif, effects gif length but not simulation length
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
            if t<5 or t>self.frames+4:
                self.draw(fig, ax)
            else:
                self.update()
                self.draw(fig, ax)
        anim = animation.FuncAnimation(fig, step, frames = np.arange(self.frames+10))        
        anim.save(filename = self.filename, fps = self.fps, dpi = self.dpi)
        plt.close()
        
        with open(self.filename,'rb') as file:
            display(Image(file.read()))
            

class CellDeathModelOne(SpatiotemporalModel):
    def __init__(self, in_filename="CellDeathModelOne.gif", in_t_max=16, in_gif_t_max=4):
        SpatiotemporalModel.__init__(self, in_filename=in_filename, in_t_max=in_t_max, in_gif_t_max=in_gif_t_max)
        
    def initialise(self):
        x_min, x_max, x_numpoints = -1, 1, 1000
        x_length = x_max - x_min
        self.dx = x_length/x_numpoints
#         self.u, self.w = np.random.random([x_numpoints, x_numpoints]), np.random.random([x_numpoints, x_numpoints])
        self.u, self.w = np.zeros([x_numpoints, x_numpoints]), np.zeros([x_numpoints, x_numpoints])
        self.u[int(x_numpoints*2/5):int(x_numpoints*3/5),int(x_numpoints*2/5):int(x_numpoints*3/5)] = np.ones([int(x_numpoints/5), int(x_numpoints/5)])/3
        self.w[int(x_numpoints*2/5):int(x_numpoints*3/5),int(x_numpoints*2/5):int(x_numpoints*3/5)] = np.ones([int(x_numpoints/5), int(x_numpoints/5)])
        self.t = 0
        self.dt = (self.dx**2)/2
        self.steps = int(self.t_max/(self.frames*self.dt))
        self.min_u, self.max_u = 0, 5
        self.min_w, self.max_w = 0, 5
        
    def update(self):
        for _ in range(self.steps):
            self.t += self.dt
            self._update()
            
    def _update(self):
        self.u += self.dt*(self.u*(1-self.u))
        self.w += self.dt*(0)
    
    def initialise_figure(self):
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=np.array([16, 9]))
        self.cb1, self.cb2 = None, None
        self.colorbar_exists = 0
        return fig, ax
    
#     def draw(self, fig, ax):
#         self.u += self.u*(1-self.u)
#         self.w += 0
    
#     def initialise_figure(self):
#         fig, ax = plt.subplots(nrows=1, ncols=2, figsize=np.array([16, 9]))
#         return fig, ax
    
    def draw(self, fig, ax):
        ax[0].clear()
        ax[1].clear()
        ax[0].get_yaxis().set_visible(False)
        ax[1].get_yaxis().set_visible(False)
        ax[0].get_xaxis().set_visible(False)
        ax[1].get_xaxis().set_visible(False)
#         shw1 = ax[0].imshow(self.u, cmap='Reds', interpolation='lanczos', vmin=self.min_u, vmax = self.max_u)
#         shw2 = ax[1].imshow(self.w, cmap='Blues', interpolation='lanczos', vmin=self.min_w, vmax = self.max_w)
        shw1 = ax[0].imshow(self.u, cmap='Reds', vmin=self.min_u, vmax = self.max_u)
        shw2 = ax[1].imshow(self.w, cmap='Blues', vmin=self.min_w, vmax = self.max_w)
        ax[0].imshow(self.u, cmap='Reds', interpolation='lanczos', vmin=self.min_u, vmax = self.max_u)
        ax[1].imshow(self.w, cmap='Blues', interpolation='lanczos', vmin=self.min_w, vmax = self.max_w)
        ax[0].grid(b=False)
        ax[1].grid(b=False)
        ax[0].set_title("u, t = {:.1f}".format(self.t))
        ax[1].set_title("w, t = {:.1f}".format(self.t))
        if self.colorbar_exists == 0:
            self.cb1 = fig.colorbar(shw1, ax=ax[0])
            self.cb2 = fig.colorbar(shw2, ax=ax[1])
            self.colorbar_exists = 1
        else:
            self.cb1.remove()
            self.cb2.remove()
            self.cb1 = fig.colorbar(shw1, ax=ax[0])
            self.cb2 = fig.colorbar(shw2, ax=ax[1])
            
class CellDeathModelTwo(CellDeathModelOne):
    def __init__(self, in_filename="CellDeathModelTwo.gif", in_t_max=64, in_gif_t_max=4):
        CellDeathModelOne.__init__(self, in_filename=in_filename, in_t_max=in_t_max, in_gif_t_max=in_gif_t_max)
        self.sigma, self.u_k, self.n, self.gamma = 0.7, 1.5, 4, 0.05
    def _update(self):
        self.u += self.dt*(self.u*(1-self.u))
        self.w += self.dt*(self.sigma*( ((self.u/self.u_k)**self.n)/(1 + ((self.u/self.u_k)**self.n)) ) - self.gamma*self.w)

class CellDeathModelThree(CellDeathModelTwo):
    def __init__(self, in_filename="CellDeathModelThree.gif", in_t_max=64, in_gif_t_max=4):
        CellDeathModelTwo.__init__(self, in_filename=in_filename, in_t_max=in_t_max, in_gif_t_max=in_gif_t_max)
        self.d = 0.0001        
    def _update(self):
        self.u += self.dt*(self.u*(1-self.u))
        self.w += self.dt*(self.sigma*( ((self.u/self.u_k)**self.n)/(1 + ((self.u/self.u_k)**self.n)) ) - self.gamma*self.w + self.d*laplacian2D(self.w, self.dx))

class CellDeathModelFour(CellDeathModelThree):
    def __init__(self, in_filename="CellDeathModelFour.gif", in_t_max=1, in_gif_t_max=1):
        CellDeathModelThree.__init__(self, in_filename=in_filename, in_t_max=in_t_max, in_gif_t_max=in_gif_t_max)
        self.d = 0.01
    def _update(self):
#         print("u")
#         print(self.u)
#         print(np.shape(self.u))
        grad_u = grad2D(self.u, self.dx)
#         print("grad(u)")
#         print(grad_u)
#         print(np.shape(grad_u))
        grad_w = grad2D(self.w, self.dx)
#         print("grad(w)")
#         print(grad_w)
#         print(np.shape(grad_w))
        u_grad_w = self.u*grad_w
#         print("u*grad(w)")
#         print(np.shape(u_grad_w))
        D_u_grad_w = np.exp(-self.w)*(grad_u - u_grad_w)
#         print("D*u*grad(w)")
#         print(np.shape(D_u_grad_w))
        div_D_u_grad_w = div2D(D_u_grad_w, self.dx)
#         print("div(grad(u))=  ", np.shape(div_D_u_grad_w))
        self.u += self.dt*(self.u*(1-self.u) + div_D_u_grad_w)
        self.w += self.dt*(self.sigma*( ((self.u/self.u_k)**self.n)/(1 + ((self.u/self.u_k)**self.n)) ) - self.gamma*self.w + self.d*laplacian2D(self.w, self.dx))

def laplacian2D(Y, dx):
    return (-4*Y + np.roll(Y,1,axis=0) + np.roll(Y,-1,axis=0) + np.roll(Y,+1,axis=1) + np.roll(Y,-1,axis=1)
    ) / (dx ** 2)
def grad2D(Y, dx):
    return np.array([ (np.roll(Y, 1, axis=1)-np.roll(Y, -1, axis=1))/(2*dx), (np.roll(Y, -1, axis=0)-np.roll(Y, 1, axis=0))/(2*dx) ])
def div2D(Y, dx):
#     return ( np.roll(Y[:,:,0], 1, axis=1)-np.roll(Y[:,:,0], -1, axis=1) + np.roll(Y[:,:,1], -1, axis=0)-np.roll(Y[:,:,1], 1, axis=0) )/(2*dx)
    return ( np.roll(Y[0], 1, axis=1)-np.roll(Y[0], -1, axis=1) + np.roll(Y[1], -1, axis=0)-np.roll(Y[1], 1, axis=0) )/(2*dx)

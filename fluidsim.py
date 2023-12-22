import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.optim as optim

# from imageio import imread
from imageio.v2 import imread

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
from numpy import dtype

# Fluid simulation code based on
# "Real-Time Fluid Dynamics for Games" by Jos Stam
# http://www.intpowertechcorp.com/GDC03.pdf

def project(vx, vy):
    """Project the velocity field to be approximately mass-conserving,
       using a few iterations of Gauss-Seidel."""
    # p = np.zeros(vx.shape)
    p = torch.zeros(vx.shape, dtype=torch.float, device=device)
    h = 1.0/vx.shape[0]

    div = -0.5 * h * (torch.roll(vx, -1, dims=0) - torch.roll(vx, 1, dims=0)
                    + torch.roll(vy, -1, dims=1) - torch.roll(vy, 1, dims=1))

    for k in range(10):
        p = (div + torch.roll(p, 1, dims=0) + torch.roll(p, -1, dims=0)
                 + torch.roll(p, 1, dims=1) + torch.roll(p, -1, dims=1))/4.0
    
    vx = vx - 0.5*(torch.roll(p, -1, dims=0) - torch.roll(p, 1, dims=0))/h
    vy = vy - 0.5*(torch.roll(p, -1, dims=1) - torch.roll(p, 1, dims=1))/h

    return vx, vy


def advect(f, vx, vy):
    """Move field f according to x and y velocities (u and v)
       using an implicit Euler integrator."""
    rows, cols = f.shape
    cell_ys, cell_xs  = torch.meshgrid(torch.arange(rows), torch.arange(cols), indexing='xy')
    
    center_xs = torch.ravel(cell_xs - vx)
    center_ys = torch.ravel(cell_ys - vy)

    left_ix = torch.floor(center_xs).to(torch.int64)
    top_ix  = torch.floor(center_ys).to(torch.int64)
    
    rw = center_xs - left_ix              # Relative weight of right-hand cells.
    bw = center_ys - top_ix               # Relative weight of bottom cells.

    left_ix  = torch.remainder(left_ix,     rows)  # Wrap around edges of simulation.
    right_ix = torch.remainder(left_ix + 1, rows)
    top_ix   = torch.remainder(top_ix,      cols)
    bot_ix   = torch.remainder(top_ix  + 1, cols)

    # A linearly-weighted sum of the 4 surrounding cells.
    flat_f = (1 - rw) * ((1 - bw)*f[left_ix,  top_ix] + bw*f[left_ix,  bot_ix]) \
                 + rw * ((1 - bw)*f[right_ix, top_ix] + bw*f[right_ix, bot_ix])
                
    return torch.reshape(flat_f, (rows, cols))
                 

def simulate(vx, vy, smoke, num_time_steps, ax=None, render=False):
    print("Running simulation...")
    for t in range(num_time_steps):
        if ax: plot_matrix(ax, smoke.detach().numpy(), t, render)
        vx_updated = advect(vx, vx, vy)
        vy_updated = advect(vy, vx, vy)
        vx, vy = project(vx_updated, vy_updated)
        smoke = advect(smoke, vx, vy)
    if ax: plot_matrix(ax, smoke.detach().numpy(), num_time_steps, render)
    return smoke

def plot_matrix(ax, mat, t, render=False):
    plt.cla()
    ax.matshow(mat, cmap="Blues" )
    ax.set_xticks([])
    ax.set_yticks([])
    plt.draw()
    if render:
        matplotlib.image.imsave('step{0:03d}.png'.format(t), mat, cmap="Blues")
    plt.pause(0.001)


class FluidModel(nn.Module):
    
    def __init__(self, init_vx, init_vy, num_time_steps):
        super(FluidModel, self).__init__()
                
        self.init_vx = Parameter(init_vx)
        self.init_vy = Parameter(init_vy)
        self.num_time_steps = num_time_steps 

    def forward(self, smoke):
        smoke = smoke.clone();
        return simulate(self.init_vx, self.init_vy, smoke, self.num_time_steps)
        


if __name__ == '__main__':

    simulation_timesteps = 100

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'

    try:
       basepath = os.path.dirname(__file__)
    except NameError:
       basepath = "/home/mkrej/dyskE/MojePrg/_Python/FluidSimTorch/FluidSimTorch"

    print (basepath)

    print("Loading initial and target states...")
    init_smoke = imread(os.path.join(basepath, 'init_smoke.png'))[:,:,0]
    init_smoke = np.array(init_smoke) 
    init_smoke.shape
    init_smoke = torch.tensor(init_smoke, dtype=torch.float, device=device)
    
    #target = imread('peace.png')[::2,::2,3]
    #target = imread(os.path.join(basepath, 'skull.png'))[::2,::2]
    #target = imread(os.path.join(basepath, 'baka.png')) 
    #target = imread(os.path.join(basepath, 'asia.png'))
    #target = imread(os.path.join(basepath, 'hana.png'))
    target = imread(os.path.join(basepath, 'wiml.png'))  
    target = target[:,:,1] 
    target.shape
    target = np.array(target)
    target = torch.tensor(target, dtype=torch.float, device=device)
    
    
    rows, cols = target.shape

    init_dx_and_dy = np.zeros((2, rows, cols)).ravel()

    def distance_from_target_image(smoke):
        return torch.mean((target - smoke)**2)

    def convert_param_vector_to_matrices(params):
        vx = np.reshape(params[:(rows*cols)], (rows, cols))
        vy = np.reshape(params[(rows*cols):], (rows, cols))
        vx = torch.tensor(vx, requires_grad=True, dtype=torch.float, device=device)
        vy = torch.tensor(vy, requires_grad=True, dtype=torch.float, device=device)
        return vx, vy


    init_vx, init_vy = convert_param_vector_to_matrices(init_dx_and_dy)
    
    model = FluidModel(init_vx, init_vy, simulation_timesteps)

    lr = 1e-2

    # TODO conjugate gradient algorithm.
    optimizer = optim.SGD(model.parameters(), lr=lr)

    if False:
        vx = init_vx
        vy = init_vy
        advect(vx, vx, vy)
        smoke = init_smoke
        loss = distance_from_target_image(final_smoke)
        final_smoke = simulate(init_vx, init_vy, init_smoke, simulation_timesteps)
        final_smoke = model(init_smoke)

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, frameon=False)

    print("Optimizing initial conditions...")
    # result = minimize(objective_with_grad, init_dx_and_dy, jac=True, method='CG',
    #                   options={'maxiter':25, 'disp':True}, callback=callback)

    for k in range(100):
        model.train()
        final_smoke = model(init_smoke)
        loss = distance_from_target_image(final_smoke)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if k % 10 == 1:
           with torch.no_grad():
               simulate(model.init_vx.clone(), model.init_vy.clone(), init_smoke.clone(), simulation_timesteps, ax)
 
    print("Rendering optimized flow...")
    
    with torch.no_grad():
        simulate(model.init_vx.clone(), model.init_vy.clone(), init_smoke.clone(), simulation_timesteps, ax, render=True)

    print("Converting frames to an animated GIF...")
    
    os.system("convert -delay 5 -loop 0 step*.png -delay 250 result.gif")  # Using imagemagick.

    os.system("rm step*.png")
 
    print("--- done ---")    


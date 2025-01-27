"""
Visualization of KE diffusion.

Usage:
    diffusion.py <exp> <files>... [--output=<dir>]

Options:
    --output=<dir>  Output directory [default: ./frames]

"""

import h5py
import numpy as np
import matplotlib
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.pyplot import rc
from matplotlib import use, rc
from dedalus import public as de
from numpy import load
from docopt import docopt
from scipy import integrate


use('Agg')
plt.ioff()

#Parameters
Lx, H = 16, 1
nx, nz = 2048, 128
nt = 200 #Simulation outputs
Tf = 2*np.pi #Inertial periods

def main(filename, start, count, output):
    """ Plot temporal and spatial average """
    
    args = docopt(__doc__)
    exp = str(args['<exp>'])

    x_basis = de.Fourier('x', nx, interval=(-Lx/2, Lx/2), dealias=3/2)
    z_basis = de.Chebyshev('z', nz, interval=(-H, 0.), dealias=3/2)
    domain = de.Domain([x_basis, z_basis], grid_dtype=np.float64)
    x, z = domain.grids(scales=1)
    
    with h5py.File(filename, mode='r') as file:
        t = np.array(file['scales']['sim_time'])/Tf    
    
    T,X,Z = np.meshgrid(t,x,z, indexing='ij')
    
    field_ini = load('diffusion-ini-' + exp + '.npy')
    field_eii = load('diffusion-eii-' + exp + '.npy')
    
    field_hor_avg_ini = (1/Lx)*integrate.simpson(field_ini, X[0,:,0], axis=1)
    field_hor_avg_eii = (1/Lx)*integrate.simpson(field_eii, X[0,:,0], axis=1)
    
    field_temp_avg_ini = (1/nt)*integrate.simpson(field_ini, t, axis=0)
    field_temp_avg_eii = (1/nt)*integrate.simpson(field_eii, t, axis=0)
    
    # Plot settings
    #plt.style.use('dark_background')
    hsize = 9 #Image horizontal size
    vsize = 6 #Image vertical size
    fs = 12
    
    lim_inf = -3.5e-1
    lim_sup = 1.5e-1
    nlev = 100 #Number of contours
    levels = np.linspace(-3.5e-1, 1.5e-1, nlev)
    
    class MidpointNormalize(colors.Normalize):
        def __init__(self, vmin=None, vmax=None, vcenter=None, clip=False):
            self.vcenter = vcenter
            super().__init__(vmin, vmax, clip)

        def __call__(self, value, clip=None):
            # I'm ignoring masked values and all kinds of edge cases to make a
            # simple example...
            # Note also that we must extrapolate beyond vmin/vmax
            x, y = [self.vmin, self.vcenter, self.vmax], [0, 0.5, 1.]
            return np.ma.masked_array(np.interp(value, x, y,
                                                left=-np.inf, right=np.inf))

        def inverse(self, value):
            y, x = [self.vmin, self.vcenter, self.vmax], [0, 0.5, 1]
            return np.interp(value, x, y, left=-np.inf, right=np.inf)


    midnorm = MidpointNormalize(vmin=-3.25e-1, vcenter=0, vmax=1.5e-1)
    midnorm = MidpointNormalize(vmin=-3.5e-2, vcenter=0, vmax=1.5e-2)
    midnorm = MidpointNormalize(vmin=-3.5e-1, vcenter=0, vmax=1.5e-1)
    
    font = {'family': 'serif',
        'weight': 'normal',
        'size': 12,
        }
        
    rc('font', **font)
    
    plt.rcParams['text.usetex'] = True
    
    #Limits of domain to plot
    x_left = nx//4
    x_right = 3*nx//4
    
    #PLOT!   
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, sharey=False, sharex=False, figsize=(hsize, vsize))
    CS = ax1.contourf(X[0,x_left:x_right,:],Z[0,x_left:x_right,:],field_temp_avg_ini[x_left:x_right,:], cmap='PuOr', norm=midnorm, levels=levels, extend='both')
    CS3 = ax3.contourf(T[:,0,:],Z[:,0,:],field_hor_avg_ini, cmap='PuOr', norm=midnorm, levels=levels, extend='both')
    ax1.set_xlabel('$x^\prime$', fontsize=fs)
    ax1.set_ylabel('$z$', fontsize=fs)
    ax3.set_xlabel('$t/2\pi$', fontsize=fs)
    ax3.set_ylabel('$z$', fontsize=fs)
    ax1.set_xticks([-4,-2,0,2,4])
    ax1.set_yticks([-1,-0.5,0])
    ax3.set_xticks([0,5,10,15,20])
    ax3.set_yticks([-1,-0.5,0])
    ax1.set_title('a) Inertial Instability \n Temporal Average', fontsize=fs, linespacing = 1.5)
    ax3.set_title('c) Horizontal Average', fontsize=fs, linespacing = 1.5)
    CS2=ax2.contourf(X[0,x_left:x_right,:],Z[0,x_left:x_right,:],field_temp_avg_eii[x_left:x_right,:], cmap='PuOr', norm=midnorm, levels=levels, extend='both')
    ax2.set_xlabel('$x^\prime$', fontsize=fs)
    CS4=ax4.contourf(T[:,0,:],Z[:,0,:],field_hor_avg_eii, cmap='PuOr', norm=midnorm, levels=levels, extend='both')
    ax2.set_xlabel('$x^\prime$', fontsize=fs)
    ax2.set_xticks([-4,-2,0,2,4])
    ax2.set_yticks([])
    ax4.set_xticks([0,5,10,15,20])
    ax4.set_yticks([])
    ax4.set_xlabel('$t/2\pi$', fontsize=fs)
    ax2.set_title('b) Ekman-Inertial Instability \n Temporal Average', fontsize=fs, linespacing = 1.5)
    ax4.set_title('d) Horizontal Average', fontsize=fs, linespacing = 1.5)
    fig.subplots_adjust(bottom=0.25, top=0.92, left=0.075, right=0.95, wspace=0.25, hspace=0.5)
    cb_ax = fig.add_axes([0.07, 0.1, 0.88, 0.03])
    cbar = fig.colorbar(CS, ticks=[-3.5e-1, -3e-1, -2.5e-1, -2e-1, -1.5e-1, -1e-1, -0.5e-1, 0, 0.5e-1, 1e-1, 1.5e-1], cax=cb_ax, orientation='horizontal')
    cbar.formatter.set_powerlimits((0, 0))
    cbar.set_label(r'$\displaystyle \mathcal{D}$', fontsize=fs,labelpad=7)
    cbar.ax.tick_params(labelsize=fs)
    # Save figure
    fig.savefig(output.joinpath('diffusion.png'), dpi=300)
    plt.close(fig)
    
if __name__ == "__main__":

    import pathlib
    from docopt import docopt
    from dedalus.tools import post
    from dedalus.tools.parallel import Sync

    args = docopt(__doc__)

    output_path = pathlib.Path(args['--output']).absolute()
    # Create output directory if needed
    with Sync() as sync:
        if sync.comm.rank == 0:
            if not output_path.exists():
                output_path.mkdir()
    post.visit_writes(args['<files>'], main, output=output_path)

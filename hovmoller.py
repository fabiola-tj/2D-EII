"""
Create Hovmoller plots from a given field.

Usage:
    hovmoller.py <field> <exp> <files>... [--output=<dir>]

Options:
    --output=<dir>  Output directory [default: ./frames]

"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib import use, rc
from matplotlib.pyplot import rc
from dedalus import public as de
from numpy import load
from docopt import docopt

use('Agg')
plt.ioff()

#Parameters
Lx, H = 16, 1
nx, nz = 2048, 128
nt = 200 #Simulation outputs
Tf = 2*np.pi #Inertial periods

def main(filename, output):
    """ Create Hovmoller plots. """
    
    args = docopt(__doc__)
    
    field_str = str(args['<field>']) 
    exp = str(args['<exp>'])

    x_basis = de.Fourier('x', nx, interval=(-Lx/2, Lx/2), dealias=3/2)
    z_basis = de.Chebyshev('z', nz, interval=(-H, 0.), dealias=3/2)
    domain = de.Domain([x_basis, z_basis], grid_dtype=np.float64)
    x, z = domain.grids(scales=1) 
    
    with h5py.File(filename, mode='r') as file:
        t = np.array(file['scales']['sim_time'])/Tf
    
    T,X = np.meshgrid(t,x) 
    
    #Load field to create Hovmoller plots of, e.g. 'w'
    field_ini = load(field_str + '-ini-' + exp + '.npy')
    field_eii = load(field_str + '-eii-' + exp + '.npy')
    
    #Load Rossby number field
    ro_ini = load('dv_dx-shifted-ini-' + exp + '.npy')
    ro_eii = load('dv_dx-shifted-eii-' + exp + '.npy')
    
    #Plot settings
    #plt.style.use('dark_background')
    hsize = 16 #Image horizontal size
    vsize = 8 #Image vertical size
    fs = 28
    
    font = {'family': 'serif',
        'weight': 'normal',
        'size': 28,
        }
        
    rc('font', **font)

    plt.rcParams['text.usetex'] = True
    
    #Contour levels settings
    lim = np.abs(field_eii).max()
    nlev = 100 #Number of contours
    levels = np.linspace(-lim, lim, nlev)        #Contours for field
    levels_stable = np.linspace(-0.7, -0.7, 1)   #Contours for marginally stable Ro 
    levels_unstable = np.linspace(-1.0, -1.0, 1) #Contours for unstable Ro
    
    locz=int(0.5*nz) #Select mid-depth level for the plots
    
    fig, ((ax1, ax2)) = plt.subplots(nrows=1, ncols=2, sharey=all, sharex=all, figsize=(hsize, vsize))
    CS = ax1.contourf(X[:,:],T[:,:],np.transpose(field_ini[:,:,locz]), cmap='seismic', levels=levels, extend='max')
    CS2 = ax1.contour(X[:,:],T[:,:],np.transpose(ro_ini[:,:,locz]), colors='lightseagreen', extend='max', levels=levels2, linewidths=1, linestyles='solid')
    CS3 = ax1.contour(X[:,:],T[:,:],np.transpose(ro_ini[:,:,locz]), colors='darkgoldenrod', extend='max', levels=levels3, linewidths=1, linestyles='dashed')
    ax1.set_ylabel('$t/2\pi$', fontsize=fs)
    ax1.set_xlabel('$x^\prime$', fontsize=fs)
    ax1.set_title('a) Inertial Instability', fontsize=fs, linespacing = 1.5)
    CS4=ax2.contourf(X[:,:],T[:,:],np.transpose(field_eii[:,:,locz]), cmap='seismic', levels=levels, extend='max')
    CS5=ax2.contour(X[:,:],T[:,:],np.transpose(ro_eii[:,:,locz]), colors='lightseagreen', extend='max', levels=levels2, linewidths=1, linestyles='solid')
    CS6=ax2.contour(X[:,:],T[:,:],np.transpose(ro_eii[:,:,locz]), colors='darkgoldenrod', extend='max', levels=levels3, linewidths=1, linestyles='dashed')
    ax2.set_xlabel('$x^\prime$', fontsize=fs)
    ax2.set_title('b) Ekman-Inertial Instability', fontsize=fs, linespacing = 1.5)
    fig.subplots_adjust(bottom=0.27, top=0.9, left=0.075, right=0.95, wspace=0.15, hspace=0.5)
    plt.xticks(np.arange(-Lx/2,Lx/(1*2) + 1,4), fontsize=fs)
    plt.yticks(np.arange(0, 21, 5), fontsize=fs)
    cb_ax = fig.add_axes([0.07, 0.12, 0.88, 0.03])
    cbar = fig.colorbar(CS, ticks=[-3.5e-1, -1.7e-1, 0, 1.7e-1, 3.5e-1], cax=cb_ax, orientation='horizontal')
    cbar.formatter.set_powerlimits((0, 0))
    cbar.set_label('$w$', fontsize=fs,labelpad=7)
    cbar.ax.tick_params(labelsize=fs)
    # Save figure
    fig.savefig(output.joinpath('hovmoller'), format='png', dpi=300)
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

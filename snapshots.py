"""
Plot planes from joint analysis files.

Usage:
    snapshots.py <field> <exp> <files>... [--output=<dir>]

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
    """ Save plot of evolution over time. """
    
    args = docopt(__doc__)
    
    field_str = str(args['<field>']) 
    exp = str(args['<exp>']) 

    x_basis = de.Fourier('x', nx, interval=(-Lx/2, Lx/2), dealias=3/2)
    z_basis = de.Chebyshev('z', nz, interval=(-H, 0.), dealias=3/2)
    domain = de.Domain([x_basis, z_basis], grid_dtype=np.float64)
    x, z = domain.grids(scales=1)
    Z, X = np.meshgrid(z, x)

    with h5py.File(filename, mode='r') as file:
        t = np.array(file['scales']['sim_time'])/Tf

    #Load field to create snapshots of, e.g. 'w'
    field_ini = load(field_str + '-ini-' + exp + '.npy')
    field_eii = load(field_str + '-eii-' + exp + '.npy')
    
    #Load Rossby number field
    ro_ini = load('dv_dx-shifted-ini-' + exp + '.npy')
    ro_eii = load('dv_dx-shifted-eii-' + exp + '.npy')
    
    # Plot settings
    #plt.style.use('dark_background')
    hsize = 7 #Image horizontal size
    vsize = 7 #Image vertical size
    fs = 12
    
    font = {'family': 'serif',
        'weight': 'normal',
        'size': 12,
        }
        
    rc('font', **font)
    
    plt.rcParams['text.usetex'] = True

    #Contour levels settings
    lim = np.abs(field_eii).max()
    nlev = 100 #Number of contours
    levels = np.linspace(-lim, lim, nlev)       #Contours for field
    levels_stable = np.linspace(-0.99, -0.7, 2) #Contours for marginally stable Ro
    levels_unstable = np.linspace(-100, -1, 2)  #Contours for unstable Ro

    #Limits of domain to plot
    x_left = nx//4
    x_right = 3*nx//4

    #Times for the snapshots
    #EII
    t_onset_eii = 18
    t_interm_eii = 25
    t_weak_eii = 32
    t_stable_eii = 60
    #INI
    t_onset_ini = 75
    t_interm_ini = 82
    t_weak_ini = 100
    t_stable_ini = 120

    fig, ((ax1, ax2), (ax3, ax4), (ax5,ax6), (ax7,ax8)) = plt.subplots(nrows=4, ncols=2, sharey=all, sharex=all, figsize=(hsize, vsize))
    CS = ax1.contourf(X[x_left:x_right,:], Z[x_left:x_right,:], field_ini[t_onset_ini,x_left:x_right,:], cmap='seismic', levels=levels, extend='max')
    CS2 = ax1.contourf(X[x_left:x_right,:],Z[x_left:x_right,:],ro_ini[t_onset_ini,x_left:x_right,:], colors='darkgoldenrod', extend='neither', levels=levels_stable, alpha=0.2)
    CS3 = ax1.contourf(X[x_left:x_right,:],Z[x_left:x_right,:],ro_ini[t_onset_ini,x_left:x_right,:], colors='seagreen', extend='neither', levels=levels_unstable, alpha=0.3)
    ax1.set_ylabel('$z$', fontsize=fs)
    ax1.set_title('Inertial Instability \n (a) $t/2\pi=7.5$', fontsize=fs, linespacing = 1.5)
    ax2.contourf(X[x_left:x_right,:], Z[x_left:x_right,:], field_eii[t_onset_eii,x_left:x_right,:], cmap='seismic', levels=levels, extend='max')
    ax2.contourf(X[x_left:x_right,:],Z[x_left:x_right,:],ro_eii[t_onset_eii,x_left:x_right,:], colors='darkgoldenrod', extend='neither', levels=levels_stable, alpha=0.2)
    ax2.contourf(X[x_left:x_right,:],Z[x_left:x_right,:],ro_eii[t_onset_eii,x_left:x_right,:], colors='seagreen', extend='neither', levels=levels_unstable, alpha=0.3)
    ax2.set_title('Ekman-Inertial Instability \n (b) $t/2\pi=1.8$', fontsize=fs, linespacing = 1.5)
    ax3.contourf(X[x_left:x_right,:], Z[x_left:x_right,:], field_ini[t_interm_ini,x_left:x_right,:], cmap='seismic', levels=levels, extend='max')
    ax3.contourf(X[x_left:x_right,:],Z[x_left:x_right,:],ro_ini[t_interm_ini,x_left:x_right,:], colors='darkgoldenrod', extend='neither', levels=levels_stable, alpha=0.2)
    ax3.set_ylabel('$z$', fontsize=fs)
    ax3.contourf(X[x_left:x_right,:],Z[x_left:x_right,:],ro_ini[t_interm_ini,x_left:x_right,:], colors='seagreen', extend='neither', levels=levels_unstable, alpha=0.3)
    ax3.set_title('(c) $t/2\pi=8.2$', fontsize=fs)
    ax4.contourf(X[x_left:x_right,:], Z[x_left:x_right,:], field_eii[t_interm_eii,x_left:x_right,:], cmap='seismic', levels=levels, extend='max')
    ax4.contourf(X[x_left:x_right,:],Z[x_left:x_right,:],ro_eii[t_interm_eii,x_left:x_right,:], colors='darkgoldenrod', extend='neither', levels=levels_stable, alpha=0.2)
    ax4.contourf(X[x_left:x_right,:],Z[x_left:x_right,:],ro_eii[t_interm_eii,x_left:x_right,:], colors='seagreen', extend='neither', levels=levels_unstable, alpha=0.3)
    ax4.set_title('(d) $t/2\pi=2.5$', fontsize=fs)
    ax5.contourf(X[x_left:x_right,:], Z[x_left:x_right,:], field_ini[t_weak_ini,x_left:x_right,:], cmap='seismic', levels=levels, extend='max')
    ax5.contourf(X[x_left:x_right,:],Z[x_left:x_right,:],ro_ini[t_weak_ini,x_left:x_right,:], colors='darkgoldenrod', extend='neither', levels=levels_stable, alpha=0.2)
    ax5.contourf(X[x_left:x_right,:],Z[x_left:x_right,:],ro_ini[t_weak_ini,x_left:x_right,:], colors='seagreen', extend='neither', levels=levels_unstable, alpha=0.3)
    ax5.set_ylabel('$z$', fontsize=fs)
    ax5.set_title('(e) $t/2\pi=10.0$', fontsize=fs)
    ax6.contourf(X[x_left:x_right,:], Z[x_left:x_right,:], field_eii[t_weak_eii,x_left:x_right,:], cmap='seismic', levels=levels, extend='max')
    ax6.contourf(X[x_left:x_right,:],Z[x_left:x_right,:],ro_eii[t_weak_eii,x_left:x_right,:], colors='darkgoldenrod', extend='neither', levels=levels_stable, alpha=0.2)
    ax6.contourf(X[x_left:x_right,:],Z[x_left:x_right,:],ro_eii[t_weak_eii,x_left:x_right,:], colors='seagreen', extend='neither', levels=levels_unstable, alpha=0.3)
    ax6.set_title('(f) $t/2\pi=3.2$', fontsize=fs)
    ax7.contourf(X[x_left:x_right,:], Z[x_left:x_right,:], field_ini[t_stable_ini,x_left:x_right,:], cmap='seismic', levels=levels, extend='max')
    ax7.contourf(X[x_left:x_right,:],Z[x_left:x_right,:],ro_ini[t_stable_ini,x_left:x_right,:], colors='darkgoldenrod', extend='neither', levels=levels_stable, alpha=0.2)
    ax7.contourf(X[x_left:x_right,:],Z[x_left:x_right,:],ro_ini[t_stable_ini,x_left:x_right,:], colors='seagreen', extend='neither', levels=levels_unstable, alpha=0.3)
    ax7.set_xlabel('$x^\prime$', fontsize=fs)
    ax7.set_ylabel('$z$', fontsize=fs)
    ax7.set_title('(g) $t/2\pi=12.0$', fontsize=fs)
    ax8.contourf(X[x_left:x_right,:], Z[x_left:x_right,:], field_eii[t_stable_eii,x_left:x_right,:], cmap='seismic', levels=levels, extend='max')
    ax8.contourf(X[x_left:x_right,:],Z[x_left:x_right,:],ro_eii[t_stable_eii,x_left:x_right,:], colors='darkgoldenrod', extend='neither', levels=levels_stable, alpha=0.2)
    ax8.contourf(X[x_left:x_right,:],Z[x_left:x_right,:],ro_eii[t_stable_eii,x_left:x_right,:], colors='seagreen', extend='neither', levels=levels_unstable, alpha=0.3)
    ax8.set_xlabel('$x^\prime$', fontsize=fs)
    ax8.set_title('(h) $t/2\pi=6.0$', fontsize=fs)
    fig.subplots_adjust(bottom=0.18, top=0.9, left=0.12, right=0.95, wspace=0.15, hspace=0.5)
    plt.xticks(np.arange(-Lx/4,Lx/4 + 1,2), fontsize=fs)
    plt.yticks(np.arange(-1,0.1,.5), fontsize=fs)
    cb_ax = fig.add_axes([0.1, 0.08, 0.85, 0.02])
    cbar = fig.colorbar(CS, ticks=[-3.5e-1, -1.7e-1, 0, 1.7e-1, 3.5e-1], cax=cb_ax, orientation='horizontal')
    cbar.formatter.set_powerlimits((0, 0))
    cbar.set_label('$w$', labelpad=5)
    # Save figure
    fig.savefig(output.joinpath('snapshots'), format='png', dpi=300)
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

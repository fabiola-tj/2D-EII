"""
Plot time series of growth rates (EII & InI) for multiple Rossby numbers

Usage:
    gr-ro-sweep.py <files>... [--output=<dir>]

Options:
    --output=<dir>  Output directory [default: ./frames]

"""

import h5py
import numpy as np
from matplotlib.pyplot import rc
import matplotlib.pyplot as plt
import matplotlib.pylab as plt
from matplotlib import use
from numpy import load
from docopt import docopt
import matplotlib as mpl

use('Agg')
plt.ioff()

#Parameters
Lx, H = 16, 1
nx, nz = 2048, 128
nt = 200 #Simulation outputs
Tf = 2*np.pi #Inertial periods

def main(filename, start, count, output):
    """ Plot time series of growth rate """

    with h5py.File(filename, mode='r') as file:       
        t = np.array(file['scales']['sim_time'])/Tf
            
    nexp = 9 #Number of Rossby experiments
    
    growth_rate_full_ini = np.empty([nexp,nt-1])
    growth_rate_full_eii = np.empty([nexp,nt-1])
    for i in range(nexp):
        exp_id = str(11 + i)
        growth_rate_full_ini[i,:] = load('growth-rate-ini-' + exp_id + '.npy')
        growth_rate_full_eii[i,:] = load('growth-rate-eii-' + exp_id + '.npy')
        
    #Plot settings
    #plt.style.use('dark_background')
    hsize = 13 #Image horizontal size
    vsize = 5 #Image vertical size    
    color_ini = plt.cm.Purples(np.linspace(0, 0.95, nexp))
    color_eii = plt.cm.Oranges(np.linspace(0, 0.95, nexp))
        
    font = {'family': 'serif',
    'weight': 'normal',
    'size': 24,
    }
    
    rc('font', **font)
    
    plt.rcParams['text.usetex'] = True
        
    #PLOT!
    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(hsize, vsize))
        ax.plot(t[:-81], growth_rate_full_eii[4,:-80], label = r'$\sigma_{\mathrm{EII}}|_{\mathrm{Ro}_{m} = -1.5}$', color=color_eii[6], marker='.')
        ax.plot(t[:-81], growth_rate_full_ini[4,:-80], label = r'$\sigma_{\mathrm{InI}}|_{\mathrm{Ro}_{m} = -1.5}$', color=color_ini[6], marker='.')
        for i in range(nexp):
            ax.plot(t[:-81], growth_rate_full_eii[i,:-80], color=color_eii[i])
            ax.plot(t[:-81], growth_rate_full_ini[i,:-80], color=color_ini[i])
        ax.grid(True)
        ax.set_xlabel('$t/2\pi$')
        ax.set_ylabel('$\sigma$', fontsize=26)
        cb_ax_1 = fig.add_axes([0.78, 0.12, 0.015, 0.85]) #left,bottom, width, height
        cb_ax_2 = fig.add_axes([0.9, 0.12, 0.015, 0.85]) #left,bottom, width, height
        fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(-1.0, -1.9), cmap='Oranges_r'), cax=cb_ax_1, orientation='vertical', label='$\mathrm{Ro}_{\mathrm{min}}$ (EII)', ticks=[-1.9, -1.7, -1.5, -1.3, -1.1], extend='both')
        fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(-1.0, -1.9), cmap='Purples_r'), cax=cb_ax_2, orientation='vertical', label='$\mathrm{Ro}_{\mathrm{min}}$ (InI)', ticks=[-1.9, -1.7, -1.5, -1.3, -1.1], extend='both')
        fig.subplots_adjust(bottom=0.18, top=0.95, left=0.07, right=0.75, wspace=0.15, hspace=0.15)
        fig.savefig(output.joinpath('gr-ro-sweep.png'), dpi=300)
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
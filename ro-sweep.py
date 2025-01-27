"""
Plot max. growth rates (EII & InI) as function of Rossby number

Usage:
    ro-sweep.py <files>... [--output=<dir>]

Options:
    --output=<dir>  Output directory [default: ./frames]

"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as plt
from matplotlib import use, rc
from numpy import load
from docopt import docopt

use('Agg')
plt.ioff()

#Parameters
Lx, H = 16, 1
nx, nz = 2048, 128
nt = 200 #Simulation outputs
Tf = 2*np.pi #Inertial periods

def main(filename, start, count, output):
    """ Plot max. growth rate as function of Ro """

    with h5py.File(filename, mode='r') as file:
        t = np.array(file['scales']['sim_time'])/Tf

    #Define sweep range
    Ro_min = 11
    Ro_max = 19
    Rossby = np.arange(Ro_min, Ro_max+1)
    num_exp = len(Rossby)

    Rossby_ticks = np.empty([num_exp])
    for i in range(num_exp):
        Rossby_ticks[i] = -1.1 - i*0.1

    #Sigma from linear stability analysis
    sigma_theory = np.sqrt(-1-Rossby_ticks)

    #Plot settings
    #plt.style.use('dark_background')
    hsize = 7 #Image horizontal size
    vsize = 3 #Image vertical size

    font = {'family': 'serif',
        'weight': 'normal',
        'size': 12,
        }

    rc('font', **font)

    plt.rcParams['text.usetex'] = True  
    
    #Compute max. growth rate of perturbations
    max_ini = np.empty([num_exp])
    max_eii = np.empty([num_exp])
    max_eii_4 = np.empty([num_exp])
    max_eii_7 = np.empty([num_exp])
    max_eii_9 = np.empty([num_exp])
    
    for i in range(num_exp):
        aux = 11 + i
        exp_ini = 'ini-' + str(aux)
        exp_eii = 'eii-' + str(aux)
        gr_ini = load('grp-' + exp_ini + '.npy')
        gr_eii = load('grp-' + exp_eii + '.npy')
        gr_ini = gr_ini.tolist()
        gr_eii = gr_eii.tolist()
        max_ini[i] = max(gr_ini[5:]) #Compute max skipping transient growth rate regime
        max_eii[i] = max(gr_eii[5:])
         
        Rossby_ticks = np.empty([num_exp])
        for i in range(num_exp):
            Rossby_ticks[i] = -1.1 - i*0.1

        #Sigma from linear stability analysis
        sigma_theory = np.sqrt(-1-Rossby_ticks)
        
        #PLOT!
        fig, ax = plt.subplots(nrows=1, ncols=1, sharex=all, figsize=(hsize, vsize))
        fig.subplots_adjust(hspace=0.08)
        ax.plot(Rossby_ticks, max_eii, label = r'$\mathrm{EII}$', color='y', marker='.')
        ax.plot(Rossby_ticks, max_ini, label = r'$\mathrm{InI}$', color='c', marker='.')
        ax.plot(Rossby_ticks, sigma_theory, label = r'$\sigma_{\mathrm{ref}}$', color='k', linewidth=2.5, marker='.')
        ax.set_yticks([0,0.2,0.4,0.6,0.8,1])
        ax.invert_xaxis()
        ax.grid()
        ax.legend(loc='lower right')
        ax.set_xlabel(r'$\mathrm{Ro}_{\mathrm{min}}$')
        ax.set_ylabel(r'$\sigma_{\mathrm{max}}$', fontsize=14)
        fig.subplots_adjust(bottom=0.18, top=0.95, left=0.092, right=0.97, wspace=0.25, hspace=0.15)
        fig.savefig(output.joinpath('ro-sweep.png'), dpi=300)
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
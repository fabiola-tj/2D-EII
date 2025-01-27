"""
Plot energy budget (EII & InI) for a given Rossby number

Usage:
    eb-plot.py <nexp> <files>... [--output=<dir>]

Options:
    --output=<dir>  Output directory [default: ./frames]

"""

import matplotlib.pyplot as plt
import matplotlib.pylab as plt
import h5py
import numpy as np
from matplotlib import use, rc
from matplotlib.pyplot import rc
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
    """ Plot energy budget """
    
    args = docopt(__doc__)

    nexp = str(args['<nexp>']) 
    
    with h5py.File(filename, mode='r') as file:
        t = np.array(file['scales']['sim_time'])      
    
    LHS_ini = load('lhs-eb-ini-' + nexp + '.npy')
    LHS_eii = load('lhs-eb-eii-' + nexp + '.npy')
    advection_ini = load('advection-ini-' + nexp + '.npy')
    advection_eii = load('advection-eii-' + nexp + '.npy')
    pressure_ini = load('pressure-ini-' + nexp + '.npy')
    pressure_eii = load('pressure-eii-' + nexp + '.npy')
    diffusion_ini = load('diffusion-ini-' + nexp + '.npy')
    diffusion_eii = load('diffusion-eii-' + nexp + '.npy')
    dissipation_ini = load('dissipation-ini-' + nexp + '.npy')
    dissipation_eii = load('dissipation-eii-' + nexp + '.npy')
    advection_ini = load('advection-ini-' + nexp + '.npy')
    advection_eii = load('advection-eii-' + nexp + '.npy')
    
    # Plot settings
    #plt.style.use('dark_background')
    hsize = 12 #Image horizontal size
    vsize = 7  #Image vertical size
    fs = 20

    font = {'family': 'serif',
    'weight': 'normal',
    'size': 20,
    }
    
    rc('font', **font)
    
    plt.rcParams['text.usetex'] = True
    
    color_mine = plt.cm.plasma(np.linspace(0, 0.95, 16))
    
    
    #Plot!
    fig, ((ax1, ax2)) = plt.subplots(nrows=2, ncols=1, sharey=all, sharex=all, figsize=(hsize, vsize))
    ax1.plot((t/Tf)[:-76], LHS_ini[:-75], label=r'$\displaystyle \frac{d}{d t} \langle K \rangle$',color='maroon', marker='.', alpha=0.8, linewidth=2)
    ax1.plot((t/Tf)[:-75], diffusion_ini[:-75], label=r'$\displaystyle \mathcal{D}$',color=color_mine[12], linewidth=2)
    ax1.plot((t/Tf)[:-75], -dissipation_ini[:-75], label=r'$\displaystyle \mathcal{E}$',color=color_mine[15], linewidth=2)
    ax1.set_xlim((0,12))
    ax1.ticklabel_format(axis='y',scilimits=(0,0),useMathText=True)
    ax1.set_title('a) Inertial Instability', fontsize=fs, linespacing = 1.5)
    ax1.legend(loc='lower left')
    ax1.grid()
    ax2.plot(t[:-76]/Tf, LHS_eii[:-75], color='maroon', marker='.', alpha=0.8, linewidth=2)
    ax2.plot((t/Tf)[:-75], diffusion_eii[:-75], color=color_mine[12], linewidth=2)
    ax2.plot((t/Tf)[:-75], -dissipation_eii[:-75], color=color_mine[15], linewidth=2)
    ax2.set_xlabel('$t/2\pi$', fontsize=fs)
    ax2.set_xlim((0,12))
    ax2.set_title('b) Ekman-Inertial Instability', fontsize=fs, linespacing = 1.5)
    ax2.grid()
    fig.subplots_adjust(bottom=0.1, top=0.95, left=0.05, right=0.98, wspace=0.25, hspace=0.5)
    # Save figure
    fig.savefig(output.joinpath('eb-sample-'+nexp+'.png'), dpi=300)
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

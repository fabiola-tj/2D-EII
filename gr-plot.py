"""
Plot time series of growth rates (EII & InI) for a given Rossby number

Usage:
    gr-plot.py <nexp> <files>... [--output=<dir>]

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
    """ Plot growth rates """
    
    args = docopt(__doc__)
    
    nexp = str(args['<nexp>'])

    with h5py.File(filename, mode='r') as file:
        t = np.array(file['scales']['sim_time'])/Tf
        
    #Load growth rates for each instability
    growth_rate_ini_15 = load('growth-rate-ini-15.npy')
    growth_rate_eii_15 = load('growth-rate-eii-15.npy')
    
    print(growth_rate_ini_15.shape)
    print(growth_rate_eii_15.shape)
    
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

    #Growth rate from linear stability analysis
    sigma_ref = 0.7*np.ones([120])
    print(sigma_ref)
    print(sigma_ref.shape)
                    
    #Plot!
    fig, ax = plt.subplots(1, 1, figsize=(hsize,vsize))
    fig.subplots_adjust(hspace=0.08)
    ax.plot(t[:-81], growth_rate_eii_15[:-80], label = r'$\mathrm{EII}$', color='y', marker='.')
    ax.plot(t[:-81], growth_rate_ini_15[:-80], label = r'$\mathrm{InI}$', color='c', marker='.')
    ax.plot(t[:-80], sigma_ref, label = r'$\sigma_{\mathrm{ref}}$', color = 'k', linewidth=2.5)
    ax.grid()
    ax.set_ylabel('$\sigma$', fontsize=14)
    ax.set_xlabel('$t/2\pi$')
    plt.legend()
    fig.subplots_adjust(bottom=0.18, top=0.95, left=0.092, right=0.97, wspace=0.25, hspace=0.15)
    fig.savefig(output.joinpath('gr-sample-'+nexp+'.png'), dpi=300)
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
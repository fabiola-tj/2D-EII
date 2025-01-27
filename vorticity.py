"""
Script to create vorticity histograms

Usage:
    vorticity.py <exp> <files>... [--output=<dir>]

Options:
    --output=<dir>  Output directory [default: ./frames]

"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as plt
from dedalus import public as de
from matplotlib.colors import Normalize
from matplotlib.pyplot import rc
from matplotlib import use
from numpy import load
from docopt import docopt

use('Agg')
plt.ioff()

#Parameters
Lx, H = 16, 1
nx, nz = 2048, 128
nt = 200 #Simulation outputs
Tf = 2*np.pi #Inertial periods

def label_time(t): return '$t/2\pi$ = {:.0f}'.format(t)

def main(filename, output):
    """ Create vorticity histograms """
    
    args = docopt(__doc__)
    
    exp = str(args['<exp>'])

    with h5py.File(filename, mode='r') as file:
        
        t = np.array(file['scales']['sim_time'])/Tf #6 periods

    #Load vorticity field
    rossby_ini_data = load('dv_dx-ini-' + exp + '.npy')
    rossby_eii_data = load('dv_dx-eii-' + exp + '.npy')
        
    # Plot settings
    #plt.style.use('dark_background')
    hsize = 16#Image horizontal size
    vsize = 8 #Image vertical size
    fs=28 #Fontsize
        
    colors = plt.cm.viridis(np.linspace(0, 0.95, 20))
    
    font = {'family': 'serif',
        'weight': 'normal',
        'size': 28,
        }
        
    rc('font', **font)
    
    plt.rcParams['text.usetex'] = True

    spacing = 10 #Time spacing
    
    fig, ((ax1, ax2)) = plt.subplots(nrows=1, ncols=2, sharey=all, sharex=all, figsize=(hsize, vsize))
    for k in range(0,nt,spacing):
        rossby_ini = rossby_ini_data[k,:,:]
        rossby_eii = rossby_eii_data[k,:,:]
        rossby_ini = rossby_ini.flatten()
        rossby_eii = rossby_eii.flatten()
        #Plot!
        label_t = label_time(t[k])
        ax1.hist(rossby_ini, bins=14, range=(-1.5,-0.1), histtype='step', color=colors[int(k/spacing)], linewidth=1.5, alpha=1, label=label_t)
        ax2.hist(rossby_eii, bins=14, range=(-1.5,-0.1), histtype='step', color=colors[int(k/spacing)], linewidth=1.5, alpha=1, label=label_t)
        ax1.hist(rossby_ini, bins=14, range=(0.1,1.5), histtype='step', color=colors[int(k/spacing)], linewidth=1.5, alpha=1, label=label_t)
        ax2.hist(rossby_eii, bins=14, range=(0.1,1.5), histtype='step', color=colors[int(k/spacing)], linewidth=1.5, alpha=1, label=label_t)
        ax1.set_ylabel('Counts')
        ax1.set_xlabel(r'$\mathrm{Ro}$', fontsize=fs)
        ax1.set_title('a) Inertial Instability', fontsize=fs, linespacing = 1.5)
        ax2.set_xlabel(r'$\mathrm{Ro}$', fontsize=fs)
        ax2.set_title('b) Ekman-Inertial Instability', fontsize=fs, linespacing = 1.5)
        fig.subplots_adjust(bottom=0.27, top=0.9, left=0.095, right=0.95, wspace=0.15, hspace=0.5)
        #plt.xticks(np.arange(-1.5,2,0.5), fontsize=fs)
        cb_ax = fig.add_axes([0.07, 0.12, 0.88, 0.03]) #Left extreme, height level, right extreme, width ofcb
        cbar = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0,20), cmap='viridis'), ticks=[0, 5, 10, 15, 20], cax=cb_ax, orientation='horizontal', extend='max')
        cbar.set_label('$t/2\pi$', fontsize=fs,labelpad=7)
        cbar.ax.tick_params(labelsize=fs)
    fig.savefig(output.joinpath('rossby-histogram-together.png'), dpi=300)

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
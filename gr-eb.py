"""
Script to compute growth rate and energy budget for a given experiment

Usage:
    gr-eb.py <in-exp> <files>... [--output=<dir>]

Options:
    --output=<dir>  Output directory [default: ./frames]

"""


import h5py
import numpy as np
from dedalus import public as de
from matplotlib import use
from numpy import load
from docopt import docopt
from numpy import save

#use('Agg')
#plt.ioff()

#Parameters
Lx, H = 16, 1
nx, nz = 2048, 128
nt = 200 #Simulation outputs
Tf = 2*np.pi #Inertial periods

def main(filename, start, count, output):
    """ Compute growth rate for a given instability """
    
    args = docopt(__doc__)
    
    in_exp = str(args['<in-exp>'])

    with h5py.File(filename, mode='r') as file:
        
        #Growth rate
        t = np.array(file['scales']['sim_time'])
        perturbations = np.squeeze(np.array(file['tasks']['w2-int']))
                
        growth_rate_perturbations = np.empty([nt-1])
        growth_rate_perturbations[0] = (1/perturbations[1])*(np.diff(perturbations))[0]/(np.diff(t))[0]
        growth_rate_perturbations[1:] = (1/(0.5*(perturbations[1:-1]+perturbations[2:])))*(np.diff(perturbations))[1:]/(np.diff(t))[1:]
        growth_rate_perturbations = (1/2)*growth_rate_perturbations #Add a factor of 1/2 because of w^2
        
        print('Growth Rate perturbations:')
        print(growth_rate_perturbations)
        
        print('Max. growth rate perturbations:')
        print(np.max(growth_rate_perturbations))
        
        save("growth-rate-" + in_exp + '.npy', growth_rate_perturbations)

        #Energy Budget
        advection_x = np.squeeze(np.array(file['tasks']['advection-x-int']))
        advection_z = np.squeeze(np.array(file['tasks']['advection-z-int']))
        pressure_x = np.squeeze(np.array(file['tasks']['pressure-x-int']))
        pressure_z = np.squeeze(np.array(file['tasks']['pressure-z-int']))
        diffusion = np.squeeze(np.array(file['tasks']['diffusion-int']))
        dissipation = np.squeeze(np.array(file['tasks']['dissipation-int']))

        advection = advection_x + advection_z
        pressure = pressure_x + pressure_z
        
        KE_int = np.squeeze(np.array(file['tasks']['KE-int']))
        
        LHS = np.diff(KE_int)/np.diff(t)
        RHS = -advection - pressure + diffusion - dissipation
        
        save('lhs-eb-' + in_exp + '.npy', LHS)
        save('advection-' + in_exp + '.npy', advection)
        save('pressure-' + in_exp + '.npy', pressure)
        save('diffusion-' + in_exp + '.npy', diffusion)
        save('dissipation-' + in_exp + '.npy', dissipation)

        #PLOT!
        #fig, ax = plt.subplots()
        #ax.plot(t[:-1]/Tf, LHS, label = 'LHS', color='firebrick', marker='*')
        #ax.plot(t/Tf, RHS, label = 'RHS', color='darkblue', marker='v')
        #ax.plot(t/Tf, -advection, label = 'advection', color='dodgerblue')
        #ax.plot(t/Tf, -pressure, label = 'pressure', color='blueviolet')
        #ax.plot(t/Tf, diffusion, label = 'diffusion', color='red')
        #ax.plot(t/Tf, -dissipation, label = 'dissipation', color='darkorange')
        #ax.ticklabel_format(scilimits=(0,3))
        #ax.set_xlabel('t/2$\pi$')
        #ax.set_ylabel('Energy Budget')
        #plt.legend()
        #plt.tight_layout()
        #Save figure
        #fig.savefig(output.joinpath('energy-budget-' + in_exp + '.png'), dpi=100) 
        #plt.close(fig)
        

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
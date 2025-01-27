"""
Script to remove near-inertial oscillations

Usage:
    near-inertial-oscillations.py <in-exp> <field> <files>... [--output=<dir>]

Options:
    --output=<dir>  Output directory [default: ./frames]

"""

import matplotlib.pyplot as plt
import h5py
import numpy as np
from dedalus import public as de
from scipy.interpolate import interpn
from matplotlib import use
from numpy import save
from docopt import docopt


#use('Agg')
#plt.ioff()

#Parameters
Lx, H = 16, 1
nx, nz = 2048, 128
nt = 200 #Simulation outputs
Tf = 2*np.pi #Inertial periods

def main(filename, output):
    """ Remove near-inertial oscillations """

    args = docopt(__doc__)
    
    in_exp = str(args['<in-exp>']) #e.g., InI with Ro = -1.1: ini-11
    field_str = str(args['<field>']) #e.g., w
    
    with h5py.File(filename, mode='r') as file:
              
        t = np.array(file['scales']['sim_time'])
        xA = np.squeeze(np.array(file['tasks']['xA'])) #Mean along-x displacement.       
        field = np.squeeze(np.array(file['tasks'][field_str])) #Field to remove oscillations from
        
        x_basis = de.Fourier('x', nx, interval=(-Lx/2, Lx/2), dealias=3/2)
        z_basis = de.Chebyshev('z', nz, interval=(-H, 0.), dealias=3/2)
        domain = de.Domain([x_basis, z_basis], grid_dtype=np.float64)
        x, z = domain.grids(scales=1)

        T,X,Z = np.meshgrid(t,x,z, indexing='ij')
        
        #STEP 1: Shift frame of reference
        X_new = X - xA
        #Extend "X_new" & "field" by "n" in the x-direction
        n = 500
        X_extra_left=np.empty([nt,n,nz])
        X_extra_right=np.empty([nt,n,nz])
        print('Extending arrays by n=500')
        for i in range(n):
            X_extra_left[:,i,:] = X_new[:,-n+i,:] - Lx
            X_extra_right[:,i,:] = X_new[:,i,:] + Lx
            print('i = ' + '{:.2f}'.format(i))
        X_new_extended = np.append(X_new,X_extra_right, axis=1)
        X_new_extended = np.insert(X_new_extended, [0], X_extra_left, axis=1)
        field_extended = np.append(field,field[:,0:n,:], axis=1)
        field_extended = np.insert(field_extended, [0], field[:,-n:,:], axis=1)
        
        #STEP 2: Interpolate to new positions
        field_shifted = np.empty([nt,nx,nz])
        print('Interpolating')
        for k in range(nt):
            points = (X_new_extended[k,:,0], Z[k,0,:]) #Where we have data
            values = field_extended[k,:,:]
            points_int = np.array([[X[k,i,0], Z[k,0,j]] for i in range(nx) for j in range(nz)]) #Where we want data
            field_shifted[k,:,:] = interpn(points, values, points_int, method='linear', bounds_error=True).reshape((nx,nz))
            print('k = ' + '{:.2f}'.format(k))    

        #STEP 3: Remove mean field that includes near-inertial oscillations
        field_initial = np.tile(field_shifted[0,:,:], (nt,1,1)) #Original field, including the jet, in new frame of reference
        field_shifted_no_initial = field_shifted - field_initial #Field with no jet, only near-inertial oscillations + fluctuations
        field_mean = np.mean(field_shifted_no_initial[:,-1,:],axis=1) #Mean field, including near-inertial oscillations
        field_mean_tile = np.transpose(np.tile(field_mean, (nx,nz,1)), [2,0,1]) #Make arrays fit
        field_fluctuations = field_shifted - field_mean_tile #Fluctuations from mean field
        
        #STEP 4: Save data!
        save(field_str + '-shifted-' + in_exp + '.npy', field_shifted)
        save(field_str + '-mean-' + in_exp + '.npy', field_mean_tile)
        save(field_str + '-fluctuations-' + in_exp + '.npy', field_fluctuations)
    
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

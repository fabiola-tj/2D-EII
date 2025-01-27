"""
Dedalus script simulating either EII or InI.

"""

import sys
import time
import numpy as np

from mpi4py import MPI
from dedalus import public as de
from dedalus.extras import flow_tools

import logging
logger = logging.getLogger(__name__)

#Command line arguments
instability = sys.argv[1] #Options: eii OR ini
Ro_number = sys.argv[2]   #Rossby number
Ro_id = Ro_number.replace(".","").replace("-","") #Ro identifier

# Parameters
Lx, H = 16, 1 #Domain dimensions
nx, nz = 2048, 128 #Grid points
Ro = float(Ro_number)
alpha = 1 #Aspect ratio alpha=H/L L: Half-width of the current
viscosity_ratio = 1 #nux/nuz
Ekz = 1e-3 #Ekman number Ekz = delta**2/H**2
Ekx = viscosity_ratio*Ekz*(alpha**2) #Ekman number (horizontal)
v0 = -Ro*np.exp(1/2)
#v0 = vel_int*Rom/(2*np.pi)
tau = alpha #tau = alpha*T/(rho*nuz) T:wind stress

# Time stepping.
dt = 1e-3
Tf = 2*np.pi #Inertial period
t_end = 20*Tf

# Create bases and domain
x_basis = de.Fourier('x', nx, interval=(-Lx/2, Lx/2), dealias=3/2)
z_basis = de.Chebyshev('z', nz, interval=(-1, 0.), dealias=3/2)
domain = de.Domain([x_basis, z_basis], grid_dtype=np.float64)
x, z = domain.grids(scales=1)
X, Z = np.meshgrid(x, z)

# Initial Conditions
v_jet = -v0*np.exp(-(x**2)/2) #Velocity of the initial jet

ICv = domain.new_field()  # initial v
ICvz = domain.new_field()  # initial dv/dz

ICv['g'] = v_jet
ICv.differentiate('z', out=ICvz)

# Problem: 2D Boussinesq Hydrodynamics
problem = de.IVP(domain, variables = ['u', 'v', 'w', 'uz', 'vz', 'wz', 'p', 'xA'], time = 't')
problem.meta['u', 'v', 'wz','p']['z']['dirichlet'] = False

problem.parameters['alpha'] = alpha
problem.parameters['Lx'] = Lx
problem.parameters['H'] = H
problem.parameters['Ekz'] = Ekz
problem.parameters['Ekx'] = Ekx
problem.parameters['tau'] = tau
problem.parameters['ICvz'] = ICvz

# Governing equations
problem.substitutions['advection(A, Az)'] = "- u*dx(A) - w*Az"
problem.substitutions['diffusion(A, Az)'] = "- Ekx*d(A, x=2) - Ekz*dz(Az)"

problem.add_equation("dx(u) + wz = 0")
problem.add_equation("dt(u) - v + dx(p) + diffusion(u, uz) = advection(u, uz)")
problem.add_equation("dt(v) + u + diffusion(v, vz) = advection(v, vz)")
problem.add_equation("dt(w) + (1/alpha**2)*dz(p) + diffusion(w, wz) = advection(w, wz)")
problem.add_equation("uz - dz(u) = 0")
problem.add_equation("vz - dz(v) = 0")
problem.add_equation("wz - dz(w) = 0")
problem.add_equation("dt(xA) = integ(u)/(Lx*H)")

# Boundary conditions in z
problem.add_bc("left(uz) = 0")
problem.add_bc("right(uz) = 0")
problem.add_bc("left(w) = 0")
problem.add_bc("right(w) = 0", condition="(nx != 0)")
problem.add_bc("right(p) = 0", condition="(nx == 0)")

if instability == "eii":
        problem.add_bc("left(vz) = left(ICvz)")
        problem.add_bc("right(vz) = tau")
elif instability == "ini":  # free slip for v
    problem.add_bc("left(vz) = 0")
    problem.add_bc("right(vz) = 0")
else:
    raise NameError('Boundary conditions for v not recognized')


# Build solver ---------------------------------------------------------------|
solver = problem.build_solver(de.timesteppers.RK222)
logger.info('Solver built')

# Initial conditions for v -----------------------------------------------|
IC_v = solver.state['v']  # initial condition for v
IC_vz = solver.state['vz']  # maybe we can get rid of this

# Random perturbations, initialized globally for same results in parallel
gshape = domain.dist.grid_layout.global_shape(scales=1)
slices = domain.dist.grid_layout.slices(scales=1)
rand = np.random.RandomState(seed=42)
noise = rand.standard_normal(gshape)[slices]

# Linear background + perturbations damped at walls
zb, zt = z_basis.interval
pert =  1e-9 * noise * (zt - z) * (z - zb)

# Initial velocity
IC_v['g'] = v_jet + pert
IC_v.differentiate('z', out=IC_vz)

# Integration parameters
solver.stop_sim_time = t_end
solver.stop_wall_time = np.inf
solver.stop_iteration = np.inf

# Analysis
diagnostics = solver.evaluator.add_file_handler('diagnostics'+'-'+instability+'-'+Ro_id, sim_dt=Tf/10, mode='append')
diagnostics.add_task("xA", scales=1, name='xA')
diagnostics.add_task("u", scales=1, name='u')
diagnostics.add_task("v", scales=1, name='v')
diagnostics.add_task("w", scales=1, name='w')
diagnostics.add_task("p", scales=1, name='p')
diagnostics.add_task("0.5*(u**2 + v**2 + w**2)", scales=1, name='KE')
diagnostics.add_task("d(u, x=1)", scales=1, name='du_dx')
diagnostics.add_task("d(v, x=1)", scales=1, name='dv_dx')
diagnostics.add_task("d(w, x=1)", scales=1, name='dw_dx')
diagnostics.add_task("d(p, x=1)", scales=1, name='dp_dx')
diagnostics.add_task("d(p, z=1)", scales=1, name='dp_dz')
diagnostics.add_task("uz", scales=1, name='du_dz')
diagnostics.add_task("vz", scales=1, name='dv_dz')
diagnostics.add_task("wz", scales=1, name='dw_dz')
diagnostics.add_task("d(0.5*(u**2 + v**2 + w**2), x=1)", scales=1, name='dKE_dx')
diagnostics.add_task("d(0.5*(u**2 + v**2 + w**2), z=1)", scales=1, name='dKE_dz')
diagnostics.add_task("d(0.5*(u**2 + v**2 + w**2), x=2)", scales=1, name='d2KE_dx')
diagnostics.add_task("d(0.5*(u**2 + v**2 + w**2), z=2)", scales=1, name='d2KE_dz')
diagnostics.add_task("integ(0.5*(u**2 + v**2 + w**2))", scales=1, name='KE-int')
diagnostics.add_task("integ(w**2)", scales=1, name='w2-int')
diagnostics.add_task("integ(u*d(0.5*(u**2 + v**2 + w**2), x=1))", scales=1, name='advection-x-int')
diagnostics.add_task("integ(w*d(0.5*(u**2 + v**2 + w**2), z=1))", scales=1, name='advection-z-int')
diagnostics.add_task("integ(u*d(p, x=1))", scales=1, name='pressure-x-int')
diagnostics.add_task("integ(w*d(p, z=1))", scales=1, name='pressure-z-int')
diagnostics.add_task("integ(Ekz*(d(0.5*(u**2 + v**2 + w**2), x=2) + d(0.5*(u**2 + v**2 + w**2), z=2))", scales=1, name='diffusion-int')
diagnostics.add_task("Ekz*(d(0.5*(u**2 + v**2 + w**2), x=2) + d(0.5*(u**2 + v**2 + w**2), z=2))", scales=1, name='diffusion')
diagnostics.add_task("integ(Ekz*((dx(u))**2 + (dx(v))**2 + (dx(w))**2 + (dz(u))**2 + (dz(v))**2 + (dz(w))**2))", scales=1, name='dissipation-int')
diagnostics.add_task("Ekz*((dx(u))**2 + (dx(v))**2 + (dx(w))**2 + (dz(u))**2 + (dz(v))**2 + (dz(w))**2)", scales=1, name='dissipation')


# CFL
CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=10, safety=0.8, max_change=1.5, threshold=0.05, max_dt=500.)
CFL.add_velocities(('u', 'w'))

# Main loop
try:
    logger.info('Starting loop')
    start_run_time = time.time()
    while solver.ok:
        dt = solver.step(dt)
        if (solver.iteration-1) % 100 == 0:
            logger.info('Iteration: {0:d}, Time: {1:e}, dt: {2:e}'.format(solver.iteration, solver.sim_time, dt))
except NameError:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_run_time = time.time()
    logger.info('Iterations: {0:d}'.format(solver.iteration))
    logger.info('Sim end time: {0:f}'.format(solver.sim_time))
    logger.info('Run time: {0:.2f} sec'.format(end_run_time-start_run_time))
    logger.info('Run time: {0:f} cpu-hr'.format((end_run_time-start_run_time)/3600 * domain.dist.comm_cart.size))
    logger.info('')
    if end_run_time > t_end*0.99:
        logger.info('|-+-+-END-OF-THE-SIMULATION-+-+-|')
    else:
        logger.info('|-+-+-SIMULATION-SHOULD-CONTINUE-+-+-|')

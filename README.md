PetscOpenCL
===========

Demonstration of implicit time stepping with Petsc along with OpenCL to perform residual evaluation

This code simply solves a set of decoupled hyperbolic PDEs in 2D:

dU1/dt + grad(U1) = 0

dU2/dt + grad(U2) = 0
       .
       .
       .
       .

The number of equations that one wants to solve (DOF) as well as the resolution
(N1, N2) can be set in constants.h

To run the code do the following:

time ./petsc_opencl -ts_monitor -ts_type theta -ts_theta_theta 0.5 -ts_dt 0.005 -snes_monitor -ts_max_snes_failures -1

For a full list of options do
./petsc_opencl -help

The code also demonstrates efficient use of memory hierachy in OpenCL and
provides diagnostics of all the memory usage in an OpenCL kernel.

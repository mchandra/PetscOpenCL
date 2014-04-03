#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <cmath>
#include <ctime>
#include <CL/cl.hpp>
#include <petsc.h>
#include "constants.h"

static const char help[] = 
    "Prototype Hyperbolic evolution code to demonstrate the usage of Petsc and OpenCL";
extern void InitialCondition(TS ts, Vec prim);
extern PetscErrorCode ComputeResidual(TS ts,
                               PetscScalar t,
                               Vec Prim, Vec dPrim_dt,
                               Vec F, void *ptr);

cl_int clErr;
std::vector<cl::Platform> platforms;
std::vector<cl::Device> devices;
cl::Context context;
cl::CommandQueue queue;
cl::Program program;
cl::Kernel kernel;

PetscErrorCode CheckCLErrors(cl_int clErr, std::string errMsg)
{
    if (clErr!=CL_SUCCESS) {
        SETERRQ(PETSC_COMM_SELF, 1, errMsg.c_str());
    }
    return(0);
}

int main(int argc, char **argv)
{
    TS ts; //Time stepper
    Vec soln; //Holds the solution vector, including all the primitive
              //variables. 
    DM dmda; //Manages the computational grid and parallelization.

    int X1Start, X2Start;
    int X1Size, X2Size;

    PetscInitialize(&argc, &argv, PETSC_NULL, help);

    // Create the computational domain.
    DMDACreate2d(PETSC_COMM_WORLD, 
                 DM_BOUNDARY_GHOSTED, DM_BOUNDARY_GHOSTED,
                 DMDA_STENCIL_STAR,
                 N1, N2,
                 PETSC_DECIDE, PETSC_DECIDE,
                 DOF, NG, PETSC_NULL, PETSC_NULL, &dmda);

    // When running in parallel, each process computes from
    // [X1Start, X1Start+X1Size] x [X2Start, X2Start+X2Size]
    DMDAGetCorners(dmda, 
                   &X1Start, &X2Start, NULL,
                   &X1Size, &X2Size, NULL);

    // Create the solution vector.
    DMCreateGlobalVector(dmda, &soln);

    // Create the time stepper and link it to the computational grid and the
    // residual evaluation function.
    TSCreate(PETSC_COMM_WORLD, &ts);
    TSSetDM(ts, dmda);
    TSSetIFunction(ts, PETSC_NULL, ComputeResidual, NULL);

    // OpenCL boilerplate code.
    clErr = cl::Platform::get(&platforms);
    CheckCLErrors(clErr, "cl::Platform::get");

    // Select computation device here.
    clErr = platforms.at(1).getDevices(CL_DEVICE_TYPE_CPU, &devices);
    CheckCLErrors(clErr, "cl::Platform::getDevices");

    context = cl::Context(devices, NULL, NULL, NULL, &clErr);
    CheckCLErrors(clErr, "cl::Context::Context");

    queue = cl::CommandQueue(context, devices.at(0), 0, &clErr);
    CheckCLErrors(clErr, "cl::CommandQueue::CommandQueue");

    std::ifstream sourceFile("computeresidual.cl");
    std::string sourceCode((std::istreambuf_iterator<char>(sourceFile)),
                            std::istreambuf_iterator<char>());
    cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(),
                                sourceCode.length()+1));
    
    program = cl::Program(context, source, &clErr);
    CheckCLErrors(clErr, "cl::Program::Program");

    // Pass in constants to the OpenCL kernel as compiler switches. This is an
    // efficient way to handle constants such as domain sizes in OpenCL.
    std::string BuildOptions("\
                              -D X1_SIZE=" +
                             std::to_string(X1Size) +
                             " -D X2_SIZE=" + 
                             std::to_string(X2Size) +
                             " -D TOTAL_X1_SIZE=" + 
                             std::to_string(X1Size+2*NG) + 
                             " -D TOTAL_X2_SIZE=" +
                             std::to_string(X2Size+2*NG));

    // Compile the OpenCL program and extract the kernel.
    PetscScalar start = std::clock();
    clErr = program.build(devices, BuildOptions.c_str(), NULL, NULL);
    const char *buildlog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(
                                                devices.at(0),
                                                &clErr).c_str();
    PetscPrintf(PETSC_COMM_WORLD, "%s\n", buildlog);
    CheckCLErrors(clErr, "cl::Program::build");
    PetscScalar end = std::clock();

    PetscScalar time = (end - start)/(PetscScalar)CLOCKS_PER_SEC;
    PetscPrintf(PETSC_COMM_WORLD, 
                "Time taken for kernel compilation = %f\n", time);


    kernel = cl::Kernel(program, "ComputeResidual", &clErr);
    CheckCLErrors(clErr, "cl::Kernel::Kernel");

    // How much memory is the kernel using?
    cl_ulong localMemSize = kernel.getWorkGroupInfo<CL_KERNEL_LOCAL_MEM_SIZE>(
                                        devices.at(0), &clErr);
    cl_ulong privateMemSize = kernel.getWorkGroupInfo<CL_KERNEL_PRIVATE_MEM_SIZE>(
                                        devices.at(0), &clErr);
    printf("Local memory used = %llu\n", (unsigned long long)localMemSize);
    printf("Private memory used = %llu\n", (unsigned long long)privateMemSize);


    // Set initial conditions.
    InitialCondition(ts, soln);

    TSSetSolution(ts, soln);
    TSSetType(ts, TSTHETA);
    TSSetFromOptions(ts);

    // Finally solve! All time stepping options can be controlled from the
    // command line.
    TSSolve(ts, soln);

    // Delete the data structures in the following order.
    DMDestroy(&dmda);
    VecDestroy(&soln);
    TSDestroy(&ts);

    PetscFinalize();
    return(0);
}

PetscErrorCode ComputeResidual(TS ts,
                               PetscScalar t,
                               Vec Prim, Vec dPrim_dt,
                               Vec F, void *ptr)
{
    PetscErrorCode ierr;
    PetscScalar *prim, *dprim_dt, *f;

    // Get pointers to Petsc Vecs so that we can access the data.
    ierr = VecGetArray(Prim, &prim); CHKERRQ(ierr);
    ierr = VecGetArray(dPrim_dt, &dprim_dt); CHKERRQ(ierr);
    ierr = VecGetArray(F, &f); CHKERRQ(ierr);
    
    // OpenCL buffers.
    cl::Buffer primBuffer, dprimBuffer_dt, fbuffer;
    PetscInt size = DOF*N1*N2*sizeof(PetscScalar);

    // Create OpenCL buffers from the data pointers to Petsc Vecs.
    primBuffer = cl::Buffer(context,
                            CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                            size, &(prim[0]), &clErr);
    dprimBuffer_dt = cl::Buffer(context,
                                CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                size, &(dprim_dt[0]), &clErr);
    fbuffer = cl::Buffer(context,
                         CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY,
                         size, &(f[0]), &clErr);


    // Set kernel args.
    clErr = kernel.setArg(0, primBuffer);
    clErr = kernel.setArg(1, dprimBuffer_dt);
    clErr = kernel.setArg(2, fbuffer);

    // Kernel launch parameters and execution.
    cl::NDRange global(N1, N2);
    cl::NDRange local(TILE_SIZE_X1, TILE_SIZE_X2);
    clErr = queue.enqueueNDRangeKernel(kernel,
                                       cl::NullRange,
                                       global, local,
                                       NULL, NULL);

    // The following "buffer mapping" is not needed if running on CPU but is
    // needed if the OpenCL device executing the kernel is a GPU in order to
    // sync the data. For CPUs this routine is zero cost when used with buffers
    // created using CL_MEM_USE_HOST_PTR like we did above. For GPUs, the GPU
    // will access the data on the RAM as and when needed automatically without
    // user intervention.
    f = (PetscScalar*)queue.enqueueMapBuffer(fbuffer,
                                             CL_FALSE,
                                             CL_MAP_READ,
                                             0, size,
                                             NULL, NULL, &clErr);

    // Global sync point for all the threads to ensure execution is complete.
    clErr = queue.finish();

    // Restore the pointers.
    ierr = VecRestoreArray(Prim, &prim); CHKERRQ(ierr);
    ierr = VecRestoreArray(dPrim_dt, &dprim_dt); CHKERRQ(ierr);
    ierr = VecRestoreArray(F, &f); CHKERRQ(ierr);

    return(0);
}

void InitialCondition(TS ts, Vec X)
{
    PetscScalar *x;

    VecGetArray(X, &x);

    for (PetscInt j=0; j<N2; j++) {
        for (PetscInt i=0; i<N1; i++) {
            for (PetscInt var=0; var<DOF; var++) {
            
                PetscScalar X1Coord = X1_MIN + DX1/2. + i*DX1;
                PetscScalar X2Coord = X2_MIN + DX2/2. + j*DX2;

                PetscScalar X1Center = (X1_MIN + X1_MAX)/2.;
                PetscScalar X2Center = (X2_MIN + X2_MAX)/2.;

                PetscScalar r = PetscSqrtScalar(
                                    PetscPowScalar(X1Coord-X1Center, 2.0) + 
                                    PetscPowScalar(X2Coord-X2Center, 2.0));

                x[INDEX_GLOBAL(i,j,var)] = exp(-r*r/.01);
            }
        }
    }

    VecRestoreArray(X, &x);
}

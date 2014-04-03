ALL: petsc_opencl

PETSC_DIR=/home/mc/Downloads/petsc_optimized

CFLAGS = -lOpenCL -std=c++0x -O3

FFLAGS =

CPPFLAGS = -lOpenCL -std=c++0x -O3

FPPFLAGS =

include ${PETSC_DIR}/conf/variables

include ${PETSC_DIR}/conf/rules

petsc_opencl: petsc_opencl.o
	-${CLINKER} -o petsc_opencl petsc_opencl.o ${PETSC_LIB}
	${RM} petsc_opencl.o

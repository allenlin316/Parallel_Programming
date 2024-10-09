
# Calculate number of pixels of a circle on a 2D monitor
* reference: NTU Prof. Chun-Yi Lee lecture

## Input & Output
* Suppose we want to draw a filled circle of radius r on a 2D monitor, how many pixels will be filled?
We fill a pixel when any part of the circle overlaps with the pixel. We also assume that the circle center is at the boundary of 4 pixels.
For example 88 pixels are filled when r=5.


## System environment (on Twnia3 supercomputer)
* `module load gcc/13`
* `module load openmpi`

## Compile & Execute
Compile  
* compile a C program with MPI: `mpicc path/to/source.c`
* compile a C++ program with MPI: `mpicxx path/to/source.cpp`   

Execute  
* Run 5 MPI processes: `srun -n 5 -A <plan> path/to/program`
* Run 5 MPI processes, giving each process 4 CPUs: `srun -n 5 -c 4 -A <plan> path/to/program`   

I/O Format
* Input foramt: `srun -n 5 ./lab2 r k`
    * r: integer, radius of the circle
    * k: integer (to make it smaller for mod operation)

# Adaptive Filtering
* reference: NTU Prof. Chun-Yi Lee lecture

## Goal
* Implement a program to optimize the adaptive filter for removing noise in images
* Parallelize the program with Pthreads or OpenMP
* C++:std::thread library is prohibited

## Input & Output
* input: noisy images of varied sizes
* output: the corresponding denoised images

## Compile & Execute (run on Twnia3 supercomputer)
* compile
    * `module load gcc/13.2.0`
    * `g++ -std=c++11 -O3 -lpng -lpthread -fopenmp hw1.cpp -o hw1`
* execute
    * `srun -A <plan> -n 1 -c 8 ./hw1 path/to/input.png path/to/output.png`

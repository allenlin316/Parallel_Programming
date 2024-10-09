#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char** argv) {
	double start = MPI_Wtime();
    MPI_Init(&argc, &argv);  // Initialize MPI
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // Get the rank of the process
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);  // Get the total number of processes

    if (argc != 3) {
        if (rank == 0) {
            fprintf(stderr, "must provide exactly 2 arguments!\n");
        }
        MPI_Finalize();
        return 1;
    }

    unsigned long long r = atoll(argv[1]);  // Radius
    unsigned long long k = atoll(argv[2]);  // Modulus
    unsigned long long local_pixels = 0;    // Each process's local sum of pixels

    // Determine the range of x values for each process
    unsigned long long chunk_size = r / world_size;
    unsigned long long start_x = rank * chunk_size;
    unsigned long long end_x = (rank == world_size - 1) ? r : start_x + chunk_size;

    // Each process computes its portion of the sum
    for (unsigned long long x = start_x; x < end_x; x++) {
        unsigned long long y = ceil(sqrtl(r * r - x * x));
        local_pixels += y;
        local_pixels %= k;  // Keep the result modulo k to avoid overflow
    }

    // Reduce all local sums into the global sum (sum all values from all processes)
    unsigned long long global_pixels = 0;
    MPI_Reduce(&local_pixels, &global_pixels, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    // Only rank 0 will print the result
    if (rank == 0) {
        printf("%llu\n", (4 * global_pixels) % k);
    }

    MPI_Finalize();  // Finalize MPI
	double end = MPI_Wtime();
	//printf("%d sec", (end-start)*1000);
    //printf("%llu\n", (4 * pixels) % k);
    return 0;
}
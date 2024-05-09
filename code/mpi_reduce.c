#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>

#define ARRAY_SIZE 10

int main(int argc, char** argv) {
    int rank, size;
    int local_array[ARRAY_SIZE]; // Local array on each process
    int global_array[ARRAY_SIZE]; // Final result array after reduction

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Initialize local_array with some values
    for (int i = 0; i < ARRAY_SIZE; i++) {
        local_array[i] = rank * ARRAY_SIZE + i;
    }

    // Perform the reduction operation on the array
    MPI_Reduce(local_array, global_array, ARRAY_SIZE, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // Print the global array on the root process
    if (rank == 0) {
        printf("Global array: ");
        for (int i = 0; i < ARRAY_SIZE; i++) {
            printf("%d ", global_array[i]);
        }
        printf("\n");
    }

    MPI_Finalize();
    return 0;
}
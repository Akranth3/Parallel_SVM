#include <stdio.h>
#include <mpi.h>


int main(int argc, char** argv) {
    int rank, size;
    int data[100][10]; // 2D matrix to be scattered
    int local_data[25][10]; // Local portion of the matrix for each process

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Initialize the data matrix on the root process (rank 0)
    if (rank == 0) {
        for (int i = 0; i < 100; i++) {
            for (int j = 0; j < 10; j++) {
                data[i][j] = i * 10 + j;
            }
        }
    }
    printf("data reading kinda done, rank %d\n", rank);

    // // Create a custom data type for a row of the local matrix
    // MPI_Datatype row_type;
    // MPI_Type_vector(5, 1, 10, MPI_INT, &row_type);
    // MPI_Type_commit(&row_type);

    // // Create a custom data type for the local matrix
    // MPI_Datatype local_matrix_type;
    // MPI_Type_vector(20, 5, 10 * sizeof(int), row_type, &local_matrix_type);
    // MPI_Type_commit(&local_matrix_type);

    // printf("ok?\n");

    // Scatter the matrix from the root process to all processes
    MPI_Scatter(data, 250, MPI_INT, local_data, 250, MPI_INT, 0, MPI_COMM_WORLD);

    // Each process can now work on its local portion of the matrix
    // ...
    printf("not ok?\n");
    // Clean up
    // MPI_Type_free(&row_type);
    // MPI_Type_free(&local_matrix_type);
    if(rank==3){
        for (int i = 0; i < 25; i++) {
            for (int j = 0; j < 10; j++) {
                printf("%d ", local_data[i][j]);
            }
            printf("\n");
        }
    }
    MPI_Finalize();
    


    return 0;
}
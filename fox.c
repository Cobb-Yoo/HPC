
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include "mpi.h"
#include "lib.c"

#define N 24

float **malloc_2d(); // 2d malloc
void decomp(); // 2d decomposition
void comp(); // // 2d composition
void init(); // initialize with 0.0
void mmult(); // matrix multiplication
void check(); // check the results

main(int argc, char* argv[])
{
   float A[N][N], B[N][N], C[N][N];
   float **local_A, **local_A2, **local_B, **local_C;
   int np2, np, pid, source, dest, local_N, local_N2;
   int direct, shift, i, j;
   MPI_Comm grid_comm, row_comm;
   int dim_sizes[2], wrap_around[2], coord[2], reorder;
   MPI_Status status;
   int tag;

   direct = 0;
   shift = -1;

   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &pid);
   MPI_Comm_size(MPI_COMM_WORLD, &np2);

   np = sqrt(np2);
   local_N = N/np;
   local_N2 = local_N*local_N;

   local_A = malloc_2d(local_N, local_N);
   local_A2 = malloc_2d(local_N, local_N);

   local_B = malloc_2d(local_N, local_N);
   local_C = malloc_2d(local_N, local_N);

   if (pid == 0)
      for (i=0; i<N; i++)
         for (j=0; j<N; j++) {
            A[i][j] = (i*N+j)*0.001;
            B[i][j] = (N*N-i*N-j)*0.001;
         }


   dim_sizes[0] = dim_sizes[1] = np;
   wrap_around[0] = wrap_around[1] = 1;

   decomp(N, local_N, np, A, local_A);
   decomp(N, local_N, np, B, local_B);
   copy(local_N, local_A, local_A2); // backup local_A

   MPI_Cart_create(MPI_COMM_WORLD, 2, dim_sizes, wrap_around, reorder, &grid_comm);
   MPI_Cart_coords(grid_comm, pid, 2, coord);
   MPI_Cart_shift(grid_comm, direct, shift, &source, &dest);

   MPI_Comm_split(MPI_COMM_WORLD, coord[0], pid, &row_comm);

   init(local_N, local_C);

   for (i=0; i<np; i++) {
      copy(local_N, local_A2, local_A);
      MPI_Bcast(local_A[0], local_N2, MPI_FLOAT, (coord[0]+i)%np, row_comm);

      mmult(local_N, local_A, local_B, local_C);
      MPI_Sendrecv_replace(local_B[0], local_N2, MPI_FLOAT, dest, tag, source, tag, MPI_COMM_WORLD, &status);
   }

   comp(N, local_N, np, local_C, C);
   if (pid == 0) check(N, A, B, C);

   free(local_A); free(local_B); free(local_C);
   free(local_A2);

   MPI_Finalize();
}


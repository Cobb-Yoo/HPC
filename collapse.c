#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

#define N 12

main(int argc, char* argv[]){
   int np, pid, i, dest, source, tag = 0, local_N;
   float A[N];
   MPI_Status status;

   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &np);
   MPI_Comm_rank(MPI_COMM_WORLD, &pid);

   local_N = N/np;

   for (i = 0; i < local_N; i++)
      A[i] = pid*local_N+i;
   if (pid != 0)
      MPI_Send(A, local_N, MPI_FLOAT, 0, tag, MPI_COMM_WORLD);
   else
      for (i = 1; i < np; i++)
         MPI_Recv(A+i*local_N, local_N, MPI_FLOAT, i, tag, MPI_COMM_WORLD, &status);
   if (pid == 0) {
      for (i = 0; i < N; i++)
         printf("%f ", A[i]);
      printf("\n");
   }
   MPI_Finalize();
}


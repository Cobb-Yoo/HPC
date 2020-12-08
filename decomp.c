#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"

#define N 12

float **malloc_2d(int row, int col){
   float **A, *ptr;
   int len, i;

   len = sizeof(float *)*row + sizeof(float)*col*row;
   A = (float **)malloc(len);
   ptr = (float *)(A + row);
   for(i = 0; i < row; i++)
      A[i] = (ptr + col*i);
   return A;
}

main(int argc, char* argv[]){
   float A[N][N], **local_A;
   int local_N, i, j, np2, np, pid, source, dest, tag = 0;
   MPI_Datatype vector_t;
   MPI_Status status;

   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &pid);
   MPI_Comm_size(MPI_COMM_WORLD, &np2);

   np = sqrt(np2);
   local_N = N/np;

   local_A = malloc_2d(local_N, local_N);

   if (pid == 0) {
      for (i=0; i<N; i++) {
         for (j=0; j<N; j++) {
            A[i][j] = (i*N+j)*0.001;
            printf("%6.3f ", A[i][j]);
         }
         printf("\n");
      }
      printf("------------------------------------------------\n");
   }

   MPI_Type_vector(local_N,1,1,MPI_FLOAT, &vector_t);
   MPI_Type_commit(&vector_t);
   //(i) decompose A into local_A
   if(pid == 0){
      for(dest=1;dest<np2;dest++){
         for(i=0;i<local_N;i++){
            MPI_Send(&A[dest/np+i][dest%np],1,vector_t,dest,0,MPI_COMM_WORLD);
         }
      }
   }
   else {
      for(i=0;i<local_N;i++){
         MPI_Recv(&local_A[i][0],local_N,MPI_FLOAT,0,0,MPI_COMM_WORLD,&status);
      }
   }
   //(ii)
   for (i=0; i<local_N; i++)
      for (j=0; j<local_N; j++)
         local_A[i][j] += pid;


   //(ii) compose local_C to c
   if(pid == 0){
      for(source=1;source<np2;source++)
         for(i=0;i<local_N;i++)
            MPI_Recv(&A[(source/np)*local_N+i][(source%np)*local_N],1,vector_t,source,0,MPI_COMM_WORLD,&status);
   }
   else{
      for(i=0;i<local_N;i++)
         MPI_Send(&local_A[i][0],local_N,MPI_FLOAT,0,0,MPI_COMM_WORLD);
   }

   //check the result
   if (pid == 0)
      for (i=0; i<N; i++) {
         for (j=0; j<N; j++)
               printf("%6.3f ", A[i][j]);
         printf("\n");
      }

   free(local_A);
   MPI_Finalize();
}


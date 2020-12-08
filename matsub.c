#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

#define N 10

float **malloc_2d(int row, int col)
{
   float **A, *ptr;
   int len, i;

   len = sizeof(float *)*row + sizeof(float)*col*row;
   A = (float **)malloc(len);
   ptr = (float *)(A + row);
   for(i = 0; i < row; i++)
      A[i] = (ptr + col*i);
   return A;
}

main(int argc, char* argv[])
{
   float A[N][N], **local_A;
   int x, y, width, height, i, j, pid, tag = 0;
   MPI_Datatype vector_t;
   MPI_Status status;

   if (argc != 5) {
      printf("usage %s x y width, height\n", argv[0]);
      exit(1);
   }
   x = atoi(argv[1]); y = atoi(argv[2]);
   if (!(((x >= 0)&( x < N))&((y >= 0)&( y < N)))) {
      fprintf(stderr, "x y value not correct\n");
      exit(2);
   }

   width = atoi(argv[3]); width = (x+width)<N?width:N-x;
   height = atoi(argv[4]); height = (y+height)<N?height:N-y;
   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &pid);

   MPI_Type_vector(height,width,N,MPI_FLOAT,&vector_t);
   MPI_Type_commit(&vector_t);

   if (pid == 0) {
      for (i=0; i<N; i++) {
         for (j=0; j<N; j++) {
            A[i][j] = i*N+j;
            printf("%5.1f ", A[i][j]);
         }
         printf("\n");
      }

      MPI_Send(&A[y][x], 1, vector_t, 1, tag, MPI_COMM_WORLD);
   }

   if (pid == 1) {
      local_A = malloc_2d(height, width);

      MPI_Recv(&local_A[0][0], width*height, MPI_FLOAT, 0, tag, MPI_COMM_WORLD, &status);

      for (i=0; i<height; i++) {
         for (j=0; j<width; j++)
            printf("%5.1f ", local_A[i][j]);
         printf("\n");
      }
      free(local_A);
   }

   MPI_Finalize();
}


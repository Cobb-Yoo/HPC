#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

// 2d memory allocation
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

// 2d block decomposition
void decomp(int N, int local_N, int np, float A[N][N], float **local_A)
{
   int *displs, *counts, i, j, n;

   displs = (int*)malloc(sizeof(int)*np*np);
   counts = (int*)malloc(sizeof(int)*np*np);

   n = 0;
   for (i=0; i<np; i++) {
      for (j=0; j<np; j++) {
         displs[n] = i*local_N*N+j*local_N;
         counts[n++] = local_N;
      }
   }

   for (i=0; i<local_N; i++)
      MPI_Scatterv(&A[i][0], counts, displs, MPI_FLOAT, &local_A[i][0],
                   local_N, MPI_FLOAT, 0, MPI_COMM_WORLD);

   free(displs); free(counts);
}

// 2d block composition
void comp(int N, int local_N, int np, float **local_A, float A[N][N])
{
   int *displs, *counts, i, j, n;

   displs = (int*)malloc(sizeof(int)*np*np);
   counts = (int*)malloc(sizeof(int)*np*np);

   n = 0;
   for (i=0; i<np; i++) {
      for (j=0; j<np; j++) {
         displs[n] = i*local_N*N+j*local_N;
         counts[n++] = local_N;
      }
   }

   for (i=0; i<local_N; i++)
      MPI_Gatherv(&local_A[i][0], local_N, MPI_FLOAT, &A[i][0], counts,
                  displs, MPI_FLOAT, 0, MPI_COMM_WORLD);

   free(displs); free(counts);
}

// initialize the matrix to zero
void init(int N, float **A)
{
   int i, j;

   for (i=0; i<N; i++)
      for (j=0; j<N; j++)
        A[i][j] = 0.0;
}

// copy a matrix to another
void copy(int N, float **A, float **B)
{
   int i, j;

   for (i=0; i<N; i++)
      for (j=0; j<N; j++)
        B[i][j] = A[i][j];
}

// matrix multiplication for local matrices
void mmult(int N, float **A, float **B, float **C)
{
   int i, j, k;

   for (i=0; i<N; i++) {
      for (j=0; j<N; j++) {
         for (k=0; k<N; k++) {
            C[i][j] += A[i][k] * B[k][j];
         }
      }
   }
}

// check results
void check(int N, float A[N][N], float B[N][N], float C[N][N]) {
   int i, j, k;
   float C2[N][N];

   for (i=0; i<N; i++)
      for (j=0; j<N; j++) {
         C2[i][j] = 0.0;
         for (k=0; k<N; k++) {
            C2[i][j] += A[i][k] * B[k][j];
         }
      }
   for (i=0; i<N; i++)
      for (j=0; j<N; j++) {
         if ((C[i][j]-C2[i][j])>0.001) printf("%d %d: %f %f\n", i, j,  C[i][j], C2[i][j]);
      }
}


#include <stdio.h>
#define N 10000
#define THREADS 100

__global__ void saxpy(float *A, float*B, float X, float Y){
   int i = blockDim.x*blockIdx.x+threadIdx.x;

   B[i] = A[i]*X;
   B[i] += Y;
}

int main()
{
   float A[N], B[N], B2[N], X, Y;
   float *A_d, *B_d;
   int i;

   dim3 dimBlock(THREADS);
   dim3 dimGrid((N+dimBlock.x-1)/dimBlock.x);

   for (i=0; i<N; i++)
      A[i] = i*2;

   X = 1.23;
   Y = 2.34;
   for (i=0; i<N; i++)
      B2[i] = A[i]*X + Y; // B2 is used for checking

   cudaMalloc((void**) &A_d, sizeof(float)*N);
   cudaMalloc((void**) &B_d, sizeof(float)*N);

   cudaMemcpy(A_d, A, sizeof(float)*N, cudaMemcpyHostToDevice);
   saxpy<<<dimGrid, dimBlock>>>(A_d, B_d, X, Y);

   cudaMemcpy(B, B_d, sizeof(float)*N, cudaMemcpyDeviceToHost);

   for (i=0; i<N; i++)
      if (fabs(B[i]-B2[i]) > 0.001)
         printf("%d: %f %f\n",i, B[i], B2[i]);

   cudaFree(A_d);
   cudaFree(B_d);
}

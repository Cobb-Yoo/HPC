
#include <stdio.h>

#define N 8
#define THREADS 8

__global__ void reduce(float *A, float *result)
{
   __shared__ float sdata[THREADS];
   int i = blockDim.x*blockIdx.x+threadIdx.x;
   sdata[threadIdx.x] = A[i];

   for(unsigned s = blockDim.x/2;s > 0; s>>=1)
   {
      if(threadIdx.x < s && sdata[threadIdx.x] < sdata[threadIdx.x+s])
         sdata[threadIdx.x] = sdata[threadIdx.x+s];
      __syncthreads();
   }

   if(threadIdx.x == 0) *result = sdata[0];
}

int main()
{
   float A[N], *A_d, *result, *result_d;
   int i;

   dim3 dimBlock(THREADS);
   dim3 dimGrid((N+dimBlock.x-1)/dimBlock.x);

   for (i=0; i<N; i++)
      A[i] = N-i;
   A[3] = 2*N;
   A[N-3] = -N;

   cudaMalloc((void **) &A_d, sizeof(float)*N);
   cudaMemcpy(A_d, A, sizeof(float)*N, cudaMemcpyHostToDevice);

   cudaMalloc((void **) &result_d, sizeof(float));

   reduce<<<dimGrid, dimBlock>>>(A_d, result_d);
   result = (float*)malloc(sizeof(float));
   cudaMemcpy(result, result_d, sizeof(float), cudaMemcpyDeviceToHost);

   printf("%f\n", *result);

   cudaFree(A_d);
   cudaFree(result_d);
   cudaFree(result);
}


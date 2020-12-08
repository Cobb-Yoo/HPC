
#include <stdio.h>
#include <stdlib.h>

#define N 128000
#define THREADS 128

__global__ void solve(float a, float b, float *results)
{
   __shared__ float sdata[THREADS];
   float dx = (b-a)/N;
   float x = dx*(blockDim.x*blockIdx.x+threadIdx.x);

   sdata[threadIdx.x] = (x/((x*x+2)*(x*x+2)*(x*x+2)))*dx;

   for(unsigned int s = blockDim.x/2;s > 0;s >>=1)
   {
      if(threadIdx.x < s)
         sdata[threadIdx.x] += sdata[threadIdx.x+s];
      __syncthreads();
   }

   if(threadIdx.x == 0) results[blockIdx.x] = sdata[0];
}

main(){
   int i;
   float a,b;
   float sum, *results, *results_d;
   float elapsed_time;
   cudaEvent_t start, stop;

   dim3 dimBlock(THREADS);
   dim3 dimGrid((N+dimBlock.x-1)/dimBlock.x);
   cudaSetDevice(0);

   a = 0.0;
   b = 2.0;


   cudaEventCreate(&start);
   cudaEventCreate(&stop);
   cudaEventRecord(start,0);

   cudaMalloc((void **) &results_d, sizeof(float)*dimGrid.x);
   solve<<<dimGrid, dimBlock>>>(a, b, results_d);
   results = (float*)malloc(dimGrid.x*sizeof(float));
   cudaMemcpy(results, results_d, dimGrid.x*sizeof(float), cudaMemcpyDeviceToHost);

   cudaEventRecord(stop,0);
   cudaEventSynchronize(stop);
   cudaEventElapsedTime(&elapsed_time,start,stop);

   sum = 0.0;
   for(i=0;i<dimGrid.x;i++) sum += results[i];

   printf("elapsed Time: %f ms\n",elapsed_time);
   printf("reulst: %f\n", sum);

   cudaFree(results_d);
   free(results);
}


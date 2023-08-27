#include <math.h>
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void vecAddKernel(float *a, float *b, float *c, int n)
{
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < n)
  {
    c[i] = a[i] + b[i];
  }
}

void vecAdd(float *a_h, float *b_h, float *c_h, int n)
{
  int size = n * sizeof(float);
  float *a_d, *b_d, *c_d;

  cudaMalloc((void **)&a_d, size);
  cudaMalloc((void **)&b_d, size);
  cudaMalloc((void **)&c_d, size);

  cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice);
  cudaMemcpy(b_d, b_h, size, cudaMemcpyHostToDevice);

  vecAddKernel<<<ceil(n / 256.0), 256>>>(a_d, b_d, c_d, n);

  cudaMemcpy(c_h, c_d, size, cudaMemcpyDeviceToHost);
  cudaFree(a_d);
  cudaFree(b_d);
  cudaFree(c_d);
}

int main(int argc, char const *argv[])
{
  int n = 1000;

  float *h_a, *h_b, *h_c;
  h_a = new float[n];
  h_b = new float[n];
  h_c = new float[n];
  for (int i = 0; i < n; i++)
  {
    h_a[i] = i * 1.0;
    h_b[i] = i * 1.0;
  }

  vecAdd(h_a, h_b, h_c, n);
  for (int i = 0; i < n; i++)
  {
    printf("%f|", h_c[i]);
  }
  printf("\n");
  return 0;
}

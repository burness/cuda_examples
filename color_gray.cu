#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

__global__ void colortoGrayscaleConvertionKernel(unsigned char *pout, unsigned char *pin, int width, int height)
{
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (col < width && row < height)
  {
    int grayOffset = row * width + col;
    int rgbOffset = grayOffset * 3;
    unsigned char r = pin[rgbOffset];
    unsigned char g = pin[rgbOffset + 1];
    unsigned char b = pin[rgbOffset + 2];

    pout[grayOffset] = 0.21f * r + 0.71f * g + 0.07f * b;
    // printf("r: %d, g: %d, b: %d, pout at grayoffset: %d", r, g, b, pout[grayOffset]);
  }
}

void colortoGrayscaleConvertion(unsigned char *pout_h, unsigned char *pin_h, int width, int height)
{
  int size = 3 * height * width * sizeof(char);
  int pout_size = height * width * sizeof(char);
  unsigned char *pin_d, *pout_d;

  cudaMalloc((void **)&pin_d, size);
  cudaMalloc((void **)&pout_d, pout_size);

  cudaMemcpy(pin_d, pin_h, size, cudaMemcpyHostToDevice);
  dim3 dim_grid(ceil(width / 16.0), ceil(height / 16.0), 1);
  dim3 dim_block(16, 16, 1);

  colortoGrayscaleConvertionKernel<<<dim_grid, dim_block>>>(pout_d, pin_d, width, height);

  cudaMemcpy(pout_h, pout_d, pout_size, cudaMemcpyDeviceToHost);
  cudaFree(pin_d);
  cudaFree(pout_d);
}

int main(int argc, char const *argv[])
{
  int width = 64;
  int height = 64;
  int n = 3 * width * height;
  int pout_n = width * height;
  unsigned char *pin_h, *pout_h;
  pin_h = new unsigned char[n];
  for (int i = 0; i < n; i++)
  {
    pin_h[i] = i % 255;
  }
  pout_h = new unsigned char[pout_n];

  colortoGrayscaleConvertion(pout_h, pin_h, width, height);
  for (int i = 0; i < pout_n; i++)
  {
    printf("%d|", pout_h[i]);
  }
  delete[] pin_h;
  delete[] pout_h;
  return 0;
}

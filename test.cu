#include <stdio.h>
#include <cstdlib>
__global__ void mallocTest()
{
    char* ptr = (char*)malloc(123);
    printf("Thread %d got pointer: %p \n",threadIdx.x,ptr);
    delete ptr;
}


int main()
{
    cudaDeviceSetLimit(cudaLimitMallocHeapSize,128*1024*1024);
    mallocTest<<<1,5>>>();
    cudaDeviceSynchronize();
    return 0;
}
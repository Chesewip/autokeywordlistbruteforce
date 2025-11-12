#include <cuda_runtime.h>

// simple example kernel
__global__ void test_kernel(const uint8_t* cipher_indices,
    const uint8_t* key_indices,
    int text_len,
    int key_len,
    double* out_scores)
{
    //int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // ... compute score for combo idx ...
}
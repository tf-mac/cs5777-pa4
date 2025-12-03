#include <iostream>
#include <cassert>
#include <vector>
#include <utility>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/Atomic.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/types.h>
#include <torch/extension.h>
#include <mma.h>

using namespace torch::indexing;
using namespace nvcuda;

#define FULL_MASK 0xffffffff
#define HALF_MASK 0x0000ffff

#define CHECK_CUDA(x)           TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)     TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) 	        do { CHECK_CUDA(x); CHECK_CONTIGUOUS(x); } while(false)
#define gpuErrchk(ans)          do { gpuAssert((ans), __FILE__, __LINE__); } while (false)


__global__ void sgemm_p1_kernel(
  const float* A,
  const float* B,
  float* C,
  const size_t M,
  const size_t N,
  const size_t K
) {
  size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

  size_t iin = tid % N;
  size_t iim = tid / N;

  if (iim > M) {
    // no work for this thread to do
    return;
  }

  for(size_t i = 0; i < K, ++i) {
    C[iim * N + iin] += A[iim * K + i] * B[i * K + iin];
  }
}

// C += A @ B.t()
void sgemm_p1(
  torch::Tensor &A,
  torch::Tensor &B,
  torch::Tensor &C
) {
  CHECK_INPUT(A);
  CHECK_INPUT(B);
  CHECK_INPUT(C);

  // make sure that the current GPU is the one associated with this device
  const torch::OptionalDeviceGuard guard(C.device());

  assert(A.dim() == 2);
  assert(B.dim() == 2);
  assert(C.dim() == 2);

  const size_t M = C.sizes()[0];
  const size_t N = C.sizes()[1];
  const size_t K = A.sizes()[1];
  assert(A.sizes()[0] == M);
  assert(B.sizes()[0] == N);
  assert(B.sizes()[1] == K);

  assert(A.dtype() == torch::kFloat32);
  assert(B.dtype() == torch::kFloat32);
  assert(C.dtype() == torch::kFloat32);

  const size_t THREADS_PER_BLOCK = 256;
  const size_t TOTAL_THREADS = M * N;
  const size_t TOTAL_BLOCKS = (TOTAL_THREADS + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

  const dim3 threads(THREADS_PER_BLOCK);
  const dim3 blocks(TOTAL_BLOCKS);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  sgemm_p1_kernel<<<blocks, threads, 0, stream>>>(
    A.data_ptr<float>(),
    B.data_ptr<float>(),
    C.data_ptr<float>(),
    M,
    N,
    K
  );
}


__global__ void hgemm_p1_kernel(
  const __half* A,
  const __half* B,
  __half* C,
  const size_t M,
  const size_t N,
  const size_t K
) {
  size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

  size_t iin = tid % N;
  size_t iim = tid / N;

  if (iim > M) {
    // no work for this thread to do
    return;
  }

  for(size_t i = 0; i < K, ++i) {
    C[iim * N + iin] += A[iim * K + i] * B[i * K + iin];
  }
}

// C += A @ B.t()
void hgemm_p1(
  torch::Tensor &A,
  torch::Tensor &B,
  torch::Tensor &C
) {
  CHECK_INPUT(A);
  CHECK_INPUT(B);
  CHECK_INPUT(C);

  // make sure that the current GPU is the one associated with this device
  const torch::OptionalDeviceGuard guard(C.device());

  assert(A.dim() == 2);
  assert(B.dim() == 2);
  assert(C.dim() == 2);

  const size_t M = C.sizes()[0];
  const size_t N = C.sizes()[1];
  const size_t K = A.sizes()[1];
  assert(A.sizes()[0] == M);
  assert(B.sizes()[0] == N);
  assert(B.sizes()[1] == K);

  assert(A.dtype() == torch::kFloat16);
  assert(B.dtype() == torch::kFloat16);
  assert(C.dtype() == torch::kFloat16);

  const size_t THREADS_PER_BLOCK = 256;
  const size_t TOTAL_THREADS = M * N;
  const size_t TOTAL_BLOCKS = (TOTAL_THREADS + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

  const dim3 threads(THREADS_PER_BLOCK);
  const dim3 blocks(TOTAL_BLOCKS);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  hgemm_p1_kernel<<<blocks, threads, 0, stream>>>(
    (const __half*)A.data_ptr<at::Half>(),
    (const __half*)B.data_ptr<at::Half>(),
    (__half*)C.data_ptr<at::Half>(),
    M,
    N,
    K
  );
}



__global__ void sgemm_p2_kernel(
  const float* A,
  const float* B,
  float* C,
  const size_t M,
  const size_t N,
  const size_t K
) {
  __shared__ float* A_tile[256];
  __shared__ float* B_tile[256];
  float acc = 0;
  size_t iin_start = blockIdx.x * 16 % N;
  size_t iim_start = blockIdx.x * 16 / N;
  size_t stages = K / 16;
  for (size_t i = 0; i < stages; ++i) {
    // size_t 
    // A_tile[iin_start + threadIdx.x  ]
  }
}

// C += A @ B.t()
void sgemm_p2(
  torch::Tensor &A,
  torch::Tensor &B,
  torch::Tensor &C
) {
  CHECK_INPUT(A);
  CHECK_INPUT(B);
  CHECK_INPUT(C);

  // make sure that the current GPU is the one associated with this device
  const torch::OptionalDeviceGuard guard(C.device());

  assert(A.dim() == 2);
  assert(B.dim() == 2);
  assert(C.dim() == 2);

  const size_t M = C.sizes()[0];
  const size_t N = C.sizes()[1];
  const size_t K = A.sizes()[1];
  assert(A.sizes()[0] == M);
  assert(B.sizes()[0] == N);
  assert(B.sizes()[1] == K);

  assert(A.dtype() == torch::kFloat32);
  assert(B.dtype() == torch::kFloat32);
  assert(C.dtype() == torch::kFloat32);

  // for simplicity, restrict the matrices we support to ones that are a multiple of the block size
  assert(M % 16 == 0);
  assert(N % 16 == 0);
  assert(K % 16 == 0);

  const size_t MM_BLOCK_SIZE = 16; // split matrix into 16x16 blocks
  const size_t m_blocks = M / MM_BLOCK_SIZE;
  const size_t n_blocks = N / MM_BLOCK_SIZE;

  const dim3 threads(16,16);
  const dim3 blocks(m_blocks,n_blocks);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  sgemm_p2_kernel<<<blocks, threads, 0, stream>>>(
    A.data_ptr<float>(),
    B.data_ptr<float>(),
    C.data_ptr<float>(),
    M,
    N,
    K
  );
}


__global__ void hgemm_p2_kernel(
  const __half* A,
  const __half* B,
  __half* C,
  const size_t M,
  const size_t N,
  const size_t K
) {
  /* TODO: (2.3) */
  assert(false); // not implemented yet
}

// C += A @ B.t()
void hgemm_p2(
  torch::Tensor &A,
  torch::Tensor &B,
  torch::Tensor &C
) {
  CHECK_INPUT(A);
  CHECK_INPUT(B);
  CHECK_INPUT(C);

  // make sure that the current GPU is the one associated with this device
  const torch::OptionalDeviceGuard guard(C.device());

  assert(A.dim() == 2);
  assert(B.dim() == 2);
  assert(C.dim() == 2);

  const size_t M = C.sizes()[0];
  const size_t N = C.sizes()[1];
  const size_t K = A.sizes()[1];
  assert(A.sizes()[0] == M);
  assert(B.sizes()[0] == N);
  assert(B.sizes()[1] == K);

  assert(A.dtype() == torch::kFloat16);
  assert(B.dtype() == torch::kFloat16);
  assert(C.dtype() == torch::kFloat16);

  // for simplicity, restrict the matrices we support to ones that are a multiple of the block size
  assert(M % 16 == 0);
  assert(N % 16 == 0);
  assert(K % 16 == 0);

  const size_t MM_BLOCK_SIZE = 16; // split matrix into 16x16 blocks
  const size_t m_blocks = M / MM_BLOCK_SIZE;
  const size_t n_blocks = N / MM_BLOCK_SIZE;

  const dim3 threads(16,16);
  const dim3 blocks(m_blocks,n_blocks);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  hgemm_p2_kernel<<<blocks, threads, 0, stream>>>(
    (const __half*)A.data_ptr<at::Half>(),
    (const __half*)B.data_ptr<at::Half>(),
    (__half*)C.data_ptr<at::Half>(),
    M,
    N,
    K
  );
}

/* // uncomment this if you want to try tf32 tensorcores!
__global__ void sgemm_p3_kernel(
  const float* A,
  const float* B,
  float* C,
  const size_t M,
  const size_t N,
  const size_t K
) {
  // TODO: Optional Part 3
}

// C += A @ B.t()
void sgemm_p3(
  torch::Tensor &A,
  torch::Tensor &B,
  torch::Tensor &C
) {
  CHECK_INPUT(A);
  CHECK_INPUT(B);
  CHECK_INPUT(C);

  // make sure that the current GPU is the one associated with this device
  const torch::OptionalDeviceGuard guard(C.device());

  assert(A.dim() == 2);
  assert(B.dim() == 2);
  assert(C.dim() == 2);

  const size_t M = C.sizes()[0];
  const size_t N = C.sizes()[1];
  const size_t K = A.sizes()[1];
  assert(A.sizes()[0] == M);
  assert(B.sizes()[0] == N);
  assert(B.sizes()[1] == K);

  assert(A.dtype() == torch::kFloat32);
  assert(B.dtype() == torch::kFloat32);
  assert(C.dtype() == torch::kFloat32);

  // for simplicity, restrict the matrices we support to ones that are a multiple of the block size
  assert(M % 16 == 0);
  assert(N % 16 == 0);
  assert(K % 16 == 0);

  const size_t MM_BLOCK_SIZE = 16; // split matrix into 16x16 blocks
  const size_t m_blocks = M / MM_BLOCK_SIZE;
  const size_t n_blocks = N / MM_BLOCK_SIZE;

  const dim3 threads(32);
  const dim3 blocks(m_blocks,n_blocks);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  sgemm_p3_kernel<<<blocks, threads, 0, stream>>>(
    A.data_ptr<float>(),
    B.data_ptr<float>(),
    C.data_ptr<float>(),
    M,
    N,
    K
  );
}
*/

__global__ void hgemm_p3_kernel(
  const __half* A,
  const __half* B,
  __half* C,
  const size_t M,
  const size_t N,
  const size_t K
) {
  /* TODO: (3.1) */
  assert(false); // not implemented yet
}

// C += A @ B.t()
void hgemm_p3(
  torch::Tensor &A,
  torch::Tensor &B,
  torch::Tensor &C
) {
  CHECK_INPUT(A);
  CHECK_INPUT(B);
  CHECK_INPUT(C);

  // make sure that the current GPU is the one associated with this device
  const torch::OptionalDeviceGuard guard(C.device());

  assert(A.dim() == 2);
  assert(B.dim() == 2);
  assert(C.dim() == 2);

  const size_t M = C.sizes()[0];
  const size_t N = C.sizes()[1];
  const size_t K = A.sizes()[1];
  assert(A.sizes()[0] == M);
  assert(B.sizes()[0] == N);
  assert(B.sizes()[1] == K);

  assert(A.dtype() == torch::kFloat16);
  assert(B.dtype() == torch::kFloat16);
  assert(C.dtype() == torch::kFloat16);

  // for simplicity, restrict the matrices we support to ones that are a multiple of the block size
  assert(M % 16 == 0);
  assert(N % 16 == 0);
  assert(K % 16 == 0);

  const size_t MM_BLOCK_SIZE = 16; // split matrix into 16x16 blocks
  const size_t m_blocks = M / MM_BLOCK_SIZE;
  const size_t n_blocks = N / MM_BLOCK_SIZE;

  const dim3 threads(32);
  const dim3 blocks(m_blocks,n_blocks);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  hgemm_p3_kernel<<<blocks, threads, 0, stream>>>(
    (const __half*)A.data_ptr<at::Half>(),
    (const __half*)B.data_ptr<at::Half>(),
    (__half*)C.data_ptr<at::Half>(),
    M,
    N,
    K
  );
}


__global__ void hgemm_p4_kernel(
  const __half* A,
  const __half* B,
  float* C,
  const size_t M,
  const size_t N,
  const size_t K
) {
  /* TODO: (4.1) */
  assert(false); // not implemented yet
}

// C += A @ B.t()
void hgemm_p4(
  torch::Tensor &A,
  torch::Tensor &B,
  torch::Tensor &C
) {
  CHECK_INPUT(A);
  CHECK_INPUT(B);
  CHECK_INPUT(C);

  // make sure that the current GPU is the one associated with this device
  const torch::OptionalDeviceGuard guard(C.device());

  assert(A.dim() == 2);
  assert(B.dim() == 2);
  assert(C.dim() == 2);

  const size_t M = C.sizes()[0];
  const size_t N = C.sizes()[1];
  const size_t K = A.sizes()[1];
  assert(A.sizes()[0] == M);
  assert(B.sizes()[0] == N);
  assert(B.sizes()[1] == K);

  assert(A.dtype() == torch::kFloat16);
  assert(B.dtype() == torch::kFloat16);
  assert(C.dtype() == torch::kFloat32);

  // for simplicity, restrict the matrices we support to ones that are a multiple of the block size
  assert(M % 16 == 0);
  assert(N % 16 == 0);
  assert(K % 16 == 0);

  const size_t MM_BLOCK_SIZE = 16; // split matrix into 16x16 blocks
  const size_t m_blocks = M / MM_BLOCK_SIZE;
  const size_t n_blocks = N / MM_BLOCK_SIZE;

  const dim3 threads(32);
  const dim3 blocks(m_blocks,n_blocks);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  hgemm_p4_kernel<<<blocks, threads, 0, stream>>>(
    (const __half*)A.data_ptr<at::Half>(),
    (const __half*)B.data_ptr<at::Half>(),
    (float*)C.data_ptr<float>(),
    M,
    N,
    K
  );
}

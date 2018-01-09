#include "utils.h"

CudaTimer::CudaTimer() {
    cudaEventCreate(&start_);
    cudaEventCreate(&stop_);
}

CudaTimer::~CudaTimer() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
}

void CudaTimer::start() {
    cudaEventRecord(start_);
}

void CudaTimer::stop() {
    cudaEventRecord(stop_);
    cudaEventSynchronize(stop_);
    cudaEventElapsedTime(&milliseconds_, start_, stop_);
}

float CudaTimer::elapsed() {
    return milliseconds_;
}
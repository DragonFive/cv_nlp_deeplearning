#pragma once
#include <cuda_runtime.h>

class CudaTimer {
private:
    cudaEvent_t start_, stop_;
    float milliseconds_;

public:
    CudaTimer();
    ~CudaTimer();
    void start();
    void stop();
    float elapsed();
};
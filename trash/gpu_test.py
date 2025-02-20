import time
import numpy as np
import cupy as cp

# Matrix size (increase for more load)
N = 1000 

# CPU computation (NumPy)
A_cpu = np.random.rand(N, N)
B_cpu = np.random.rand(N, N)

start_cpu = time.time()
C_cpu = np.dot(A_cpu, B_cpu)
end_cpu = time.time()
cpu_time = end_cpu - start_cpu

print(f"CPU Time: {cpu_time:.6f} seconds")
cp.dot(cp.random.rand(10, 10), cp.random.rand(10, 10))  # Warm-up

# GPU computation (CuPy)
A_gpu = cp.random.rand(N, N)
B_gpu = cp.random.rand(N, N)

# cp.cuda.Device(0).synchronize()  # Ensure GPU is ready

start_gpu = time.time()
C_gpu = cp.matmul(A_gpu, B_gpu)
cp.cuda.Device(0).synchronize()  # Wait for completion
end_gpu = time.time()
gpu_time = end_gpu - start_gpu

print(f"GPU Time: {gpu_time:.6f} seconds")

speedup = cpu_time / gpu_time if gpu_time > 0 else float("inf")
print(f"Speedup Factor: {speedup:.2f}x")
print(cp.linalg.get_array_module(cp.array([1.0])))

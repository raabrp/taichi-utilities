Basic utilities for experimental simulations on the GPU, written in Taichi.
This is not necessarily finished code, but it's in a state that may be
beneficial to others.

MIT License.

# Fast Fourier Transform (fft.py)

+ Cooley-Tukey Fast Fourier Transform (FFT) for power-of-2-length input.
+ Breadth-first, decimation in time (DIT), radix-2, in-place.
+ Multiple dimensions, 32-bit precision.

# Psuedo Random Number Generation (prng.py) 

+ Parallel generation of pseudo-random numbers on the GPU.
+ Deterministic (for replicable experiments).
+ Uses xoshiro128+ under the hood; initial state set with numpy (PCG64).

# Bitonic Merge Sorting (sort.py) 

+ Single input can be parallelized across GPU threads for large N
+ For small N, can per form a search per-thread.

# Moore-Penrose Pseudo-Inverse (pinv.py)

+ Iterative approximation using (Gauss)-Newton

# Parallel Edge Detection with OpenMP and CUDA

## Build

### Requirements
- Nvidia GPU
- Up-to-date GPU driver
- gcc and nvcc
```sh	
make
```

## Run
```sh
# For sequential program
./seq_main papagan.JPG papagan2.JPG
```
```sh
# For CUDA program
./cuda_main papagan.JPG papagan2.JPG 4 8 16
```
```sh
# For CUDA and OpenMP program
./cuda_openMP_main papagan.JP G papagan2.JPG 4 8 16
```

## Explanation
There is 2 CUDA-OpenMP and 1 C file

- Sequential file works in a traditional way
- In cuda_main.cu, you can use either CUDA or OpenMP
    - For using CUDA, you need to put comment line in front of the 147th line
    - For using OpenMP, you need to put comment line in front of the 150th, 151st and 156th lines
- In cuda_openMP_main.cu, we used both CUDA and OpenMP libraries
    - It is basically use CUDA(GPU parallelism) and OpenMP(CPU parallelism, in main) at the same time.

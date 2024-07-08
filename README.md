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
./seq_main assets/papagan.JPG papagan2.JPG
```
```sh
# For CUDA program
./cuda_main assets/papagan.JPG papagan2.JPG 4 8 16
```
```sh
# For CUDA and OpenMP program
./cuda_openMP_main assets/papagan.JPG papagan2.JPG 4 8 16
```

## Explanation
### There is 2 CUDA-OpenMP and 1 C file
- Sequential file works in a traditional way
- In cuda_main.cu, you can use either CUDA or OpenMP
    - For using CUDA, you need to put comment line in front of the 147th line
    - For using OpenMP, you need to put comment line in front of the 150th, 151st and 156th lines
- In cuda_openMP_main.cu, we used both CUDA and OpenMP libraries
    - It is basically use CUDA(GPU parallelism) and OpenMP(CPU parallelism, in main) at the same time.

### There are 3 parameters after the image names
- First one indicates the number of OpenMP threads
- Second one indicates the number of blocks in CUDA
- Third one indicates the number of threads per block in CUDA

## Discussion
You can find a discussion about how execution time changes when we change the parameters in Report.pdf

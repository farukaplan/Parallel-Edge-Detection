/*
 * CENG342 Project-2
 *
 * Edge Detection using CUDA and OpenMP
 *
 * @group_id 03
 * 
 * @authors
 * Faruk KAPLAN
 * 
 * @version 1.0, 18 May 2024
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <omp.h>
#include <cuda.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "headers/stb_image.h"
#include "headers/stb_image_write.h"

// CUDA kernel for edge detection (Sobel Algorithm)
__global__ void edgeDetectionKernel(const uint8_t *input_image, int width, int height, uint8_t *output_image, int start_row, int num_rows) {
    int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int yIndex = blockIdx.y * blockDim.y + threadIdx.y + start_row;

    int xStride = blockDim.x * gridDim.x;
    int yStride = blockDim.y * gridDim.y;

    int x_mask[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int y_mask[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

    for (int x = xIndex; x < width; x += xStride) {
        for (int y = yIndex; y < start_row + num_rows && y < height; y += yStride) {
            if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1) {
                int gx = 0, gy = 0;

                for (int i = -1; i <= 1; i++) {
                    for (int j = -1; j <= 1; j++) {
                        int pixel_val = input_image[(y + i) * width + (x + j)];
                        gx += pixel_val * x_mask[i + 1][j + 1];
                        gy += pixel_val * y_mask[i + 1][j + 1];
                    }
                }

                int sum = (int)sqrtf((float)(gx * gx) + (float)(gy * gy));
                sum = sum > 255 ? 255 : sum;
                output_image[y * width + x] = (uint8_t)sum;
            }
        }
    }
}

int main(int argc, char *argv[]) {
    int width, height, bpp;
    uint8_t *input_image = NULL, *output_image = NULL;

    if (argc != 6) {
        fprintf(stderr, "Usage: %s <input_image> <output_image> <numThreadsOpenMP> <numBlocks> <numThreadsPerBlock>\n", argv[0]);
        return 1;
    }

    int numThreadsOpenMP = atoi(argv[3]);
    int numBlocks = atoi(argv[4]);
    int numThreadsPerBlock = atoi(argv[5]);

    input_image = stbi_load(argv[1], &width, &height, &bpp, 1);

    if (input_image == NULL) {
        fprintf(stderr, "Error in loading the image\n");
        return 1;
    }

    output_image = (uint8_t *)malloc(width * height * sizeof(uint8_t));

    // Variables for measure the time
    double start_time, end_time, total_time = 0.0;

    // OpenMP part that parallelize CPU-side code
    #pragma omp parallel for num_threads(numThreadsOpenMP) schedule(guided, 100)
    for (int thread_id = 0; thread_id < numThreadsOpenMP; ++thread_id) {
        int num_threads = numThreadsOpenMP;
        int rows_per_thread = (height + num_threads - 1) / num_threads;
        int start_row = thread_id * rows_per_thread;
        int end_row = (start_row + rows_per_thread > height) ? height : start_row + rows_per_thread;
        int num_rows = end_row - start_row;

        uint8_t *d_input_image, *d_output_image;

        cudaMalloc((void**)&d_input_image, width * height * sizeof(uint8_t));
        cudaMalloc((void**)&d_output_image, width * height * sizeof(uint8_t));

        cudaMemcpy(d_input_image, input_image, width * height * sizeof(uint8_t), cudaMemcpyHostToDevice);

        dim3 blockDim(numThreadsPerBlock, numThreadsPerBlock);
        dim3 gridDim(numBlocks, numBlocks);

        // Measure time for each thread seperately
        start_time = omp_get_wtime();

        edgeDetectionKernel<<<gridDim, blockDim>>>(d_input_image, width, height, d_output_image, start_row, num_rows);
        cudaDeviceSynchronize();

        end_time = omp_get_wtime();

        // Sum all thread's measurements
        total_time += end_time - start_time;

        cudaMemcpy(output_image + start_row * width, d_output_image + start_row * width, num_rows * width * sizeof(uint8_t), cudaMemcpyDeviceToHost);

        cudaFree(d_input_image);
        cudaFree(d_output_image);
    }

    
    printf("Edge detection time: %.6f seconds\n", total_time);

    if (!stbi_write_jpg(argv[2], width, height, 1, output_image, 100)) {
        fprintf(stderr, "Error in writing the output image\n");
    } else {
        printf("Edge detection completed successfully.\n");
    }

    stbi_image_free(input_image);
    free(output_image);

    return 0;
}


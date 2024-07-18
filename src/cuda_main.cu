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

/*
* For using CUDA, you need to put comment line in front of the 147th line
* For using OpenMP, you need to put comment line in front of the 150th, 151st and 156th lines 
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <omp.h>
#include <cmath>
#include <cstdint>
#include <omp.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "headers/stb_image.h"
#include "headers/stb_image_write.h"

// CUDA kernel for edge detection (Sobel Algorithm)
__global__ void edgeDetectionKernel(const uint8_t *input_image, int width, int height, uint8_t *output_image, int numBlock) {
    int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

    // Calculate the stride for x and y dimensions
    int xStride = blockDim.x * gridDim.x;
    int yStride = blockDim.y * gridDim.y;

    int x_mask[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int y_mask[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

    for (int x = xIndex; x < width; x += xStride) {
        for (int y = yIndex; y < height; y += yStride) {
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

void edgeDetectionCPU(const uint8_t *input_image, int width, int height, uint8_t *output_image, int num_threads) {
    int x_mask[3][3] = { {-1, 0, 1}, {-2, 0, 2},{-1, 0, 1} };
    int y_mask[3][3] = { {-1, -2, -1}, {0, 0, 0}, {1, 2, 1} };

    // Set the default scheduler
    omp_set_schedule(omp_sched_guided, 1000);

    // Initialize the edges of the output_image to zero (or some other default value)
    #pragma omp parallel for num_threads(num_threads) 
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (y == 0 || y == height - 1 || x == 0 || x == width - 1) {
                output_image[y * width + x] = 0;
            }
        }
    }

    #pragma omp parallel for num_threads(num_threads) 
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            int gx = 0, gy = 0;

            for (int i = -1; i <= 1; i++) {
                for (int j = -1; j <= 1; j++) {
                    int pixel_val = input_image[(y + i) * width + (x + j)];
                    gx += pixel_val * x_mask[i + 1][j + 1];
                    gy += pixel_val * y_mask[i + 1][j + 1];
                }
            }

            int sum = (int)std::sqrt((gx * gx) + (gy * gy));
            sum = sum > 255 ? 255 : sum;
            output_image[y * width + x] = (uint8_t)sum;
        }
    }
}

int main(int argc, char *argv[]) {
    int width, height, bpp;
    uint8_t *input_image = NULL, *output_image = NULL;
    double start_time, end_time;

    if (argc != 6) {
        fprintf(stderr, "Lack of input parameters\n", argv[0]);
        return 1;
    }

    int numThreadsOpenMP = atoi(argv[3]);
    int numThread = atoi(argv[4]);
    int numBlock = atoi(argv[5]);
    
    input_image = stbi_load(argv[1], &width, &height, &bpp, 1);

    if (input_image == NULL) {
        fprintf(stderr, "Error in loading the image\n");
        return 1;
    }

    output_image = (uint8_t *)malloc(width * height * sizeof(uint8_t));

    uint8_t *d_input_image, *d_output_image;

    // Allocate memory on the GPU
    cudaMalloc((void**)&d_input_image, width * height * sizeof(uint8_t));
    cudaMalloc((void**)&d_output_image, width * height * sizeof(uint8_t));

    // Copy data from host to device
    cudaMemcpy(d_input_image, input_image, width * height * sizeof(uint8_t), cudaMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 blockDim(numThread, numThread);
    dim3 gridDim(numBlock, numBlock);

    // Start measuring the time
    start_time = omp_get_wtime();

    /* !!! Use one of this functions, with openMP or CUDA. Put a command line in front of unused function !!! */
    // OpenMP
    edgeDetectionCPU(input_image, width, height, output_image, numThreadsOpenMP);

    // CUDA
    //edgeDetectionKernel<<<gridDim, blockDim>>>(d_input_image, width, height, d_output_image, numBlock);
    //cudaDeviceSynchronize();

    end_time = omp_get_wtime();

    // Copy the result back to host. If you are using openMP, put a command line here also 
    //cudaMemcpy(output_image, d_output_image, width * height * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    // Save the output image
    if (!stbi_write_jpg(argv[2], width, height, 1, output_image, 100)) {
        fprintf(stderr, "Error in writing the output image\n");
    } else {
        printf("Edge detection time: %.6f\n", end_time - start_time);
        printf("Edge detection completed successfully.\n");
    }

    // Free memory
    stbi_image_free(input_image);
    free(output_image);
    cudaFree(d_input_image);
    cudaFree(d_output_image);

    return 0;
}

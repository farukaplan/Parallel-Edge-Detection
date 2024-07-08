/*
 * CENG342 Project-2
 *
 * Edge Detection 
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
#include <time.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"

void seq_edgeDetection(const uint8_t *input_image, int width, int height, uint8_t *output_image) {
    int gx, gy, sum;

    
    int x_mask[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int y_mask[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            gx = gy = 0;

           
            for (int i = -1; i <= 1; i++) {
                for (int j = -1; j <= 1; j++) {
                    int pixel_val = input_image[(y + i) * width + (x + j)];
                    gx += pixel_val * x_mask[i + 1][j + 1];
                    gy += pixel_val * y_mask[i + 1][j + 1];
                }
            }

            sum = (int)sqrt((gx * gx) + (gy * gy));
            // Normalize to 0-255
            sum = sum > 255 ? 255 : sum;
            output_image[y * width + x] = (uint8_t)sum;
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input_image> <output_image>\n", argv[0]);
        exit(1);
    }

    int width, height, bpp;
    uint8_t *input_image = stbi_load(argv[1], &width, &height, &bpp, 1);
    if (input_image == NULL) {
        fprintf(stderr, "Error in loading the image\n");
        exit(1);
    }

    uint8_t *output_image = (uint8_t *)malloc(width * height);
    if (output_image == NULL) {
        fprintf(stderr, "Error in allocating memory for the output image\n");
        stbi_image_free(input_image);
        exit(1);
    }

    clock_t start_time = clock();
    seq_edgeDetection(input_image, width, height, output_image);
    clock_t end_time = clock();
    double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    if (!stbi_write_jpg(argv[2], width, height, 1, output_image, 100)) {
        fprintf(stderr, "Error in writing the image\n");
        free(output_image);
        stbi_image_free(input_image);
        exit(1);
    }

    free(output_image);
    stbi_image_free(input_image);

    printf("Edge detection completed successfully.\n");
    printf("Elapsed time: %lf seconds\n", elapsed_time);

    return 0;
}

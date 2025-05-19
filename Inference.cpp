#include <iostream>
#include <cmath>
#include<vector>
#include <numeric>
#include <algorithm>
#include "W_conv.h"
#include "b_conv.h"
#include "W_fc.h"
#include "b_fc.h"
#include "input_image.h"

#define IMG_SIZE 28
#define FILTER_SIZE 5
#define NUM_FILTERS 8
#define POOL_SIZE 2
#define DENSE_INPUT_SIZE 1152
#define DENSE_OUTPUT_SIZE 10

// ReLU
float relu(float x) {
    return std::max(0.0f, x);
}


// Softmax (as per your DOC: no max subtraction)
void softmax(float* input, int size) {
    float sum = 0.0f;
    for (int i = 0; i < 10; i++) {
        input[i] = std::exp(input[i]);
        sum += input[i];
    }
    for (int i = 0; i < 10; i++) {
        input[i] /= sum;
    }
}


// Conv2D valid padding
void conv2d(float input[IMG_SIZE][IMG_SIZE], float output[NUM_FILTERS][24][24]) {
    for (int f = 0; f < NUM_FILTERS; f++) {
        for (int i = 0; i < 24; i++) {
            for (int j = 0; j < 24; j++) {
                float sum = 0.0f;
                for (int ki = 0; ki < FILTER_SIZE; ki++) {
                    for (int kj = 0; kj < FILTER_SIZE; kj++) {
                        sum += input[i + ki][j + kj] * W_conv[f * 25 + ki * 5 + kj];
                    }
                }
                output[f][i][j] = relu(sum + b_conv[f]);
            }
        }
    }
}

// MaxPool2D 2x2
void maxpool(float input[NUM_FILTERS][24][24], float output[NUM_FILTERS][12][12]) {
    for (int f = 0; f < NUM_FILTERS; f++) {
        for (int i = 0; i < 12; i++) {
            for (int j = 0; j < 12; j++) {
                float max_val = 0.0f;
                for (int ki = 0; ki < 2; ki++) {
                    for (int kj = 0; kj < 2; kj++) {
                        float val = input[f][i*2 + ki][j*2 + kj];
                        if (ki == 0 && kj == 0 || val > max_val)
                            max_val = val;
                    }
                }
                output[f][i][j] = max_val;
            }
        }
    }
}

// Flatten
void flatten(float input[NUM_FILTERS][12][12], float output[DENSE_INPUT_SIZE]) {
    int idx = 0;
    for (int f = 0; f < NUM_FILTERS; f++) {
        for (int i = 0; i < 12; i++) {
            for (int j = 0; j < 12; j++) {
                output[idx++] = input[f][i][j];
            }
        }
    }
}

// Dense layer
void dense(float input[DENSE_INPUT_SIZE], float output[DENSE_OUTPUT_SIZE]) {
    for (int i = 0; i < DENSE_OUTPUT_SIZE; i++) {
        float sum = 0.0f;
        for (int j = 0; j < DENSE_INPUT_SIZE; j++) {
            sum += input[j] * W_fc[i * DENSE_INPUT_SIZE + j];
        }
        output[i] = sum + b_fc[i];
    }
}

int main() {
    // Convert 1D input to 2D image
    float img[IMG_SIZE][IMG_SIZE];
    for (int i = 0; i < IMG_SIZE; i++)
        for (int j = 0; j < IMG_SIZE; j++)
            img[i][j] = input_img[i][j];

    // CNN Inference
    float conv_out[NUM_FILTERS][24][24];
    float pool_out[NUM_FILTERS][12][12];
    float flat_out[DENSE_INPUT_SIZE];
    float dense_out[DENSE_OUTPUT_SIZE];

    conv2d(img, conv_out);
    maxpool(conv_out, pool_out);
    flatten(pool_out, flat_out);
    dense(flat_out, dense_out);
    softmax(dense_out, DENSE_OUTPUT_SIZE);

    // Prediction
    int predicted = std::max_element(dense_out, dense_out + DENSE_OUTPUT_SIZE) - dense_out;
    std::cout << "Predicted digit: " << predicted << std::endl;

    return 0;
}
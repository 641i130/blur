// A cli tool to apply gaussian blur and output an image
// Usage: ./tool <input image> <output image>
#include <iostream>
#include <opencv2/opencv.hpp>

__global__ void applyFilter(const unsigned char *input, unsigned char *output, const unsigned int width, const unsigned int height, const float *kernel, const unsigned int kernelWidth) {
    const unsigned int col = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int row = threadIdx.y + blockIdx.y * blockDim.y;
    if(row < height && col < width) {
        const int half = kernelWidth / 2;
        float blur = 0.0;
        float kernelSum = 0.0;
        for(int i = -half; i <= half; i++) {
            for(int j = -half; j <= half; j++) {
                const unsigned int y = max(0, min(height - 1, row + i));
                const unsigned int x = max(0, min(width - 1, col + j));
                const float w = kernel[(j + half) + (i + half) * kernelWidth];
                kernelSum += w;
                blur += w * input[x + y * width];
            }
        }
        output[col + row * width] = static_cast<unsigned char>(blur / kernelSum);
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input_image_path> <output_image_path> [blur]" << std::endl;
        return 1;
    }

    float blur = 8.0f; // default value
    if (argc > 3) {
        blur = std::stof(argv[3]);
    }
    // Load the input image
    cv::Mat input_image = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    if (input_image.empty()) {
        std::cerr << "Error: Unable to load input image" << std::endl;
        return 1;
    }

    // Set the filter kernel
   float sigma = blur; // Strength of blur
    int kernelWidth = static_cast<int>(3 * sigma);
    if (kernelWidth % 2 == 0) {
        kernelWidth -= 1;  // Make sure kernel width is odd
    }

    // Create empty matrix for the Gaussian kernel
    float* kernel = new float[kernelWidth * kernelWidth];
    int kernelHalfWidth = kernelWidth / 2;
    
    for (int i = -kernelHalfWidth; i <= kernelHalfWidth; i++) {
        for (int j = -kernelHalfWidth; j <= kernelHalfWidth; j++) {
            const float di = static_cast<float>(i * i + j * j) / (2 * sigma * sigma);
            const int index = (i + kernelHalfWidth) * kernelWidth + (j + kernelHalfWidth);
            kernel[index] = exp(-di) / (2 * M_PI * sigma * sigma);
        }
    }
    // Allocate memory for input and output data on the device
    const unsigned int width = input_image.cols;
    const unsigned int height = input_image.rows;
    const size_t input_size = width * height * sizeof(unsigned char);
    const size_t output_size = input_size;
    const size_t kernel_size = kernelWidth * kernelWidth * sizeof(float);
    unsigned char* d_input = nullptr;
    unsigned char* d_output = nullptr;
    float* d_kernel = nullptr;
    cudaMalloc(&d_input, input_size);
    cudaMalloc(&d_output, output_size);
    cudaMalloc(&d_kernel, kernel_size);

    // Copy input data and filter kernel to device memory
    cudaMemcpy(d_input, input_image.data, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, kernel_size, cudaMemcpyHostToDevice);

    // Define the kernel launch parameters
    const dim3 threads_per_block(16, 16);
    const dim3 blocks_per_grid((width + threads_per_block.x - 1) / threads_per_block.x, (height + threads_per_block.y - 1) / threads_per_block.y);

    // Launch the CUDA kernel
    applyFilter<<<blocks_per_grid, threads_per_block>>>(d_input, d_output, width, height, d_kernel, kernelWidth);

    // Copy output data from device memory to host memory
    unsigned char* output_data = new unsigned char[output_size];
    cudaMemcpy(output_data, d_output, output_size, cudaMemcpyDeviceToHost);

    
    // Create an output image from the processed data and save it to disk
    cv::Mat output_image(height, width, CV_8UC1, output_data);
    cv::imwrite(argv[2], output_image);

    // Free device memory and host data
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);
    delete[] output_data;
    return 0;
}


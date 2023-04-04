# blur
A fast, CUDA enabled (CUDA GPU required) gaussian blur image tool.

## Compile:

```nvcc -I/usr/include/opencv4 -o wow code.cu -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs```

## Usage:

```Usage: ./wow <input_image_path> <output_image_path>```

#### Credit:
(This is essentially a port of his python code to C++ with some modifications)
- https://github.com/harrytang/cuda-gaussian-blur


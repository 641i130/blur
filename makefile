# Makefile for compiling and testing the blur program

# Compiler
NVCC = nvcc

# Compiler flags
CFLAGS = -I/usr/include/opencv4 -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs

# Source files
SRC = main.cu

# Executable
EXEC = blur

all: $(EXEC)

$(EXEC): $(SRC)
	$(NVCC) $(CFLAGS) $(SRC) -o $(EXEC)

test: $(EXEC)
	scrot /tmp/ss.png
	./$(EXEC) ss.png ss.png
	sxiv ss.png

clean:
	rm -f $(EXEC) output_image.jpg


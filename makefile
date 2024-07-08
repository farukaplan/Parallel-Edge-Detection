NVCC = nvcc
GCC = gcc
CFLAGS = -Xcompiler -fopenmp
LDFLAGS = -lm

# Targets and source files
TARGET1 = cuda_main
SOURCES1 = src/cuda_main.cu

TARGET2 = cuda_openMP_main
SOURCES2 = src/cuda_openMP_main.cu

TARGET3 = seq_main
SOURCES3 = src/seq_main.c

# Default target to build 3 programs
all: $(TARGET1) $(TARGET2) $(TARGET3)

# Compile the first CUDA program
$(TARGET1): $(SOURCES1)
	$(NVCC) $(CFLAGS) -o $(TARGET1) $(SOURCES1) $(LDFLAGS)

# Compile the second CUDA program
$(TARGET2): $(SOURCES2)
	$(NVCC) $(CFLAGS) -o $(TARGET2) $(SOURCES2) $(LDFLAGS)

$(TARGET3): $(SOURCES3)
	$(GCC) $(LDFLAGS) $(SOURCES3) -o $(TARGET3) $(LDFLAGS)

# Phony target to clean up the executables
.PHONY: clean
clean:
	rm -f $(TARGET1) $(TARGET2) $(TARGET3)

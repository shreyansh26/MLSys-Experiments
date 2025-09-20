# Simple Makefile to build and run the shrink BGMV CUDA example

NVCC ?= nvcc

# Compute capability can be overridden, e.g., make SM=90
SM ?= 90a

SRC = bgmv.cu
EXE = bgmv

# Local CCCL include (thrust, cub, libcudacxx, cudax)
CCCL_INC = $(abspath ./cccl)

INCLUDES = -I$(CCCL_INC)/cub -I$(CCCL_INC)/libcudacxx/include -I$(CCCL_INC)/thrust -I$(CCCL_INC)/cudax/include

NVCCFLAGS = -std=c++17 -O3 -arch=sm_$(SM) --expt-relaxed-constexpr --extended-lambda

.PHONY: all run clean

all: $(EXE)

$(EXE): $(SRC) bgmv_kernel.cuh
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ $(SRC)

run: $(EXE)
	./$(EXE)

clean:
	rm -f $(EXE)



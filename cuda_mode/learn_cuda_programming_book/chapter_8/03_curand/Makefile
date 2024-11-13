GCC			= /usr/bin/g++
NVCC 		= /usr/bin/nvcc -ccbin ${GCC}

TARGET		= curand_host gemm_with_curand_host

INCLUDES	= -I/usr/include
NVCC_FLAGS	= -m64 -lineinfo

IS_CUDA_11	:= ${shell expr `${NVCC} --version | grep compilation | grep -Eo -m 1 '[0-9]+.[0-9]' | head -1` \>= 11.0}

# Gencode argumentes
SMS = 70
ifeq "${IS_CUDA_11}" "1"
SMS = 75 80 90
endif
$(foreach sm, ${SMS}, $(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

# Openmp
LIBRARIES += -lgomp -lcublas -lcurand
ALL_CCFLAGS += -g -Xcompiler -fopenmp -rdc=true $(NVCC_FLAGS)

all : ${TARGET}

curand_host: curand_host.cpp
	$(EXEC) $(NVCC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) $(INCLUDES) -o $@ $+ $(LIBRARIES)

gemm_with_curand_host.o: gemm_with_curand_host.cpp
	$(EXEC) $(NVCC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) $(INCLUDES) -c -o $@ $+

fp16.o: fp16.cu
	$(EXEC) $(NVCC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) $(INCLUDES) -c -o $@ $+

gemm_with_curand_host: gemm_with_curand_host.o fp16.o
	$(EXEC) $(NVCC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

clean:
	rm -f ${TARGET} *.o
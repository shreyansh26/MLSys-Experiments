GCC			= /usr/bin/g++
NVCC 		= /usr/bin/nvcc -ccbin ${GCC}

TARGET		= sgemm

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
LIBRARIES += -lgomp
ALL_CCFLAGS += -g -Xcompiler -fopenmp -rdc=true $(NVCC_FLAGS)

all : ${TARGET}

sgemm: sgemm.cu
	$(EXEC) $(NVCC) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

clean:
	rm -f ${TARGET} *.o
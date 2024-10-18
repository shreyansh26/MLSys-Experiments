GCC			= /usr/bin/g++
NVCC 		= /usr/bin/nvcc -ccbin ${GCC}

TARGET		= unified_simple unified_initialized unified_prefetch unified_64align

INCLUDES	= -I/usr/include
NVCC_FLAGS	= -m64 -lineinfo

IS_CUDA_11	:= ${shell expr `${NVCC} --version | grep compilation | grep -Eo -m 1 '[0-9]+.[0-9]' | head -1` \>= 11.0}

# Gencode argumentes
SMS = 70
ifeq "${IS_CUDA_11}" "1"
SMS = 75 80 90
endif
$(foreach sm, ${SMS}, $(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

all : unified_simple unified_initialized unified_prefetch unified_64align

unified_simple: unified_memory.cu
	${NVCC} ${INCLUDES} ${NVCC_FLAGS} ${GENCODE_FLAGS} -o $@.out $<

unified_initialized: unified_memory_initialized.cu
	${NVCC} ${INCLUDES} ${NVCC_FLAGS} ${GENCODE_FLAGS} -o $@.out $<

unified_prefetch: unified_memory_prefetch.cu
	${NVCC} ${INCLUDES} ${NVCC_FLAGS} ${GENCODE_FLAGS} -o $@.out $<

unified_64align: unified_memory_64align.cu
	${NVCC} ${INCLUDES} ${NVCC_FLAGS} ${GENCODE_FLAGS} -o $@.out $<

clean:
	rm -f *.out 
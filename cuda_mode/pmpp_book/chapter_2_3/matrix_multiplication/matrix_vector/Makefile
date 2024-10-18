
NVCC        = /usr/local/cuda/bin/nvcc
NVCC_FLAGS  = -O3 -I/usr/local/cuda/include
LD_FLAGS    = -lcudart -L/usr/local/cuda/lib64
EXE	        = matrix_vector_mult
OBJ	        = matrix_vector_mult.o

default: $(EXE)

matrix_vector_mult.o: matrix_vector_mult.cu
	$(NVCC) -c -o $@ matrix_vector_mult.cu $(NVCC_FLAGS)

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS)

clean:
	rm -rf *.o $(EXE)
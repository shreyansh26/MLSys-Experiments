
NVCC        = /usr/bin/nvcc
NVCC_FLAGS  = -O3 -I/usr/include
LD_FLAGS    = -lcudart -L/usr/lib64
EXE	        = matmul
OBJ	        = matmul.o

default: $(EXE)

matmul.o: matmul.cu
	$(NVCC) -c -o $@ matmul.cu $(NVCC_FLAGS)

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS)

clean:
	rm -rf *.o $(EXE)
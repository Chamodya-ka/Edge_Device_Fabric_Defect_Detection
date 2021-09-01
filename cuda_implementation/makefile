CC = "nvcc"

PROJECT = cuda_implementation
SRC = main.cu Image.cu ImageLoader.cu GLCMComputation.cu
OUT = build/
LIBS = `pkg-config opencv --cflags --libs`

$(PROJECT) : $(SRC)
	$(CC) $(SRC) -o $(OUT)$(PROJECT) $(LIBS)
	
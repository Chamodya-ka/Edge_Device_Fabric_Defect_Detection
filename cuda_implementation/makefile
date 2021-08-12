CC = "nvcc"

PROJECT = cuda_implementation
SRC = main.cu ImageLoader.cu Image.cu 
OUT = build/
LIBS = `pkg-config opencv --cflags --libs`

$(PROJECT) : $(SRC)
	$(CC) $(SRC) -o $(OUT)$(PROJECT) $(LIBS)
	
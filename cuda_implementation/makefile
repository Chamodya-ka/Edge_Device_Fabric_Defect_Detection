CC = "nvcc"
FLAGS = -lineinfo 
PROJECT = cuda_implementation
SRC = main.cu Image.cu ImageLoader.cu GLCMComputation.cu  FeatureComputation.cu
OUT = build/
LIBS = `pkg-config opencv --cflags --libs`

$(PROJECT) : $(SRC)
	$(CC) $(FLAGS) $(SRC) -o $(OUT)$(PROJECT) $(LIBS)
	

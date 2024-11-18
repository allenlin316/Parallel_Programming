NVCC = nvcc
NVFLAGS = -std=c++11 -O3 -dc -arch=sm_86

TARGET = hw4

.PHONY: all clean

all: $(TARGET)

$(TARGET): sha256.o hw4.o
	$(NVCC) -o $(TARGET) sha256.o hw4.o

sha256.o: sha256.cu sha256.h
	$(NVCC) $(NVFLAGS) -c sha256.cu

hw4.o: hw4.cu sha256.h
	$(NVCC) $(NVFLAGS) -c hw4.cu

clean:
	rm -f $(TARGET) *.o
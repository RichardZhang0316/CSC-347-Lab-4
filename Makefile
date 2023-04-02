all: NBody.cu
	nvcc -o NBody NBody.cu
clean:
	rm -f NBody
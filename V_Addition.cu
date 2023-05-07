#include<stdlib.h>
#include<stdio.h>
#include<time.h>
#include<iostream>

using namespace std;

#define WARP_SIZE 32
#define WARP_COUNT 16
#define BLOCK_COUNT 13

double *u_CPU, *v_CPU, *w_CPU;
double *u_GPU, *v_GPU, *w_GPU;

void allocateAndInitializeInputVectors(int vectorLength);
void allocateOutputVector(int vectorLength);
void die(const char *error); 
void check_error(cudaError e);
 
__global__ void vectorAdditionInGPU(const double *u, const double *v, 
		double *w, int vectorLength);

int main(int argc, char **argv) {

	int vectorLength = (argc > 1) ? atoi(argv[1]) : 100000;

	clock_t start = clock();	
	allocateAndInitializeInputVectors(vectorLength);
	allocateOutputVector(vectorLength);
	clock_t end = clock();
        double elapsed = (end - start) / (double) CLOCKS_PER_SEC;
        cout << "Allocation time: " << elapsed << " seconds \n";

	start = clock();
	int threadsPerBlock = WARP_SIZE * WARP_COUNT;
	vectorAdditionInGPU<<< BLOCK_COUNT, threadsPerBlock >>>(u_GPU, v_GPU, w_GPU, vectorLength);
	cudaDeviceSynchronize();
	check_error(cudaGetLastError());
	size_t sizeofW = vectorLength * sizeof(double);
	w_CPU = (double *) malloc(sizeofW);
	check_error(cudaMemcpy(w_CPU, w_GPU, sizeofW, cudaMemcpyDeviceToHost));
	end = clock();
	elapsed = (end - start) / (double) CLOCKS_PER_SEC;
	cout << "Execution time: " << elapsed << " seconds \n";
	return 0;
}

void allocateAndInitializeInputVectors(int vectorLength) {

	size_t vectorSize = vectorLength * sizeof(double);
	u_CPU = (double*) malloc(vectorSize);
	check_error(cudaMalloc((void **) &u_GPU, vectorSize));
	srand(time(NULL));
  	for (int i = 0; i < vectorLength; i++) {
		u_CPU[i] = (rand() % 5); 
	}
	check_error(cudaMemcpyAsync(u_GPU, u_CPU, vectorSize, cudaMemcpyHostToDevice, 0));
	v_CPU = (double*) malloc(vectorSize);
	check_error(cudaMalloc((void **) &v_GPU, vectorSize));
  	for (int i = 0; i < vectorLength; i++) {
		v_CPU[i] = (rand() % 10); 
	}
	check_error(cudaMemcpyAsync(v_GPU, v_CPU, vectorSize, cudaMemcpyHostToDevice, 0));
}

void allocateOutputVector(int vectorLength) {
	size_t sizeofW = vectorLength * sizeof(double);
	check_error(cudaMalloc((void **) &w_GPU, sizeofW));
}

void die(const char *error) {
        printf("%s", error);
        exit(1);
}

void check_error(cudaError e) {
        if (e != cudaSuccess) {
                printf("\nCUDA error: %s\n", cudaGetErrorString(e));
                exit(1);
        }
}

__global__ void vectorAdditionInGPU(const double *u, const double *v, double *w, int vectorLength) {

	int startIndex = blockIdx.x + threadIdx.x;
	int stride = BLOCK_COUNT * WARP_COUNT * WARP_SIZE;

	for (int i = startIndex; i < vectorLength; i += stride) {
		w[i] = u[i] + v[i];
	}
}

#include<stdlib.h>
#include<stdio.h>
#include<time.h>
#include<iostream>

using namespace std;

#define WARP_SIZE 32
#define WARP_COUNT 16
#define BLOCK_COUNT 13

double *u_CPU, *v_CPU;
double *u_GPU, *v_GPU;

void allocateAndInitializeInputVectors(int vectorLength);
void die(const char *error); 
void check_error(cudaError e);
 
__global__ void computePairwiseMultInGPU(double *u, const double *v, int vectorLength);
__global__ void computeSumInGPU(double *u, int vectorLength);
__device__ void computeWarpSum(double *elements, int threadId);

int main(int argc, char **argv) {

	int vectorLength = (argc > 1) ? atoi(argv[1]) : 100000;

	clock_t start = clock();	
	allocateAndInitializeInputVectors(vectorLength);
	clock_t end = clock();
        double elapsed = (end - start) / (double) CLOCKS_PER_SEC;
        cout << "Allocation time: " << elapsed << " seconds \n";
	start = clock();
	int threadsPerBlock = WARP_SIZE * WARP_COUNT;
	computePairwiseMultInGPU<<< BLOCK_COUNT, threadsPerBlock >>>(u_GPU, v_GPU, vectorLength);
	computeSumInGPU<<< 1, threadsPerBlock >>>(u_GPU, vectorLength);
	cudaDeviceSynchronize();

	check_error(cudaGetLastError());
	double dotProduct = 0;
	check_error(cudaMemcpy(&dotProduct, u_GPU, sizeof(double), cudaMemcpyDeviceToHost));
	end = clock();
	elapsed = (end - start) / (double) CLOCKS_PER_SEC;
	cout << "Dot product of the vectors: " << dotProduct << "\n";
	cout << "Execution time: " << elapsed << " seconds \n";

	return 0;
}

void allocateAndInitializeInputVectors(int vectorLength) {

	size_t vectorSize = vectorLength * sizeof(double);
	u_CPU = (double*) malloc(vectorSize);
	check_error(cudaMalloc((void **) &u_GPU, vectorSize));
	srand(time(NULL));
  	for (int i = 0; i < vectorLength; i++) {
		u_CPU[i] = 6 / (1.00 + (rand() % 5)); 
	}
	check_error(cudaMemcpyAsync(u_GPU, u_CPU, vectorSize, cudaMemcpyHostToDevice, 0));
	v_CPU = (double*) malloc(vectorSize);
	check_error(cudaMalloc((void **) &v_GPU, vectorSize));
  	for (int i = 0; i < vectorLength; i++) {
		v_CPU[i] = 10 / (1.00 + rand() % 10); 
	}
	check_error(cudaMemcpyAsync(v_GPU, v_CPU, vectorSize, cudaMemcpyHostToDevice, 0));
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

__global__ void computePairwiseMultInGPU(double *u, const double *v, int vectorLength) {
	int startIndex = blockIdx.x + threadIdx.x;
	int stride = BLOCK_COUNT * WARP_COUNT * WARP_SIZE;
	for (int i = startIndex; i < vectorLength; i += stride) {
		u[i] = u[i] * v[i];
	}

}
__global__ void computeSumInGPU(double *u, int vectorLength) {
	__shared__ double elements_to_sum[WARP_COUNT][WARP_SIZE];
	__shared__ double partial_sums[WARP_COUNT];
	int threadId = threadIdx.x % WARP_SIZE;
        int warpId = threadIdx.x / WARP_SIZE;
	for (int i = 0; i < WARP_COUNT; i++) {
		if (threadId == 0) {
			partial_sums[warpId] = 0.0;
		}
	}
	for (int i = WARP_SIZE * warpId; i < vectorLength; i += WARP_SIZE * WARP_COUNT) {
		int startIndex = i;
		int endIndex = (vectorLength < i + WARP_SIZE) 
			? startIndex + WARP_SIZE - 1 : vectorLength - 1;
		if (startIndex + threadId <= endIndex) {
			elements_to_sum[warpId][threadId] = u[startIndex + threadId];
		} else {
			elements_to_sum[warpId][threadId] = 0;
		}
		computeWarpSum(elements_to_sum[warpId], threadId);
		if (threadId == 0) {
			partial_sums[warpId] += elements_to_sum[warpId][0];
		}
	}
	__syncthreads();
	if (warpId == 0 && threadId == 0) {
		for (int i = 1; i < WARP_COUNT; i++) {
			partial_sums[0] += partial_sums[i];
		}
		u[0] = partial_sums[0];
	}
}

__device__ void computeWarpSum(double *elements, int threadId) {

	if (threadId < 16) elements[threadId * 2] = elements[threadId * 2] + elements[threadId * 2 + 1];
	if (threadId < 8) elements[threadId * 4] = elements[threadId * 4] + elements[threadId * 4 + 2];
	if (threadId < 4) elements[threadId * 8] = elements[threadId * 8] + elements[threadId * 8 + 4];
	if (threadId < 2) elements[threadId * 16] = elements[threadId * 16] + elements[threadId * 16 + 8];
	if (threadId == 0) elements[0] = elements[0] + elements[16];
}

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

int const BLOCK_SIZE = 32;

typedef unsigned char uchar;

__global__ void kernel(uchar *inputMatrix, int pixelIncrementation, uchar *outputMatrix) {
	int noCol = blockIdx.x  * blockDim.x + threadIdx.x;
	int noRow = blockIdx.y  * blockDim.y + threadIdx.y;
	int dim = blockDim.x * gridDim.x;
	int cudaIndex = noRow * dim + noCol;
	outputMatrix[cudaIndex] = inputMatrix[cudaIndex] + pixelIncrementation;
}
extern "C" void ParallelBlackAndWhite(uchar *pMatA, int pixelIncrementation, uchar *pMatR, dim3 matrixDimension)
{
	uchar *inputMatrixGrid, *outputMatrixGrid;
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(BLOCK_SIZE, BLOCK_SIZE);
	// Allouer l'espace memoire des 2 matrices sur la carte GPU 
	size_t memSize = matrixDimension.x * matrixDimension.y * sizeof(uchar);
	cudaMalloc(&inputMatrixGrid, memSize);
	cudaMalloc(&outputMatrixGrid, memSize);

	// Copier de la matrice A dans la memoire du GPU 
	cudaMemcpy(inputMatrixGrid, pMatA, memSize, cudaMemcpyHostToDevice);

	// Partir le kernel
	kernel<<<dimGrid, dimBloc>>>(inputMatrixGrid, pixelIncrementation, outputMatrixGrid);

	// Transfert de la matrice résultat 
	cudaMemcpy(pMatR, outputMatrixGrid, memSize, cudaMemcpyDeviceToHost);

	cudaFree(inputMatrixGrid);
	cudaFree(outputMatrixGrid);
}

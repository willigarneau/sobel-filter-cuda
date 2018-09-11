#include "cuda_runtime.h"
#include "device_launch_parameters.h"

int const BLOCK_SIZE = 32;

typedef unsigned char uchar;

__global__ void ParallelBlackAndWhite_kernel(uchar *inputMatrix, int pixelIncrementation, uchar *outputMatrix) {
	int noCol = blockIdx.x  * blockDim.x + threadIdx.x;
	int noRow = blockIdx.y  * blockDim.y + threadIdx.y;
	int dim = blockDim.x * gridDim.x;
	int cudaIndex = noRow * dim + noCol;
	outputMatrix[cudaIndex] = inputMatrix[cudaIndex] + pixelIncrementation;
}

extern "C" void ParallelBlackAndWhite(uchar *inputMatrixPointer, int pixelIncrementation, uchar *outputMatrixPointer, dim3 matrixDimension)
{
	uchar *inputMatrixGrid, *outputMatrixGrid;
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(BLOCK_SIZE, BLOCK_SIZE);
	// Allouer l'espace memoire des 2 matrices sur la carte GPU 
	size_t memSize = matrixDimension.x * matrixDimension.y * sizeof(uchar);
	cudaMalloc(&inputMatrixGrid, memSize);
	cudaMalloc(&outputMatrixGrid, memSize);

	// Copier de la matrice A dans la memoire du GPU
	cudaMemcpy(inputMatrixGrid, inputMatrixPointer, memSize, cudaMemcpyHostToDevice);

	// Partir le kernel
	ParallelBlackAndWhite_kernel<<<dimGrid, dimBlock>>>(inputMatrixGrid, pixelIncrementation, outputMatrixGrid);

	// Transfert de la matrice résultat 
	cudaMemcpy(outputMatrixPointer, outputMatrixGrid, memSize, cudaMemcpyDeviceToHost);

	cudaFree(inputMatrixGrid);
	cudaFree(outputMatrixGrid);
}

__global__ void xGradient(int* gradientX, uchar* channel, dim3 matrixDimension) {
	int const Gx[3][3] = { // valeur approximative de la dérivée horizontale
		{ -1, 0, 1 },
		{ -2, 0, 2 },
		{ -1, 0, 1 },
	};
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index == 0) return;
	gradientX[index] =
		Gx[0][0] * channel[index - 1] +
		Gx[1][0] * channel[index] +
		Gx[2][0] * channel[index + 1] +
		Gx[0][1] * channel[index + matrixDimension.x - 1] +
		Gx[1][1] * channel[index + matrixDimension.x] +
		Gx[2][1] * channel[index + matrixDimension.x - 1] +
		Gx[0][2] * channel[index + 2 * matrixDimension.x - 1] +
		Gx[1][2] * channel[index + 2 * matrixDimension.x] +
		Gx[2][2] * channel[index + 2 * matrixDimension.x + 1];
	return;
}

__global__ void yGradient(int* gradientY, uchar* channel, dim3 matrixDimension) {
	int const Gy[3][3] = { // valeur approximative de la dérivée verticale
		{ -1, 0, 1 },
		{ -2, 0, 2 },
		{ -1, 0, 1 },
	};
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index == 0) return;
	gradientY[index] =
		Gy[0][0] * channel[index - 1] +
		Gy[1][0] * channel[index] +
		Gy[2][0] * channel[index + 1] +
		Gy[0][1] * channel[index + matrixDimension.x - 1] +
		Gy[1][1] * channel[index + matrixDimension.x] +
		Gy[2][1] * channel[index + matrixDimension.x - 1] +
		Gy[0][2] * channel[index + 2 * matrixDimension.x - 1] +
		Gy[1][2] * channel[index + 2 * matrixDimension.x] +
		Gy[2][2] * channel[index + 2 * matrixDimension.x + 1];
	return;
}

__global__ void parallelSobel(uchar *outputChannel, int* xGradient, int* yGradient) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int approxGradient = int(xGradient[index] + yGradient[index]);
	if (approxGradient > 255) { approxGradient = 0; }
	outputChannel[index] = approxGradient;
	return;
}


extern "C" void ParallelSobelFilter(uchar *inputMatrixPointer, uchar *outputMatrixPointer, dim3 matrixDimension) {
	uchar *inputMatrixGrid, *outputMatrixGrid;
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(BLOCK_SIZE, BLOCK_SIZE);
	int* gradientX;
	int* gradientY;

	cudaMalloc(&gradientX, (matrixDimension.x * matrixDimension.y) * sizeof(int));
	cudaMalloc(&gradientY, (matrixDimension.x * matrixDimension.y) * sizeof(int));
	// Allouer l'espace memoire des 2 matrices sur la carte GPU 
	size_t memSize = matrixDimension.x * matrixDimension.y * sizeof(uchar);
	cudaMalloc(&inputMatrixGrid, memSize);
	cudaMalloc(&outputMatrixGrid, memSize);

	// Copier de la matrice A dans la memoire du GPU
	cudaMemcpy(inputMatrixGrid, inputMatrixPointer, memSize, cudaMemcpyHostToDevice);

	// Partir le kernel
	xGradient<<<dimGrid, dimBlock>>>(gradientX, inputMatrixGrid, matrixDimension);
	yGradient<<<dimGrid, dimBlock>>>(gradientY, inputMatrixGrid, matrixDimension);
	parallelSobel<<<dimGrid, dimBlock>>>(outputMatrixGrid, gradientX, gradientY);

	// Transfert de la matrice résultat 
	cudaMemcpy(outputMatrixPointer, outputMatrixGrid, memSize, cudaMemcpyDeviceToHost);

	cudaFree(&xGradient);
	cudaFree(&yGradient);
	cudaFree(inputMatrixGrid);
	cudaFree(outputMatrixGrid);
}

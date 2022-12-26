#include <cublas_v2.h>
#include <curand.h>
#include <iostream>

void cpuSgemmStridedBatched(
	bool transB, bool transA,
	int CCols, int CRows, int AColsBRows,
	const float* alpha,
	float* B, int ColsB, int SizeB,
	float* A, int ColsA, int SizeA,
	const float* beta,
	float* C, int ColsC, int SizeC,
	int batchCount)
{
	for (int b = 0; b < batchCount; b++)
	{
		for (int m = 0; m < CCols; m++)
			for (int n = 0; n < CRows; n++)
			{
				float sum = 0;
				for (int k = 0; k < AColsBRows; k++)
					sum += (transA ? A[k * ColsA + n] : A[n * ColsA + k]) * (transB ? B[m * ColsB + k] : B[k * ColsB + m]);
				C[n * ColsC + m] = *alpha * sum + *beta * C[n * ColsC + m];
			}
		A += SizeA;
		B += SizeB;
		C += SizeC;
	}
}

int main()
{
	// Matrix dimensions
	const int RowsA = 3, ColsA = 4, ColsB = 5;

	// Allocate host matrices
	float* h_A = new float[ColsA * RowsA];
	float* h_B = new float[ColsB * ColsA];
	float* h_C = new float[ColsB * RowsA];
	float* h_C2 = new float[ColsB * RowsA];

	// Allocate device matrices
	float* d_A, * d_B, * d_C;
	cudaMalloc(&d_A, ColsA * RowsA * sizeof(float));
	cudaMalloc(&d_B, ColsB * ColsA * sizeof(float));
	cudaMalloc(&d_C, ColsB * RowsA * sizeof(float));

	// Initialize host matrices
	for (int i = ColsA * RowsA; i--;) h_A[i] = i;
	for (int i = ColsB * ColsA; i--;) h_B[i] = i;
	for (int i = ColsB * RowsA; i--;) h_C[i] = h_C2[i] = 0;

	// Copy host matrices to device
	cudaMemcpy(d_A, h_A, ColsA * RowsA * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, ColsB * ColsA * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_C, h_C, ColsB * RowsA * sizeof(float), cudaMemcpyHostToDevice);

	// Create CUBLAS handle
	cublasHandle_t handle;
	cublasCreate(&handle);

	// Perform matrix multiplication
	const float alpha = 1.0f;
	const float beta = 0.0f;
	cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, ColsB, RowsA, ColsA, &alpha, d_B, ColsB, ColsB * ColsA, d_A, ColsA, ColsA * RowsA, &beta, d_C, ColsB, ColsB * RowsA, 1);
	cpuSgemmStridedBatched(false, false, ColsB, RowsA, ColsA, &alpha, h_B, ColsB, ColsB * ColsA, h_A, ColsA, ColsA * RowsA, &beta, h_C, ColsB, ColsB * RowsA, 1);

	// Copy result from device to host
	cudaMemcpy(h_C2, d_C, ColsB * RowsA * sizeof(float), cudaMemcpyDeviceToHost);

	std::cout << "Matrix A:" << std::endl;
	for (int i = 0; i < RowsA; i++)
	{
		for (int j = 0; j < ColsA; j++)
			std::cout << h_A[i * ColsA + j] << " ";
		std::cout << std::endl;
	}
	std::cout << std::endl;

	std::cout << "Matrix B:" << std::endl;
	for (int i = 0; i < ColsA; i++)
	{
		for (int j = 0; j < ColsB; j++)
			std::cout << h_B[i * ColsB + j] << " ";
		std::cout << std::endl;
	}
	std::cout << std::endl;

	std::cout << "Matrix C (computed using cuBLAS):" << std::endl;
	for (int i = 0; i < RowsA; i++)
	{
		for (int j = 0; j < ColsB; j++)
			std::cout << h_C2[i * ColsB + j] << " ";
		std::cout << std::endl;
	}
	std::cout << std::endl;

	std::cout << "Matrix C (computed on the CPU):" << std::endl;
	for (int i = 0; i < RowsA; i++)
	{
		for (int j = 0; j < ColsB; j++)
			std::cout << h_C[i * ColsB + j] << " ";
		std::cout << std::endl;
	}
	std::cout << std::endl;

	// Destroy CUBLAS handle
	cublasDestroy(handle);

	// Free device matrices
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	// Free host matrices
	delete[] h_A;
	delete[] h_B;
	delete[] h_C;
	delete[] h_C2;

	return 0;
}
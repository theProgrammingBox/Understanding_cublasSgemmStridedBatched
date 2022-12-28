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
	for (int b = batchCount; b--;)
	{
		for (int m = CCols; m--;)
			for (int n = CRows; n--;)
			{
				float sum = 0;
				for (int k = AColsBRows; k--;)
				{
					sum += (transA ? A[k * ColsA + n] : A[n * ColsA + k]) * (transB ? B[m * ColsB + k] : B[k * ColsB + m]);
					
					/*if (m == 1 && n == 0)
						std::cout << (transA ? A[k * ColsA + n] : A[n * ColsA + k]) << " * " << (transB ? B[m * ColsB + k] : B[k * ColsB + m]) << " = " << (transA ? A[k * ColsA + n] : A[n * ColsA + k]) * (transB ? B[m * ColsB + k] : B[k * ColsB + m]) << std::endl;
					*/
				}
					C[n * ColsC + m] = *alpha * sum + *beta * C[n * ColsC + m];
			}
		A += SizeA;
		B += SizeB;
		C += SizeC;
	}
}

using std::cout;
using std::endl;

int main()
{
	cublasHandle_t handle;
	cublasCreate(&handle);
	
	curandGenerator_t gen;
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen, 0);
	
	const float ONE = 1.0f;
	const float ZERO = 0.0f;

	const uint32_t MatrixCols = 4, MatrixRows = 4;
	const uint32_t Focuses = 3, QueryDim = 2;
	
	float* matrix;
	cudaMalloc(&matrix, MatrixCols * MatrixRows * sizeof(float));
	float* focus = matrix + (MatrixRows - Focuses) * MatrixCols;
	float* keys = matrix + QueryDim;
	float* res;
	cudaMalloc(&res, MatrixRows * Focuses * sizeof(float));

	float* cpuMatrix = new float[MatrixCols * MatrixRows];
	float* cpuFocus = cpuMatrix + (MatrixRows - Focuses) * MatrixCols;
	float* cpuKeys = cpuMatrix + QueryDim;
	float* cpuRes = new float[MatrixRows * Focuses];

	curandGenerateUniform(gen, matrix, MatrixCols * MatrixRows);
	cudaMemcpy(cpuMatrix, matrix, MatrixCols * MatrixRows * sizeof(float), cudaMemcpyDeviceToHost);

	// print matrix
	cout << "Matrix:" << endl;
	for (int i = 0; i < MatrixRows; i++)
	{
		for (int j = 0; j < MatrixCols; j++)
			cout << cpuMatrix[i * MatrixCols + j] << " ";
		cout << endl;
	}
	cout << endl;

	// print focus
	cout << "Focus:" << endl;
	for (int i = 0; i < Focuses; i++)
	{
		for (int j = 0; j < QueryDim; j++)
			cout << cpuFocus[i * MatrixCols + j] << " ";
		cout << endl;
	}
	cout << endl;

	// print keys
	cout << "Keys:" << endl;
	for (int i = 0; i < MatrixRows; i++)
	{
		for (int j = 0; j < QueryDim; j++)
			cout << cpuKeys[i * MatrixCols + j] << " ";
		cout << endl;
	}
	cout << endl;
	
	cpuSgemmStridedBatched(
		true, false,
		MatrixRows, Focuses, QueryDim,
		&ONE,
		cpuKeys, MatrixCols, 0,
		cpuFocus, MatrixCols, 0,
		&ZERO,
		cpuRes, MatrixRows, 0,
		1);

	cublasSgemmStridedBatched(
		handle,
		CUBLAS_OP_T, CUBLAS_OP_N,
		MatrixRows, Focuses, QueryDim,
		&ONE,
		keys, MatrixCols, 0,
		focus, MatrixCols, 0,
		&ZERO,
		res, MatrixRows, 0,
		1);
	
	// print result
	cout << "Result:" << endl;
	for (int i = 0; i < Focuses; i++)
	{
		for (int j = 0; j < MatrixRows; j++)
			cout << cpuRes[i * MatrixRows + j] << " ";
		cout << endl;
	}
	cout << endl;

	memset(cpuRes, 0, MatrixRows * Focuses * sizeof(float));
	cout << "GPU Result:" << endl;
	cudaMemcpy(cpuRes, res, MatrixRows * Focuses * sizeof(float), cudaMemcpyDeviceToHost);
	for (int i = 0; i < Focuses; i++)
	{
		for (int j = 0; j < MatrixRows; j++)
			cout << cpuRes[i * MatrixRows + j] << " ";
		cout << endl;
	}
	cout << endl;
	
	return 0;
	
	
	/*
	// Matrix dimensions
	const int RowsA = 3, ColsA = 4, ColsB = 5;
	const float alpha = 1.0f;
	const float beta = 0.0f;

	// Allocate host matrices
	float* h_A = new float[ColsA * RowsA];
	float* h_B = new float[ColsB * ColsA];
	float* h_C = new float[ColsB * RowsA];
	float* h_R;

	// Allocate device matrices
	float* d_A, * d_B, * d_C;
	cudaMalloc(&d_A, ColsA * RowsA * sizeof(float));
	cudaMalloc(&d_B, ColsB * ColsA * sizeof(float));
	cudaMalloc(&d_C, ColsB * RowsA * sizeof(float));

	// Create CUBLAS handle
	cublasHandle_t handle;
	cublasCreate(&handle);



	// Initialize host matrices A and B
	for (int i = ColsA * RowsA; i--;) h_A[i] = i;
	for (int i = ColsB * ColsA; i--;) h_B[i] = i;

	// Copy host matrices to device
	cudaMemcpy(d_A, h_A, ColsA * RowsA * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, ColsB * ColsA * sizeof(float), cudaMemcpyHostToDevice);

	// Perform matrix multiplication
	cpuSgemmStridedBatched(false, false, ColsB, RowsA, ColsA, &alpha, h_B, ColsB, ColsB * ColsA, h_A, ColsA, ColsA * RowsA, &beta, h_C, ColsB, ColsB * RowsA, 1);
	cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, ColsB, RowsA, ColsA, &alpha, d_B, ColsB, ColsB * ColsA, d_A, ColsA, ColsA * RowsA, &beta, d_C, ColsB, ColsB * RowsA, 1);

	// Copy result from device to host
	h_R = new float[ColsB * RowsA];
	cudaMemcpy(h_R, d_C, ColsB * RowsA * sizeof(float), cudaMemcpyDeviceToHost);

	// Print result
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
			std::cout << h_R[i * ColsB + j] << " ";
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

	delete[] h_R;



	// Initialize host matrices A and C
	for (int i = ColsA * RowsA; i--;) h_A[i] = i;
	for (int i = ColsB * RowsA; i--;) h_C[i] = i;

	// Copy host matrices to device
	cudaMemcpy(d_A, h_A, ColsA * RowsA * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_C, h_C, ColsB * RowsA * sizeof(float), cudaMemcpyHostToDevice);

	// Perform matrix multiplication
	cpuSgemmStridedBatched(false, true, ColsB, ColsA, RowsA, &alpha, h_C, ColsB, ColsB * RowsA, h_A, ColsA, ColsA * RowsA, &beta, h_B, ColsB, ColsB * ColsA, 1);
	cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_T, ColsB, ColsA, RowsA, &alpha, d_C, ColsB, ColsB * RowsA, d_A, ColsA, ColsA * RowsA, &beta, d_B, ColsB, ColsB * ColsA, 1);

	// Copy result from device to host
	h_R = new float[ColsB * ColsA];
	cudaMemcpy(h_R, d_B, ColsB * ColsA * sizeof(float), cudaMemcpyDeviceToHost);

	// Print result
	std::cout << "Matrix A:" << std::endl;
	for (int i = 0; i < RowsA; i++)
	{
		for (int j = 0; j < ColsA; j++)
			std::cout << h_A[i * ColsA + j] << " ";
		std::cout << std::endl;
	}
	std::cout << std::endl;

	std::cout << "Matrix C:" << std::endl;
	for (int i = 0; i < RowsA; i++)
	{
		for (int j = 0; j < ColsB; j++)
			std::cout << h_C[i * ColsB + j] << " ";
		std::cout << std::endl;
	}
	std::cout << std::endl;

	std::cout << "Matrix B (computed using cuBLAS):" << std::endl;
	for (int i = 0; i < ColsA; i++)
	{
		for (int j = 0; j < ColsB; j++)
			std::cout << h_R[i * ColsB + j] << " ";
		std::cout << std::endl;
	}
	std::cout << std::endl;

	std::cout << "Matrix B (computed on the CPU):" << std::endl;
	for (int i = 0; i < ColsA; i++)
	{
		for (int j = 0; j < ColsB; j++)
			std::cout << h_B[i * ColsB + j] << " ";
		std::cout << std::endl;
	}
	std::cout << std::endl;

	delete[] h_R;



	// Initialize host matrices C and B
	for (int i = ColsB * RowsA; i--;) h_C[i] = i;
	for (int i = ColsB * ColsA; i--;) h_B[i] = i;

	// Copy host matrices to device
	cudaMemcpy(d_C, h_C, ColsB * RowsA * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, ColsB * ColsA * sizeof(float), cudaMemcpyHostToDevice);

	// Perform matrix multiplication
	cpuSgemmStridedBatched(true, false, ColsA, RowsA, ColsB, &alpha, h_B, ColsB, ColsB * ColsA, h_C, ColsB, ColsB * RowsA, &beta, h_A, ColsA, ColsA * RowsA, 1);
	cublasSgemmStridedBatched(handle, CUBLAS_OP_T, CUBLAS_OP_N, ColsA, RowsA, ColsB, &alpha, d_B, ColsB, ColsB * ColsA, d_C, ColsB, ColsB * RowsA, &beta, d_A, ColsA, ColsA * RowsA, 1);

	// Copy result from device to host
	h_R = new float[ColsA * RowsA];
	cudaMemcpy(h_R, d_A, ColsA * RowsA * sizeof(float), cudaMemcpyDeviceToHost);

	// Print result
	std::cout << "Matrix C:" << std::endl;
	for (int i = 0; i < RowsA; i++)
	{
		for (int j = 0; j < ColsB; j++)
			std::cout << h_C[i * ColsB + j] << " ";
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

	std::cout << "Matrix A (computed using cuBLAS):" << std::endl;
	for (int i = 0; i < RowsA; i++)
	{
		for (int j = 0; j < ColsA; j++)
			std::cout << h_R[i * ColsA + j] << " ";
		std::cout << std::endl;
	}
	std::cout << std::endl;

	std::cout << "Matrix A (computed on the CPU):" << std::endl;
	for (int i = 0; i < RowsA; i++)
	{
		for (int j = 0; j < ColsA; j++)
			std::cout << h_A[i * ColsA + j] << " ";
		std::cout << std::endl;
	}
	std::cout << std::endl;

	delete[] h_R;



	// Destroy CUBLAS handle
	cublasDestroy(handle);

	// Free device memory
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	// Free host memory
	delete[] h_A;
	delete[] h_B;
	delete[] h_C;

	return 0;
	*/
}
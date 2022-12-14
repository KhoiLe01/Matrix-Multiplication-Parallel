#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>
#include <math.h>
#include <iostream>
#include <unistd.h>
#include <sys/syscall.h>
#define gettid() syscall(SYS_gettid)

int NUM_THREADS = 7;

// Strassen's method of matrix multiplication only work on matrices
// that has power-of-two size, so for any n, we have to find the
// closest power-of-two ceiling of n.

// n respresents the size of the matrix, while ceiling (the power-of-two ceiling of n)
// represents the actual size of the matrix.

// Struct to store matrix A, B, size and extended size
typedef struct
{
	int **A;
	int **B;
	int n;
	int ceiling;
} matrixMultStrassen;

// Struct to store a matrix with additional info
typedef struct
{
	int **A;
	int n;
	int ceiling;
} matrixS;

// Naive method of multiplying matrices.
int **naive_matrix_mult(int **data_ptr, int **data_ptr2, int n)
{
	int **matrix = (int **)malloc(n * sizeof(int *));
	for (int i = 0; i < n; i++)
	{
		matrix[i] = (int *)malloc(n * sizeof(int));
		for (int j = 0; j < n; j++)
		{
			matrix[i][j] = 0;
		}
	}
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			for (int k = 0; k < n; k++)
			{
				matrix[i][j] += data_ptr[i][k] * data_ptr2[k][j];
			}
		}
	}
	return matrix;
}

int ceil_log2(int n)
{
	/*
	This function return the power-of-two ceiling of any
	integer n.
	*/
	return (int)pow(2, ceil(log(n) / log(2)));
}

int **initMatrix(int n, int ceiling)
{
	/*
	Init the matrix based on n and the power-of-twp ceiling
	of n. Only the first nxn entries on the top left is randomly
	initiated. Other entries are initiated as 0.
	*/
	int **matrix = (int **)malloc(ceiling * sizeof(int *));
	for (int i = 0; i < ceiling; i++)
	{
		matrix[i] = (int *)malloc(ceiling * sizeof(int));
		for (int j = 0; j < ceiling; j++)
		{
			if (j < n && i < n)
			{
				matrix[i][j] = rand() % 100;
			}
			else
			{
				matrix[i][j] = 0;
			}
		}
	}
	return matrix;
}

int **initEmptyMatrix(int ceiling)
{
	/*
	Init the whole empty matrix based on the ceiling
	*/
	int **matrix = (int **)malloc(ceiling * sizeof(int *));
	for (int i = 0; i < ceiling; i++)
	{
		matrix[i] = (int *)malloc(ceiling * sizeof(int));
	}
	return matrix;
}

matrixMultStrassen *initMatrixMultStrassen(int n)
{
	/*
	Init struct matrix MultStrassen based on matrix size n.
	*/
	int ceiling = ceil_log2(n);
	matrixMultStrassen *M = (matrixMultStrassen *)malloc(sizeof(matrixMultStrassen));
	M->A = initEmptyMatrix(ceiling);
	M->B = initEmptyMatrix(ceiling);
	M->n = n;
	M->ceiling = ceiling;
	return M;
}

matrixS *initMatrixS(int n)
{
	/*
	Init struct matrixS based on matrix size n
	*/
	int ceiling = ceil_log2(n);
	matrixS *M = (matrixS *)malloc(sizeof(matrixS));
	M->A = initEmptyMatrix(ceiling);
	M->n = n;
	M->ceiling = ceiling;
	return M;
}

void freeMatrix(int **matrix, int ceiling)
{
	/*
	Free a 2d matrix
	*/
	for (int i = 0; i < ceiling; i++)
	{
		free(matrix[i]);
	}
	free(matrix);
}

void freeMatrixMultStrassen(matrixMultStrassen *M, int ceiling)
{
	/*
	Free the struct matrixMultStrassen
	*/
	freeMatrix(M->A, ceiling);
	freeMatrix(M->B, ceiling);
	free(M);
}

void printMatrix(int **matrix, int n)
{
	/*
	Print the matrix of the original size
	*/
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			std::cout << matrix[i][j] << " ";
		}
		std::cout << std::endl;
	}
}

int **addMatrix(int **matrix1, int **matrix2, int n, int ceiling)
{
	/*
	Add two matrices: matrix1 + matrix2
	*/
	int **ans = initMatrix(n, ceiling);
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			ans[i][j] = matrix1[i][j] + matrix2[i][j];
		}
	}
	return ans;
}

int **subtractMatrix(int **matrix1, int **matrix2, int n, int ceiling)
{
	/*
	Subtract two matrices: matrix1 - matrix2
	*/
	int **ans = initMatrix(n, ceiling);
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			ans[i][j] = matrix1[i][j] - matrix2[i][j];
		}
	}
	return ans;
}

int **getPartialMatrix(int **matrix, int topleft_row, int topleft_col, int bottomright_row, int bottomright_col)
{
	/*
	Get only a part of the matrix. The partial matrix is decided by
	the topleft location and bottomright location
	*/
	int n = bottomright_col - topleft_col;
	int ceiling = ceil_log2(n);
	int **ans = initMatrix(n, ceiling);
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			ans[i][j] = matrix[topleft_row + i][topleft_col + j];
		}
	}
	return ans;
}

void copyMatrix(int **source, int size, int **destination, int starting_row, int starting_col)
{
	/*
	Copy a submatrix from source to destination
	*/
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			destination[starting_row + i][starting_col + j] = source[i][j];
		}
	}
}

int **deepCopyMatrix(int **source, int n, int ceiling)
{
	/*
	Create a deep copy matrix of the source matrix
	*/
	int **ans = initMatrix(n, ceiling);
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			ans[i][j] = source[i][j];
		}
	}
	return ans;
}

void *recursiveMul(void *mat)
{
	/*
	Function for each thread to run to recursively multiply
	matrices using Strassen method
	*/
	matrixMultStrassen *M = (matrixMultStrassen *)mat;
	matrixS *c = initMatrixS(M->n);

	if (M->n == 1)
	{
		c->A[0][0] = M->A[0][0] * M->B[0][0];
	}
	else
	{
		int n = M->n;
		int ceilN = M->ceiling;
		int ceilSubSize = ceilN / 2;
		int subSize = ceilSubSize;

		// Divide the matrix A and B into four sub-matrices
		int **a11 = getPartialMatrix(M->A, 0, 0, ceilSubSize, ceilSubSize);
		int **a12 = getPartialMatrix(M->A, 0, ceilSubSize, ceilSubSize, ceilN);
		int **a21 = getPartialMatrix(M->A, ceilSubSize, 0, ceilN, ceilSubSize);
		int **a22 = getPartialMatrix(M->A, ceilSubSize, ceilSubSize, ceilN, ceilN);
		int **b11 = getPartialMatrix(M->B, 0, 0, ceilSubSize, ceilSubSize);
		int **b12 = getPartialMatrix(M->B, 0, ceilSubSize, ceilSubSize, ceilN);
		int **b21 = getPartialMatrix(M->B, ceilSubSize, 0, ceilN, ceilSubSize);
		int **b22 = getPartialMatrix(M->B, ceilSubSize, ceilSubSize, ceilN, ceilN);

		// Assign tasks for recursion
		matrixMultStrassen *P1 = initMatrixMultStrassen(ceilSubSize);
		P1->A = getPartialMatrix(M->A, 0, 0, ceilSubSize, ceilSubSize); // a11
		P1->B = subtractMatrix(b12, b22, subSize, ceilSubSize);

		matrixMultStrassen *P2 = initMatrixMultStrassen(ceilSubSize);
		P2->A = addMatrix(a11, a12, subSize, ceilSubSize);
		P2->B = deepCopyMatrix(b22, subSize, ceilSubSize);

		matrixMultStrassen *P3 = initMatrixMultStrassen(ceilSubSize);
		P3->A = addMatrix(a21, a22, subSize, ceilSubSize);
		P3->B = deepCopyMatrix(b11, subSize, ceilSubSize);

		matrixMultStrassen *P4 = initMatrixMultStrassen(ceilSubSize);
		P4->A = deepCopyMatrix(a22, subSize, ceilSubSize);
		P4->B = subtractMatrix(b21, b11, subSize, ceilSubSize);

		matrixMultStrassen *P5 = initMatrixMultStrassen(ceilSubSize);
		P5->A = addMatrix(a11, a22, subSize, ceilSubSize);
		P5->B = addMatrix(b11, b22, subSize, ceilSubSize);

		matrixMultStrassen *P6 = initMatrixMultStrassen(ceilSubSize);
		P6->A = subtractMatrix(a12, a22, subSize, ceilSubSize);
		P6->B = addMatrix(b21, b22, subSize, ceilSubSize);

		matrixMultStrassen *P7 = initMatrixMultStrassen(ceilSubSize);
		P7->A = subtractMatrix(a11, a21, subSize, ceilSubSize);
		P7->B = addMatrix(b11, b12, subSize, ceilSubSize);

		// Perform the multiplication recursively
		int **p1 = ((matrixS *)recursiveMul((void *)P1))->A;
		int **p2 = ((matrixS *)recursiveMul((void *)P2))->A;
		int **p3 = ((matrixS *)recursiveMul((void *)P3))->A;
		int **p4 = ((matrixS *)recursiveMul((void *)P4))->A;
		int **p5 = ((matrixS *)recursiveMul((void *)P5))->A;
		int **p6 = ((matrixS *)recursiveMul((void *)P6))->A;
		int **p7 = ((matrixS *)recursiveMul((void *)P7))->A;

		// Free the allocated structures/arrays
		freeMatrixMultStrassen(P1, ceilSubSize);
		freeMatrixMultStrassen(P2, ceilSubSize);
		freeMatrixMultStrassen(P3, ceilSubSize);
		freeMatrixMultStrassen(P4, ceilSubSize);
		freeMatrixMultStrassen(P5, ceilSubSize);
		freeMatrixMultStrassen(P6, ceilSubSize);
		freeMatrixMultStrassen(P7, ceilSubSize);

		freeMatrix(a11, ceilSubSize);
		freeMatrix(a12, ceilSubSize);
		freeMatrix(a21, ceilSubSize);
		freeMatrix(a22, ceilSubSize);
		freeMatrix(b11, ceilSubSize);
		freeMatrix(b12, ceilSubSize);
		freeMatrix(b21, ceilSubSize);
		freeMatrix(b22, ceilSubSize);

		// Combined the above result to calculate the parts
		// of resulting matrix C.
		int **c11 = addMatrix(p5, p4, subSize, ceilSubSize);
		c11 = subtractMatrix(c11, p2, subSize, ceilSubSize);
		c11 = addMatrix(c11, p6, subSize, ceilSubSize);
		copyMatrix(c11, subSize, c->A, 0, 0);

		int **c12 = addMatrix(p1, p2, subSize, ceilSubSize);
		copyMatrix(c12, subSize, c->A, 0, subSize);

		int **c21 = addMatrix(p3, p4, subSize, ceilSubSize);
		copyMatrix(c21, subSize, c->A, subSize, 0);

		int **c22 = addMatrix(p5, p1, subSize, ceilSubSize);
		c22 = subtractMatrix(c22, p3, subSize, ceilSubSize);
		c22 = subtractMatrix(c22, p7, subSize, ceilSubSize);
		copyMatrix(c22, subSize, c->A, subSize, subSize);

		freeMatrix(c11, ceilSubSize);
		freeMatrix(c12, ceilSubSize);
		freeMatrix(c21, ceilSubSize);
		freeMatrix(c22, ceilSubSize);

		freeMatrix(p1, ceilSubSize);
		freeMatrix(p2, ceilSubSize);
		freeMatrix(p3, ceilSubSize);
		freeMatrix(p4, ceilSubSize);
		freeMatrix(p5, ceilSubSize);
		freeMatrix(p6, ceilSubSize);
		freeMatrix(p7, ceilSubSize);
	}
	return (void *)c;
}

int **multiplyStrassen(matrixMultStrassen *M)
{
	/*
	Function to spawn threads to perform Strassen's method of matrix multiplication
	*/
	pthread_t tid[NUM_THREADS];
	int n = M->n;
	int ceilN = M->ceiling;
	int ceilSubSize = ceilN / 2;
	int subSize = ceilSubSize;
	int **c = initEmptyMatrix(ceilN);

	void *temp1;
	void *temp2;
	void *temp3;
	void *temp4;
	void *temp5;
	void *temp6;
	void *temp7;

	int **p1;
	int **p2;
	int **p3;
	int **p4;
	int **p5;
	int **p6;
	int **p7;

	// Divide the matrix A and B into four sub-matrices
	int **a11 = getPartialMatrix(M->A, 0, 0, ceilSubSize, ceilSubSize);
	int **a12 = getPartialMatrix(M->A, 0, ceilSubSize, ceilSubSize, ceilN);
	int **a21 = getPartialMatrix(M->A, ceilSubSize, 0, ceilN, ceilSubSize);
	int **a22 = getPartialMatrix(M->A, ceilSubSize, ceilSubSize, ceilN, ceilN);
	int **b11 = getPartialMatrix(M->B, 0, 0, ceilSubSize, ceilSubSize);
	int **b12 = getPartialMatrix(M->B, 0, ceilSubSize, ceilSubSize, ceilN);
	int **b21 = getPartialMatrix(M->B, ceilSubSize, 0, ceilN, ceilSubSize);
	int **b22 = getPartialMatrix(M->B, ceilSubSize, ceilSubSize, ceilN, ceilN);

	// Assign tasks for each thread
	matrixMultStrassen *P1 = initMatrixMultStrassen(ceilSubSize);
	P1->A = deepCopyMatrix(a11, subSize, ceilSubSize);
	P1->B = subtractMatrix(b12, b22, subSize, ceilSubSize);

	matrixMultStrassen *P2 = initMatrixMultStrassen(ceilSubSize);
	P2->A = addMatrix(a11, a12, subSize, ceilSubSize);
	P2->B = deepCopyMatrix(b22, subSize, ceilSubSize);

	matrixMultStrassen *P3 = initMatrixMultStrassen(ceilSubSize);
	P3->A = addMatrix(a21, a22, subSize, ceilSubSize);
	P3->B = deepCopyMatrix(b11, subSize, ceilSubSize);

	matrixMultStrassen *P4 = initMatrixMultStrassen(ceilSubSize);
	P4->A = deepCopyMatrix(a22, subSize, ceilSubSize);
	P4->B = subtractMatrix(b21, b11, subSize, ceilSubSize);

	matrixMultStrassen *P5 = initMatrixMultStrassen(ceilSubSize);
	P5->A = addMatrix(a11, a22, subSize, ceilSubSize);
	P5->B = addMatrix(b11, b22, subSize, ceilSubSize);

	matrixMultStrassen *P6 = initMatrixMultStrassen(ceilSubSize);
	P6->A = subtractMatrix(a12, a22, subSize, ceilSubSize);
	P6->B = addMatrix(b21, b22, subSize, ceilSubSize);

	matrixMultStrassen *P7 = initMatrixMultStrassen(ceilSubSize);
	P7->A = subtractMatrix(a11, a21, subSize, ceilSubSize);
	P7->B = addMatrix(b11, b12, subSize, ceilSubSize);

	// Perform the multiplication in parallel by spawning 7 threads
	pthread_create(&tid[0], NULL, recursiveMul, (void *)P1);
	pthread_create(&tid[1], NULL, recursiveMul, (void *)P2);
	pthread_create(&tid[2], NULL, recursiveMul, (void *)P3);
	pthread_create(&tid[3], NULL, recursiveMul, (void *)P4);
	pthread_create(&tid[4], NULL, recursiveMul, (void *)P5);
	pthread_create(&tid[5], NULL, recursiveMul, (void *)P6);
	pthread_create(&tid[6], NULL, recursiveMul, (void *)P7);

	// Join the threads and collect the results
	pthread_join(tid[0], &temp1);
	p1 = ((matrixS *)temp1)->A;

	pthread_join(tid[1], &temp2);
	p2 = ((matrixS *)temp2)->A;

	pthread_join(tid[2], &temp3);
	p3 = ((matrixS *)temp3)->A;

	pthread_join(tid[3], &temp4);
	p4 = ((matrixS *)temp4)->A;

	pthread_join(tid[4], &temp5);
	p5 = ((matrixS *)temp5)->A;

	pthread_join(tid[5], &temp6);
	p6 = ((matrixS *)temp6)->A;

	pthread_join(tid[6], &temp7);
	p7 = ((matrixS *)temp7)->A;

	// Combined the above result to calculate the parts
	// of resulting matrix C.
	int **c11 = addMatrix(p5, p4, subSize, ceilSubSize);
	c11 = subtractMatrix(c11, p2, subSize, ceilSubSize);
	c11 = addMatrix(c11, p6, subSize, ceilSubSize);
	copyMatrix(c11, subSize, c, 0, 0);

	int **c12 = addMatrix(p1, p2, subSize, ceilSubSize);
	copyMatrix(c12, subSize, c, 0, subSize);

	int **c21 = addMatrix(p3, p4, subSize, ceilSubSize);
	copyMatrix(c21, subSize, c, subSize, 0);

	int **c22 = addMatrix(p5, p1, subSize, ceilSubSize);
	c22 = subtractMatrix(c22, p3, subSize, ceilSubSize);
	c22 = subtractMatrix(c22, p7, subSize, ceilSubSize);
	copyMatrix(c22, subSize, c, subSize, subSize);

	freeMatrixMultStrassen(P1, ceilSubSize);
	freeMatrixMultStrassen(P2, ceilSubSize);
	freeMatrixMultStrassen(P3, ceilSubSize);
	freeMatrixMultStrassen(P4, ceilSubSize);
	freeMatrixMultStrassen(P5, ceilSubSize);
	freeMatrixMultStrassen(P6, ceilSubSize);
	freeMatrixMultStrassen(P7, ceilSubSize);

	freeMatrix(a11, ceilSubSize);
	freeMatrix(a12, ceilSubSize);
	freeMatrix(a21, ceilSubSize);
	freeMatrix(a22, ceilSubSize);
	freeMatrix(b11, ceilSubSize);
	freeMatrix(b12, ceilSubSize);
	freeMatrix(b21, ceilSubSize);
	freeMatrix(b22, ceilSubSize);
	freeMatrix(c11, ceilSubSize);
	freeMatrix(c12, ceilSubSize);
	freeMatrix(c21, ceilSubSize);
	freeMatrix(c22, ceilSubSize);

	freeMatrix(p1, ceilSubSize);
	freeMatrix(p2, ceilSubSize);
	freeMatrix(p3, ceilSubSize);
	freeMatrix(p4, ceilSubSize);
	freeMatrix(p5, ceilSubSize);
	freeMatrix(p6, ceilSubSize);
	freeMatrix(p7, ceilSubSize);

	return c;
}

int main()
{
	// Declare info about the matrices
	int n = 7;
	int ceiling = ceil_log2(n);

	// Initiate the time
	struct timespec begin_seq, end_seq;
	struct timespec begin_strassen, end_strassen;

	// Init struct to perform matrix multiplication
	matrixMultStrassen *M = initMatrixMultStrassen(n);
	M->A = initMatrix(n, ceiling);
	M->B = initMatrix(n, ceiling);

	// Start Strassen matrix multiplcation
	clock_gettime(CLOCK_REALTIME, &begin_strassen);
	int **res = multiplyStrassen(M);
	clock_gettime(CLOCK_REALTIME, &end_strassen);

	long seconds_strassen = end_strassen.tv_sec - begin_strassen.tv_sec;
	long nanos_strassen = end_strassen.tv_nsec - begin_strassen.tv_nsec;
	double elapsed_strassen = seconds_strassen * 1000 + nanos_strassen * 1e-6;
	std::cout << "Strassen: ";
	std::cout << elapsed_strassen << "ms" << std::endl;

	// Start sequential matrix multiplication.
	clock_gettime(CLOCK_REALTIME, &begin_seq);
	int **res_seq = naive_matrix_mult(M->A, M->B, M->n);
	clock_gettime(CLOCK_REALTIME, &end_seq);

	long seconds_seq = end_seq.tv_sec - begin_seq.tv_sec;
	long nanos_seq = end_seq.tv_nsec - begin_seq.tv_nsec;
	double elapsed_seq = seconds_seq * 1000 + nanos_seq * 1e-6;
	std::cout << "Sequential: ";
	std::cout << elapsed_seq << "ms" << std::endl;

	printMatrix(res, n);
	printMatrix(res_seq, n);
	freeMatrix(res, ceiling);
	freeMatrix(res_seq, n);
	freeMatrixMultStrassen(M, ceiling);
	return 0;
}
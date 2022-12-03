#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>
#include <math.h>
#include <iostream>
#include <unistd.h>
#include <sys/syscall.h>
#define gettid() syscall(SYS_gettid)

// Improve the Strassen Matrix multiplication by reducing
// the number of allocation of matrices

int NUM_THREADS = 7;

typedef struct
{
    int **A;
    int **B;
    int n;
    int ceiling;
} matrixMultStrassen;

typedef struct
{
    int **A;
    int n;
    int ceiling;
} matrixS;

// Part 1
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
    return (int)pow(2, ceil(log(n) / log(2)));
}

int **initMatrix(int n, int ceiling)
{
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
    int **matrix = (int **)malloc(ceiling * sizeof(int *));
    for (int i = 0; i < ceiling; i++)
    {
        matrix[i] = (int *)malloc(ceiling * sizeof(int));
    }
    return matrix;
}

matrixMultStrassen *initMatrixMultStrassen(int n)
{
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
    int ceiling = ceil_log2(n);
    matrixS *M = (matrixS *)malloc(sizeof(matrixS));
    M->A = initEmptyMatrix(ceiling);
    M->n = n;
    M->ceiling = ceiling;
    return M;
}

void freeMatrix(int **matrix, int ceiling)
{
    for (int i = 0; i < ceiling; i++)
    {
        free(matrix[i]);
    }
    free(matrix);
}

void freeMatrixMultStrassen(matrixMultStrassen *M, int ceiling)
{
    freeMatrix(M->A, ceiling);
    freeMatrix(M->B, ceiling);
    free(M);
}

void printMatrix(int **matrix, int n)
{
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

void addMatrixInplace(int **matrix1, int **matrix2, int n, int ceiling)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            matrix1[i][j] += matrix2[i][j];
        }
    }
}

void subtractMatrixInplace(int **matrix1, int **matrix2, int n, int ceiling)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            matrix1[i][j] -= matrix2[i][j];
        }
    }
}

int **getPartialMatrix(int **matrix, int topleft_row, int topleft_col, int bottomright_row, int bottomright_col)
{
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

int **getPartialMatrixSubtract(int **matrix, int topleft_row1, int topleft_col1, int bottomright_row1, int bottomright_col1, int topleft_row2, int topleft_col2, int bottomright_row2, int bottomright_col2)
{
    /*
    Perform matrix addition on sub-matrices of the matrix
    */
    int n = bottomright_col1 - topleft_col1;
    int ceiling = ceil_log2(n);
    int **ans = initMatrix(n, ceiling);
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            ans[i][j] = matrix[topleft_row1 + i][topleft_col1 + j] - matrix[topleft_row2 + i][topleft_col2 + j];
        }
    }
    return ans;
}

int **getPartialMatrixAdd(int **matrix, int topleft_row1, int topleft_col1, int bottomright_row1, int bottomright_col1, int topleft_row2, int topleft_col2, int bottomright_row2, int bottomright_col2)
{
    /*
    Perform matrix subtraction on sub-matrices of the matrix
    */
    int n = bottomright_col1 - topleft_col1;
    int ceiling = ceil_log2(n);
    int **ans = initMatrix(n, ceiling);
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            ans[i][j] = matrix[topleft_row1 + i][topleft_col1 + j] + matrix[topleft_row2 + i][topleft_col2 + j];
        }
    }
    return ans;
}

void copyMatrix(int **source, int size, int **destination, int starting_row, int starting_col)
{
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

        // Main difference: Instead of allocating for each sub-matrice of A and B
        // we directly perform calculation on the original matrices A and B
        matrixMultStrassen *P1 = initMatrixMultStrassen(ceilSubSize);
        // a11
        P1->A = getPartialMatrix(M->A, 0, 0, ceilSubSize, ceilSubSize);
        // b11 - b22
        P1->B = getPartialMatrixSubtract(M->B, 0, 0, ceilSubSize, ceilSubSize, ceilSubSize, ceilSubSize, ceilN, ceilN);

        matrixMultStrassen *P2 = initMatrixMultStrassen(ceilSubSize);
        // a11 + a12
        P2->A = getPartialMatrixAdd(M->A, 0, 0, ceilSubSize, ceilSubSize, 0, ceilSubSize, ceilSubSize, ceilN);
        // b22
        P2->B = getPartialMatrix(M->B, ceilSubSize, ceilSubSize, ceilN, ceilN);

        matrixMultStrassen *P3 = initMatrixMultStrassen(ceilSubSize);
        // a21 + a22
        P3->A = getPartialMatrixAdd(M->A, ceilSubSize, 0, ceilN, ceilSubSize, ceilSubSize, ceilSubSize, ceilN, ceilN);
        // b11
        P3->B = getPartialMatrix(M->B, 0, 0, ceilSubSize, ceilSubSize);

        matrixMultStrassen *P4 = initMatrixMultStrassen(ceilSubSize);
        // a22
        P4->A = getPartialMatrix(M->A, ceilSubSize, ceilSubSize, ceilN, ceilN);
        // b21 - b11
        P4->B = getPartialMatrixSubtract(M->B, ceilSubSize, 0, ceilN, ceilSubSize, 0, 0, ceilSubSize, ceilSubSize);

        matrixMultStrassen *P5 = initMatrixMultStrassen(ceilSubSize);
        // a11 + a22
        P5->A = getPartialMatrixAdd(M->A, 0, 0, ceilSubSize, ceilSubSize, ceilSubSize, ceilSubSize, ceilN, ceilN);
        // b11 + b22
        P5->B = getPartialMatrixAdd(M->B, 0, 0, ceilSubSize, ceilSubSize, ceilSubSize, ceilSubSize, ceilN, ceilN);

        matrixMultStrassen *P6 = initMatrixMultStrassen(ceilSubSize);
        // a12 - a22
        P6->A = getPartialMatrixSubtract(M->A, 0, ceilSubSize, ceilSubSize, ceilN, ceilSubSize, ceilSubSize, ceilN, ceilN);
        // b21 + b22
        P6->B = getPartialMatrixAdd(M->B, ceilSubSize, 0, ceilN, ceilSubSize, ceilSubSize, ceilSubSize, ceilN, ceilN);

        matrixMultStrassen *P7 = initMatrixMultStrassen(ceilSubSize);
        // a11 - a21
        P7->A = getPartialMatrixSubtract(M->A, 0, 0, ceilSubSize, ceilSubSize, ceilSubSize, 0, ceilN, ceilSubSize);
        // b11 + b12
        P7->B = getPartialMatrixAdd(M->B, 0, 0, ceilSubSize, ceilSubSize, 0, ceilSubSize, ceilSubSize, ceilN);

        int **p1 = ((matrixS *)recursiveMul((void *)P1))->A;
        int **p2 = ((matrixS *)recursiveMul((void *)P2))->A;
        int **p3 = ((matrixS *)recursiveMul((void *)P3))->A;
        int **p4 = ((matrixS *)recursiveMul((void *)P4))->A;
        int **p5 = ((matrixS *)recursiveMul((void *)P5))->A;
        int **p6 = ((matrixS *)recursiveMul((void *)P6))->A;
        int **p7 = ((matrixS *)recursiveMul((void *)P7))->A;

        freeMatrixMultStrassen(P1, ceilSubSize);
        freeMatrixMultStrassen(P2, ceilSubSize);
        freeMatrixMultStrassen(P3, ceilSubSize);
        freeMatrixMultStrassen(P4, ceilSubSize);
        freeMatrixMultStrassen(P5, ceilSubSize);
        freeMatrixMultStrassen(P6, ceilSubSize);
        freeMatrixMultStrassen(P7, ceilSubSize);

        int **c11 = addMatrix(p5, p4, subSize, ceilSubSize);
        subtractMatrixInplace(c11, p2, subSize, ceilSubSize);
        addMatrixInplace(c11, p6, subSize, ceilSubSize);
        copyMatrix(c11, subSize, c->A, 0, 0);

        int **c12 = addMatrix(p1, p2, subSize, ceilSubSize);
        copyMatrix(c12, subSize, c->A, 0, subSize);

        int **c21 = addMatrix(p3, p4, subSize, ceilSubSize);
        copyMatrix(c21, subSize, c->A, subSize, 0);

        int **c22 = addMatrix(p5, p1, subSize, ceilSubSize);
        subtractMatrixInplace(c22, p3, subSize, ceilSubSize);
        subtractMatrixInplace(c22, p7, subSize, ceilSubSize);
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

    // Main difference: Instead of allocating for each sub-matrice of A and B
    // we directly perform calculation on the original matrices A and B
    matrixMultStrassen *P1 = initMatrixMultStrassen(ceilSubSize);
    // a11
    P1->A = getPartialMatrix(M->A, 0, 0, ceilSubSize, ceilSubSize);
    // b11 - b22
    P1->B = getPartialMatrixSubtract(M->B, 0, 0, ceilSubSize, ceilSubSize, ceilSubSize, ceilSubSize, ceilN, ceilN);

    matrixMultStrassen *P2 = initMatrixMultStrassen(ceilSubSize);
    // a11 + a12
    P2->A = getPartialMatrixAdd(M->A, 0, 0, ceilSubSize, ceilSubSize, 0, ceilSubSize, ceilSubSize, ceilN);
    // b22
    P2->B = getPartialMatrix(M->B, ceilSubSize, ceilSubSize, ceilN, ceilN);

    matrixMultStrassen *P3 = initMatrixMultStrassen(ceilSubSize);
    // a21 + a22
    P3->A = getPartialMatrixAdd(M->A, ceilSubSize, 0, ceilN, ceilSubSize, ceilSubSize, ceilSubSize, ceilN, ceilN);
    // b11
    P3->B = getPartialMatrix(M->B, 0, 0, ceilSubSize, ceilSubSize);

    matrixMultStrassen *P4 = initMatrixMultStrassen(ceilSubSize);
    // a22
    P4->A = getPartialMatrix(M->A, ceilSubSize, ceilSubSize, ceilN, ceilN);
    // b21 - b11
    P4->B = getPartialMatrixSubtract(M->B, ceilSubSize, 0, ceilN, ceilSubSize, 0, 0, ceilSubSize, ceilSubSize);

    matrixMultStrassen *P5 = initMatrixMultStrassen(ceilSubSize);
    // a11 + a22
    P5->A = getPartialMatrixAdd(M->A, 0, 0, ceilSubSize, ceilSubSize, ceilSubSize, ceilSubSize, ceilN, ceilN);
    // b11 + b22
    P5->B = getPartialMatrixAdd(M->B, 0, 0, ceilSubSize, ceilSubSize, ceilSubSize, ceilSubSize, ceilN, ceilN);

    matrixMultStrassen *P6 = initMatrixMultStrassen(ceilSubSize);
    // a12 - a22
    P6->A = getPartialMatrixSubtract(M->A, 0, ceilSubSize, ceilSubSize, ceilN, ceilSubSize, ceilSubSize, ceilN, ceilN);
    // b21 + b22
    P6->B = getPartialMatrixAdd(M->B, ceilSubSize, 0, ceilN, ceilSubSize, ceilSubSize, ceilSubSize, ceilN, ceilN);

    matrixMultStrassen *P7 = initMatrixMultStrassen(ceilSubSize);
    // a11 - a21
    P7->A = getPartialMatrixSubtract(M->A, 0, 0, ceilSubSize, ceilSubSize, ceilSubSize, 0, ceilN, ceilSubSize);
    // b11 + b12
    P7->B = getPartialMatrixAdd(M->B, 0, 0, ceilSubSize, ceilSubSize, 0, ceilSubSize, ceilSubSize, ceilN);

    pthread_create(&tid[0], NULL, recursiveMul, (void *)P1);
    pthread_create(&tid[1], NULL, recursiveMul, (void *)P2);
    pthread_create(&tid[2], NULL, recursiveMul, (void *)P3);
    pthread_create(&tid[3], NULL, recursiveMul, (void *)P4);
    pthread_create(&tid[4], NULL, recursiveMul, (void *)P5);
    pthread_create(&tid[5], NULL, recursiveMul, (void *)P6);
    pthread_create(&tid[6], NULL, recursiveMul, (void *)P7);

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

    int **c11 = addMatrix(p5, p4, subSize, ceilSubSize);
    subtractMatrixInplace(c11, p2, subSize, ceilSubSize);
    addMatrixInplace(c11, p6, subSize, ceilSubSize);
    copyMatrix(c11, subSize, c, 0, 0);

    int **c12 = addMatrix(p1, p2, subSize, ceilSubSize);
    copyMatrix(c12, subSize, c, 0, subSize);

    int **c21 = addMatrix(p3, p4, subSize, ceilSubSize);
    copyMatrix(c21, subSize, c, subSize, 0);

    int **c22 = addMatrix(p5, p1, subSize, ceilSubSize);
    subtractMatrixInplace(c22, p3, subSize, ceilSubSize);
    subtractMatrixInplace(c22, p7, subSize, ceilSubSize);
    copyMatrix(c22, subSize, c, subSize, subSize);

    freeMatrixMultStrassen(P1, ceilSubSize);
    freeMatrixMultStrassen(P2, ceilSubSize);
    freeMatrixMultStrassen(P3, ceilSubSize);
    freeMatrixMultStrassen(P4, ceilSubSize);
    freeMatrixMultStrassen(P5, ceilSubSize);
    freeMatrixMultStrassen(P6, ceilSubSize);
    freeMatrixMultStrassen(P7, ceilSubSize);

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
    int n = 100;
    int ceiling = ceil_log2(n);
    struct timespec begin_seq, end_seq;
    struct timespec begin_strassen, end_strassen;
    matrixMultStrassen *M = initMatrixMultStrassen(n);
    M->A = initMatrix(n, ceiling);
    M->B = initMatrix(n, ceiling);

    clock_gettime(CLOCK_REALTIME, &begin_strassen);
    int **res = multiplyStrassen(M);
    clock_gettime(CLOCK_REALTIME, &end_strassen);

    long seconds_strassen = end_strassen.tv_sec - begin_strassen.tv_sec;
    long nanos_strassen = end_strassen.tv_nsec - begin_strassen.tv_nsec;
    double elapsed_strassen = seconds_strassen * 1000 + nanos_strassen * 1e-6;
    std::cout << "Strassen: ";
    std::cout << elapsed_strassen << "ms" << std::endl;

    clock_gettime(CLOCK_REALTIME, &begin_seq);
    int **res_seq = naive_matrix_mult(M->A, M->B, M->n);
    clock_gettime(CLOCK_REALTIME, &end_seq);

    long seconds_seq = end_seq.tv_sec - begin_seq.tv_sec;
    long nanos_seq = end_seq.tv_nsec - begin_seq.tv_nsec;
    double elapsed_seq = seconds_seq * 1000 + nanos_seq * 1e-6;
    std::cout << "Sequential: ";
    std::cout << elapsed_seq << "ms" << std::endl;

    // printMatrix(res, n);
    // printMatrix(res_seq, n);
    freeMatrix(res, ceiling);
    freeMatrix(res_seq, n);
    freeMatrixMultStrassen(M, ceiling);
    return 0;
}
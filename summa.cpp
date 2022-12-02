#include <iostream>
#include <pthread.h>
#include <mutex>
#include <time.h>
#include <math.h>
#include <omp.h>
using namespace std;
#define WORK_LOAD 10

// timing naive sequential, parallel, and summa parallel

// Part 1
int **naive_matrix_mult(int **data_ptr, int **data_ptr2, int n)
{
    int **res = (int **)malloc(n * sizeof(int *));
    for (int i = 0; i < n; i++)
        res[i] = (int *)malloc(n * sizeof(int));

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            for (int k = 0; k < n; k++)
            {
                res[i][j] += data_ptr[i][k] * data_ptr2[k][j];
            }
        }
    }
    return res;
}

// Part 2
int **summa_matrix_mult(int **data_ptr, int **data_ptr2, int n)
{
    int **res = (int **)malloc(n * sizeof(int *));
    for (int i = 0; i < n; i++)
        res[i] = (int *)malloc(n * sizeof(int));

    for (int k = 0; k < n; k++)
    {
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                res[i][j] += data_ptr[i][k] * data_ptr2[k][j];
            }
        }
    }
    return res;
}

// Helper function
int compareMatrix(int **matrix1, int **matrix2, int n)
{
    int count = 0;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (matrix1[i][j] != matrix2[i][j])
            {
                // std::cout << matrix1[i][j] << " " << matrix2[i][j] << std::endl;
                count++;
                // return -1;
            }
        }
    }
    return count;
}

int ceil_log2(int n)
{
    return (int)pow(2, ceil(log(n) / log(2)));
}

int **initMatrix(int n)
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
    return matrix;
}

int **getPartialMatrix(int **matrix, int topleft_row, int topleft_col, int bottomright_row, int bottomright_col)
{
    int n = bottomright_col - topleft_col;
    int **ans = initMatrix(n);
    // std::cout << n << " " << topleft_row << " " << topleft_col << std::endl;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            ans[i][j] = matrix[topleft_row + i][topleft_col + j];
        }
    }
    return ans;
}

// Part 3
struct summa
{
    int id;
    int work;
    int **res;
    int **data_ptr;
    int **data_ptr2;
    int n;
    pthread_mutex_t *lock;
};

// Struct for pipeline summa
// {x|y}_data{n1}_{n2} indicates the x or y coordinate
// of the data_ptr{n1} of the phase n2
struct pipelinesumma
{
    int id;
    int **res;
    int **data_ptr;
    int **data_ptr2;
    int x_data1_1;
    int y_data1_1;
    int x_data2_1;
    int y_data2_1;
    int x_data1_2;
    int y_data1_2;
    int x_data2_2;
    int y_data2_2;
    int n;
    pthread_mutex_t *lock;
};

// Struct for improved pipeline summa
// {x|y}_data{n1}_1 indicates the location to be updated on data_ptr{n1}
// {x|y}_data{n1}_2 indicates the location to be worked on data_ptr{n1}

struct pipelinesumma2
{
    int id;
    int **res;
    int **data_ptr;
    int **data_ptr2;
    int x_data1_1;
    int y_data1_1;
    int x_data2_1;
    int y_data2_1;
    int x_data1_2;
    int y_data1_2;
    int x_data2_2;
    int y_data2_2;
    int n;
    pthread_mutex_t *lock;
};

void *thread_mult(void *arg)
{
    summa *data = (summa *)arg;
    int id = data->id;
    int *col = (int *)malloc(sizeof(int) * data->n * WORK_LOAD);
    int *row = (int *)malloc(sizeof(int) * data->n * WORK_LOAD);
    // broadcast to local matrix A_id and B_id
    for (int k = 0; k < WORK_LOAD; k++)
    {
        for (int i = 0; i < data->n; i++)
        {
            col[k * data->n + i] = data->data_ptr[i][id * WORK_LOAD + k];
        }
        for (int j = 0; j < data->n; j++)
        {
            row[k * data->n + j] = data->data_ptr2[id * WORK_LOAD + k][j];
        }
        pthread_mutex_lock(data->lock);
#pragma omp parallel for
        for (int i = 0; i < data->n; i++)
        {
            for (int j = 0; j < data->n; j++)
            {
                data->res[i][j] += col[k * data->n + i] * row[k * data->n + j];
            }
        }
        pthread_mutex_unlock(data->lock);
    }
    free(col);
    free(row);
    return NULL;
}

int **parallel_summa(int **data_ptr, int **data_ptr2, int n)
{
    int **result = (int **)malloc(n * sizeof(int *));
    for (int i = 0; i < n; i++)
        result[i] = (int *)malloc(n * sizeof(int));

    pthread_t tid[n];
    summa thread_info[n];
    pthread_mutex_t *lock;
    lock = (pthread_mutex_t *)malloc(sizeof(pthread_mutex_t));
    pthread_mutex_init(lock, NULL);
    for (int k = 0; k < n / WORK_LOAD; k++)
    {
        thread_info[k].id = k;
        thread_info[k].work = WORK_LOAD;
        thread_info[k].res = result;
        thread_info[k].data_ptr = data_ptr;
        thread_info[k].data_ptr2 = data_ptr2;
        thread_info[k].n = n;
        thread_info[k].lock = lock;
        pthread_create(&tid[k], NULL, thread_mult, (void *)&thread_info[k]);
    }
    for (int k = 0; k < n / WORK_LOAD; k++)
    {
        pthread_join(tid[k], NULL);
    }
    pthread_mutex_destroy(lock);
    free(lock);
    return result;
}

void *pipeline_mult(void *arg)
{
    /*
    Pipeline matrix multiplication for each thread.
    In this case, we are dividing it into 4 threads
    */
    pipelinesumma *data = (pipelinesumma *)arg;
    int id = data->id;
    int *col = (int *)malloc(sizeof(int) * data->n * data->n);
    int *row = (int *)malloc(sizeof(int) * data->n * data->n);
    // broadcast to local matrix A_id and B_id
    for (int k = 0; k < data->n; k++)
    {
        for (int i = 0; i < data->n; i++)
        {
            col[k * data->n + i] = data->data_ptr[data->x_data1_1 + i][data->y_data1_1 + k];
        }
        for (int j = 0; j < data->n; j++)
        {
            row[k * data->n + j] = data->data_ptr2[data->x_data2_1 + k][data->y_data2_1 + j];
        }
        pthread_mutex_lock(data->lock);
        for (int i = 0; i < data->n; i++)
        {
            for (int j = 0; j < data->n; j++)
            {
                data->res[data->x_data1_1 + i][data->y_data1_1 + j] += col[k * data->n + i] * row[k * data->n + j];
            }
        }
        pthread_mutex_unlock(data->lock);
    }
    for (int k = 0; k < data->n; k++)
    {
        for (int i = 0; i < data->n; i++)
        {
            col[k * data->n + i] = data->data_ptr[data->x_data1_2 + i][data->y_data1_2 + k];
        }
        for (int j = 0; j < data->n; j++)
        {
            row[k * data->n + j] = data->data_ptr2[data->x_data2_2 + k][data->y_data2_2 + j];
        }
        pthread_mutex_lock(data->lock);
#pragma omp parallel for
        for (int i = 0; i < data->n; i++)
        {
            for (int j = 0; j < data->n; j++)
            {
                data->res[data->x_data1_1 + i][data->y_data1_1 + j] += col[k * data->n + i] * row[k * data->n + j];
            }
        }
        pthread_mutex_unlock(data->lock);
    }
    free(col);
    free(row);
    return NULL;
}

int **pipelined_summa(int **data_ptr, int **data_ptr2, int n)
{
    /*
    Main function to spawn 4 threads and perform summa by
    dividing into 4 blocks
    */
    int **result = (int **)malloc(n * sizeof(int *));
    for (int i = 0; i < n; i++)
        result[i] = (int *)malloc(n * sizeof(int));

    pthread_t tid[4];
    pipelinesumma thread_info[4];
    pthread_mutex_t *lock;
    lock = (pthread_mutex_t *)malloc(sizeof(pthread_mutex_t));
    pthread_mutex_init(lock, NULL);
    for (int k = 0; k < 4; k++)
    {
        thread_info[k].id = k;
        thread_info[k].res = result;
        thread_info[k].data_ptr = data_ptr;
        thread_info[k].data_ptr2 = data_ptr2;
        thread_info[k].x_data1_1 = (k / 2) * (n / 2);
        thread_info[k].y_data1_1 = (k % 2) * (n / 2);
        thread_info[k].x_data2_1 = (k / 2) * (n / 2);
        thread_info[k].y_data2_1 = (k % 2) * (n / 2);
        thread_info[k].x_data1_2 = (k / 2) * (n / 2);
        thread_info[k].y_data1_2 = ((k + 1) % 2) * (n / 2);
        thread_info[k].x_data2_2 = ((k + 1) % 2) * (n / 2);
        thread_info[k].y_data2_2 = (k % 2) * (n / 2);
        thread_info[k].n = n / 2; // Assume that n is divisible by 2
        thread_info[k].lock = lock;
        pthread_create(&tid[k], NULL, pipeline_mult, (void *)&thread_info[k]);
    }
    for (int k = 0; k < 4; k++)
    {
        pthread_join(tid[k], NULL);
    }
    pthread_mutex_destroy(lock);
    free(lock);
    return result;
}

void *pipeline_mult2(void *arg)
{
    /*
    Improved version of pipeline summa
    Pipeline matrix multiplication for each thread.
    In this case, we are dividing it into 8 threads
    */
    pipelinesumma2 *data = (pipelinesumma2 *)arg;
    int id = data->id;
    int *col = (int *)malloc(sizeof(int) * data->n * data->n);
    int *row = (int *)malloc(sizeof(int) * data->n * data->n);
    // broadcast to local matrix A_id and B_id
    for (int k = 0; k < data->n; k++)
    {
        for (int i = 0; i < data->n; i++)
        {
            col[k * data->n + i] = data->data_ptr[data->x_data1_2 + i][data->y_data1_2 + k];
        }
        for (int j = 0; j < data->n; j++)
        {
            row[k * data->n + j] = data->data_ptr2[data->x_data2_2 + k][data->y_data2_2 + j];
        }
        pthread_mutex_lock(data->lock);
#pragma omp parallel for
        for (int i = 0; i < data->n; i++)
        {
#pragma omp parallel for
            for (int j = 0; j < data->n; j++)
            {
                data->res[data->x_data1_1 + i][data->y_data1_1 + j] += col[k * data->n + i] * row[k * data->n + j];
            }
        }
        pthread_mutex_unlock(data->lock);
    }
    free(col);
    free(row);
    return NULL;
}

int **pipelined_summa2(int **data_ptr, int **data_ptr2, int n)
{
    /*
    Main function to spawn 8 threads and perform summa by
    dividing into 4 blocks
    */
    int **result = (int **)malloc(n * sizeof(int *));
    for (int i = 0; i < n; i++)
        result[i] = (int *)malloc(n * sizeof(int));

    pthread_t tid[8];
    pipelinesumma2 thread_info[8];
    pthread_mutex_t *lock;
    lock = (pthread_mutex_t *)malloc(sizeof(pthread_mutex_t));
    pthread_mutex_init(lock, NULL);
    for (int k = 0; k < 8; k++)
    {
        if (k < 4)
        {
            thread_info[k].id = k;
            thread_info[k].res = result;
            thread_info[k].x_data1_1 = (k / 2) * (n / 2);
            thread_info[k].y_data1_1 = (k % 2) * (n / 2);
            thread_info[k].x_data2_1 = (k / 2) * (n / 2);
            thread_info[k].y_data2_1 = (k % 2) * (n / 2);
            thread_info[k].x_data1_2 = (k / 2) * (n / 2);
            thread_info[k].y_data1_2 = (k % 2) * (n / 2);
            thread_info[k].x_data2_2 = (k / 2) * (n / 2);
            thread_info[k].y_data2_2 = (k % 2) * (n / 2);
            thread_info[k].data_ptr = data_ptr;
            thread_info[k].data_ptr2 = data_ptr2;
            thread_info[k].n = n / 2; // Assume that n is divisible by 2
            thread_info[k].lock = lock;
            pthread_create(&tid[k], NULL, pipeline_mult2, (void *)&thread_info[k]);
        }
        else
        {
            thread_info[k].id = k;
            thread_info[k].res = result;
            thread_info[k].x_data1_1 = ((k - 4) / 2) * (n / 2);
            thread_info[k].y_data1_1 = ((k - 4) % 2) * (n / 2);
            thread_info[k].x_data2_1 = ((k - 4) / 2) * (n / 2);
            thread_info[k].y_data2_1 = ((k - 4) % 2) * (n / 2);
            thread_info[k].x_data1_2 = ((k - 4) / 2) * (n / 2);
            thread_info[k].y_data1_2 = ((k - 4 + 1) % 2) * (n / 2);
            thread_info[k].x_data2_2 = ((k - 4 + 1) % 2) * (n / 2);
            thread_info[k].y_data2_2 = ((k - 4) % 2) * (n / 2);
            thread_info[k].data_ptr = data_ptr;
            thread_info[k].data_ptr2 = data_ptr2;
            thread_info[k].n = n / 2; // Assume that n is divisible by 2
            thread_info[k].lock = lock;
            pthread_create(&tid[k], NULL, pipeline_mult2, (void *)&thread_info[k]);
        }
    }
    for (int k = 0; k < 8; k++)
    {
        pthread_join(tid[k], NULL);
    }
    pthread_mutex_destroy(lock);
    free(lock);
    return result;
}

int main()
{
    // Prep the timing structs
    struct timespec begin_seq, end_seq;
    struct timespec begin_summa, end_summa;
    struct timespec begin_pipeline1, end_pipeline1;
    struct timespec begin_pipeline, end_pipeline;
    int size = 1800;

    int **dataSample = (int **)malloc(size * sizeof(int *));
    for (int i = 0; i < size; i++)
        dataSample[i] = (int *)malloc(size * sizeof(int));

    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            dataSample[i][j] = j;
        }
    }

    int **dataSample2 = (int **)malloc(size * sizeof(int *));
    for (int i = 0; i < size; i++)
        dataSample2[i] = (int *)malloc(size * sizeof(int));

    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            dataSample2[i][j] = j;
        }
    }

    clock_gettime(CLOCK_REALTIME, &begin_seq);
    int **res = naive_matrix_mult(dataSample, dataSample2, size);
    clock_gettime(CLOCK_REALTIME, &end_seq);

    long seconds_seq = end_seq.tv_sec - begin_seq.tv_sec;
    long nanos_seq = end_seq.tv_nsec - begin_seq.tv_nsec;
    double elapsed_seq = seconds_seq * 1000 + nanos_seq * 1e-6; // convert to ms
    std::cout << "Sequential: " << elapsed_seq << " ms" << std::endl;

    clock_gettime(CLOCK_REALTIME, &begin_pipeline1);
    int **resPipelined1 = pipelined_summa(dataSample, dataSample2, size);
    clock_gettime(CLOCK_REALTIME, &end_pipeline1);

    long seconds_pipelined1 = end_pipeline1.tv_sec - begin_pipeline1.tv_sec;
    long nanos_pipelined1 = end_pipeline1.tv_nsec - begin_pipeline1.tv_nsec;
    double elapsed_pipelined1 = seconds_pipelined1 * 1000 + nanos_pipelined1 * 1e-6; // convert to ms
    std::cout << "Pipelined summa 1: " << elapsed_pipelined1 << " ms" << std::endl;

    clock_gettime(CLOCK_REALTIME, &begin_pipeline);
    int **resPipelined = pipelined_summa2(dataSample, dataSample2, size);
    clock_gettime(CLOCK_REALTIME, &end_pipeline);

    long seconds_pipelined = end_pipeline.tv_sec - begin_pipeline.tv_sec;
    long nanos_pipelined = end_pipeline.tv_nsec - begin_pipeline.tv_nsec;
    double elapsed_pipelined = seconds_pipelined * 1000 + nanos_pipelined * 1e-6; // convert to ms
    std::cout << "Pipelined summa 2: " << elapsed_pipelined << " ms" << std::endl;

    clock_gettime(CLOCK_REALTIME, &begin_summa);
    int **resParallel = parallel_summa(dataSample, dataSample2, size);
    clock_gettime(CLOCK_REALTIME, &end_summa);

    long seconds_parallel = end_summa.tv_sec - begin_summa.tv_sec;
    long nanos_parallel = end_summa.tv_nsec - begin_summa.tv_nsec;
    double elapsed_parallel = seconds_parallel * 1000 + nanos_parallel * 1e-6; // convert to ms
    std::cout << "Parallel summa: " << elapsed_parallel << " ms" << std::endl;

    std::cout << compareMatrix(res, resPipelined1, size) << std::endl;
    std::cout << compareMatrix(res, resPipelined, size) << std::endl;
    std::cout << compareMatrix(res, resParallel, size) << std::endl;

    free(dataSample);
    free(dataSample2);
    return 0;
}

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <cuda.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <time.h>

#define MAXLINE 2000

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

// Funzione per calcolare la distanza euclidea (versione GPU)
__device__ float euclideanDistance(float *point, float *center, int samples) {
    float dist = 0.0;
    for (int i = 0; i < samples; i++) {
        float diff = point[i] - center[i];
        dist += diff * diff;
    }
    return sqrt(dist);
}

// Kernel CUDA per assegnare i punti ai cluster
__global__ void assignClusters(float *data, float *centroids, int *clusterMap, int samples, int K, int pointsPerProcess) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < pointsPerProcess) {
        float minDist = FLT_MAX;
        int cluster_id = 0;
        for (int j = 0; j < K; j++) {
            float dist = euclideanDistance(&data[idx * samples], &centroids[j * samples], samples);
            if (dist < minDist) {
                minDist = dist;
                cluster_id = j + 1;
            }
        }
        clusterMap[idx] = cluster_id;
    }
}

// Funzioni di utility
void showFileError(int error, char* filename) {
    printf("Error\n");
    switch (error) {
        case -1:
            fprintf(stderr, "\tFile %s has too many columns.\n", filename);
            fprintf(stderr, "\tThe maximum number of columns has been exceeded. MAXLINE: %d.\n", MAXLINE);
            break;
        case -2:
            fprintf(stderr, "Error reading file: %s.\n", filename);
            break;
        case -3:
            fprintf(stderr, "Error writing file: %s.\n", filename);
            break;
    }
    fflush(stderr);
}

int readInput(char* filename, int *lines, int *samples) {
    FILE *fp;
    char line[MAXLINE] = "";
    char *ptr;
    const char *delim = "\t";
    int contlines, contsamples = 0;

    contlines = 0;

    if ((fp = fopen(filename, "r")) != NULL) {
        while (fgets(line, MAXLINE, fp) != NULL) {
            if (strchr(line, '\n') == NULL) {
                return -1;
            }
            contlines++;
            ptr = strtok(line, delim);
            contsamples = 0;
            while (ptr != NULL) {
                contsamples++;
                ptr = strtok(NULL, delim);
            }
        }
        fclose(fp);
        *lines = contlines;
        *samples = contsamples;
        return 0;
    } else {
        return -2;
    }
}

int readInput2(char* filename, float* data) {
    FILE *fp;
    char line[MAXLINE] = "";
    char *ptr;
    const char *delim = "\t";
    int i = 0;

    if ((fp = fopen(filename, "rt")) != NULL) {
        while (fgets(line, MAXLINE, fp) != NULL) {
            ptr = strtok(line, delim);
            while (ptr != NULL) {
                data[i] = atof(ptr);
                i++;
                ptr = strtok(NULL, delim);
            }
        }
        fclose(fp);
        return 0;
    } else {
        return -2;
    }
}

int writeResult(int *clusterMap, int lines, const char* filename) {
    FILE *fp;

    if ((fp = fopen(filename, "wt")) != NULL) {
        for (int i = 0; i < lines; i++) {
            fprintf(fp, "%d\n", clusterMap[i]);
        }
        fclose(fp);
        return 0;
    } else {
        return -3;
    }
}

void initCentroids(const float *data, float* centroids, int* centroidPos, int samples, int K) {
    int i;
    int idx;
    for (i = 0; i < K; i++) {
        idx = centroidPos[i];
        memcpy(&centroids[i * samples], &data[idx * samples], (samples * sizeof(float)));
    }
}

void zeroFloatMatrix(float *matrix, int rows, int columns) {
    int i, j;
    for (i = 0; i < rows; i++)
        for (j = 0; j < columns; j++)
            matrix[i * columns + j] = 0.0;
}

void zeroIntArray(int *array, int size) {
    int i;
    for (i = 0; i < size; i++)
        array[i] = 0;
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    clock_t start_time, end_time;
    double comp_time;

    if (argc != 7) {
        if (rank == 0) {
            fprintf(stderr, "Usage: mpirun -np <num_processes> %s <input_file> <num_clusters> <max_iterations> <min_changes> <threshold> <output_file>\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    char *inputFile = argv[1];
    int K = atoi(argv[2]);
    int maxIterations = atoi(argv[3]);
    float minChangesPercent = atof(argv[4]);
    float maxThreshold = atof(argv[5]);
    char *outputFile = argv[6];

    int lines = 0, samples = 0;
    float *data = NULL;
    float *centroids = NULL;
    int *clusterMap = NULL;

    int *displs = NULL;
    int *recvcounts = NULL;

    displs = (int *)malloc(size * sizeof(int));
    recvcounts = (int *)malloc(size * sizeof(int));
    if (displs == NULL || recvcounts == NULL) {
        fprintf(stderr, "Memory allocation error.\n");
        MPI_Abort(MPI_COMM_WORLD, -4);
    }

    if (rank == 0) {

        start_time = clock();
        int error = readInput(inputFile, &lines, &samples);
        if (error != 0) {
            showFileError(error, inputFile);
            MPI_Abort(MPI_COMM_WORLD, error);
        }

        if (lines < K) {
            fprintf(stderr, "Error: Number of points (%d) must be greater than or equal to the number of clusters (%d).\n", lines, K);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }

        data = (float *)calloc(lines * samples, sizeof(float));
        if (data == NULL) {
            fprintf(stderr, "Memory allocation error.\n");
            MPI_Abort(MPI_COMM_WORLD, -4);
        }

        error = readInput2(inputFile, data);
        if (error != 0) {
            showFileError(error, inputFile);
            MPI_Abort(MPI_COMM_WORLD, error);
        }

        centroids = (float *)calloc(K * samples, sizeof(float));
        int *centroidPos = (int *)calloc(K, sizeof(int));
        if (centroids == NULL || centroidPos == NULL) {
            fprintf(stderr, "Memory allocation error.\n");
            MPI_Abort(MPI_COMM_WORLD, -4);
        }

        srand(0);  // Usa lo stesso seed per garantire la riproducibilità
        for (int i = 0; i < K; i++) {
            centroidPos[i] = rand() % lines;
        }
        initCentroids(data, centroids, centroidPos, samples, K);
        free(centroidPos);
    }

    MPI_Bcast(&lines, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&samples, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        centroids = (float *)calloc(K * samples, sizeof(float));
    }
    MPI_Bcast(centroids, K * samples, MPI_FLOAT, 0, MPI_COMM_WORLD);

    int minChanges = (int)(lines * minChangesPercent / 100.0);

    int pointsPerProcess = lines / size;
    int remainder = lines % size;
    int start = rank * pointsPerProcess + MIN(rank, remainder);
    int end = start + pointsPerProcess + (rank < remainder ? 1 : 0);

    float *localData = (float *)malloc((end - start) * samples * sizeof(float));
    int *localClusterMap = (int *)malloc((end - start) * sizeof(int));
    int *previousClusterMap = (int *)malloc((end - start) * sizeof(int));

    if (localData == NULL || localClusterMap == NULL || previousClusterMap == NULL) {
        fprintf(stderr, "Memory allocation error.\n");
        MPI_Abort(MPI_COMM_WORLD, -4);
    }

    if (rank == 0) {
        for (int i = 0; i < size; i++) {
            recvcounts[i] = (lines / size) * samples;
            if (i < lines % size) {
                recvcounts[i] += samples;
            }
            displs[i] = (i == 0) ? 0 : displs[i - 1] + recvcounts[i - 1];
        }
    }

    MPI_Scatterv(data, recvcounts, displs, MPI_FLOAT, localData, (end - start) * samples, MPI_FLOAT, 0, MPI_COMM_WORLD);

    float *d_data, *d_centroids;
    int *d_clusterMap;

    cudaMalloc(&d_data, (end - start) * samples * sizeof(float));
    cudaMalloc(&d_centroids, K * samples * sizeof(float));
    cudaMalloc(&d_clusterMap, (end - start) * sizeof(int));

    cudaMemcpy(d_data, localData, (end - start) * samples * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, centroids, K * samples * sizeof(float), cudaMemcpyHostToDevice);

    int it = 0;
    int changes = 0;
    float maxDist = 0.0;

    do {
        it++;

        int blockSize = 256;
        int gridSize = (end - start + blockSize - 1) / blockSize;
        assignClusters<<<gridSize, blockSize>>>(d_data, d_centroids, d_clusterMap, samples, K, end - start);

        // Sincronizza il dispositivo per garantire che il kernel sia completato
        cudaDeviceSynchronize();

        // Copia i risultati dalla GPU alla CPU
        cudaMemcpy(localClusterMap, d_clusterMap, (end - start) * sizeof(int), cudaMemcpyDeviceToHost);

        float *newCentroids = (float *)calloc(K * samples, sizeof(float));
        int *pointsPerCluster = (int *)calloc(K, sizeof(int));

        for (int i = 0; i < end - start; i++) {
            int cluster_id = localClusterMap[i] - 1;
            pointsPerCluster[cluster_id]++;
            for (int j = 0; j < samples; j++) {
                newCentroids[cluster_id * samples + j] += localData[i * samples + j];
            }
        }

        MPI_Allreduce(MPI_IN_PLACE, newCentroids, K * samples, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, pointsPerCluster, K, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        for (int i = 0; i < K; i++) {
            if (pointsPerCluster[i] != 0) {
                for (int j = 0; j < samples; j++) {
                    newCentroids[i * samples + j] /= pointsPerCluster[i];
                }
            } else {
                for (int j = 0; j < samples; j++) {
                    newCentroids[i * samples + j] = centroids[i * samples + j];
                }
            }
        }

        maxDist = 0.0;
        for (int i = 0; i < K; i++) {
            float dist = 0.0;
            for (int j = 0; j < samples; j++) {
                float diff = centroids[i * samples + j] - newCentroids[i * samples + j];
                dist += diff * diff;
            }
            dist = sqrt(dist);
            if (dist > maxDist) {
                maxDist = dist;
            }
        }

        memcpy(centroids, newCentroids, K * samples * sizeof(float));
        // Copia i nuovi centroidi sulla GPU
        cudaMemcpy(d_centroids, centroids, K * samples * sizeof(float), cudaMemcpyHostToDevice);

        changes = 0;
        for (int i = 0; i < end - start; i++) {
            if (localClusterMap[i] != previousClusterMap[i]) {
                changes++;
            }
        }
        memcpy(previousClusterMap, localClusterMap, (end - start) * sizeof(int));

        MPI_Allreduce(MPI_IN_PLACE, &changes, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        free(newCentroids);
        free(pointsPerCluster);

    } while (it < maxIterations && (changes > minChanges || maxDist > maxThreshold));

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Termination condition:\n");
        if (changes <= minChanges) {
            printf("Minimum number of changes reached: %d\n", changes);
        } else if (it >= maxIterations) {
            printf("Maximum number of iterations reached: %d\n", it);
        } else {
            printf("Centroid update precision reached: %f\n", maxDist);
        }
    }

    if (rank == 0) {
        clusterMap = (int *)malloc(lines * sizeof(int));
    }

    if (rank == 0) {
        for (int i = 0; i < size; i++) {
            recvcounts[i] = (lines / size);
            if (i < lines % size) {
                recvcounts[i] += 1;
            }
            displs[i] = (i == 0) ? 0 : displs[i - 1] + recvcounts[i - 1];
        }
    }

    MPI_Gatherv(localClusterMap, end - start, MPI_INT, clusterMap, recvcounts, displs, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        free(recvcounts);
        free(displs);
    }

    if (rank == 0) {
        int error = writeResult(clusterMap, lines, outputFile);
        if (error != 0) {
            showFileError(error, outputFile);
            MPI_Abort(MPI_COMM_WORLD, error);
        }
    }

    free(localData);
    free(localClusterMap);
    free(previousClusterMap);
    cudaFree(d_data);
    cudaFree(d_centroids);
    cudaFree(d_clusterMap);

    if (rank == 0) {
        free(data);
        free(centroids);
        free(clusterMap);

        end_time = clock();
        comp_time = ((double)(end-start)) / CLOCKS_PER_SEC;
        printf("exec time: %f.\n", comp_time);
    }

    MPI_Finalize();
    return 0;
}

/*
 * k-Means clustering algorithm
 *
 * pthread version.
 *
 * Parallel computing (Degree in Computer Engineering)
 * 2022/2023
 *
 * Version: 1.0
 *
 * (c) 2022 Diego García-Álvarez, Arturo Gonzalez-Escribano
 * Grupo Trasgo, Universidad de Valladolid (Spain)
 *
 * This work is licensed under a Creative Commons Attribution-ShareAlike 4.0 International License.
 * https://creativecommons.org/licenses/by-sa/4.0/
 */

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <float.h>
#include <pthread.h>

#define MAXLINE 2000
#define MAXCAD 200
#define THREADS 4

// Macros
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))

void showFileError(int error, char* filename) {
    printf("Error\n");
    switch (error) {
        case -1:
            fprintf(stderr,"\tFile %s has too many columns.\n", filename);
            fprintf(stderr,"\tThe maximum number of columns has been exceeded. MAXLINE: %d.\n", MAXLINE);
            break;
        case -2:
            fprintf(stderr,"Error reading file: %s.\n", filename);
            break;
        case -3:
            fprintf(stderr,"Error writing file: %s.\n", filename);
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

    if ((fp=fopen(filename,"r"))!=NULL) {
        while(fgets(line, MAXLINE, fp)!= NULL) {
            if (strchr(line, '\n') == NULL) {
                return -1;
            }
            contlines++;       
            ptr = strtok(line, delim);
            contsamples = 0;
            while(ptr != NULL) {
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
    
    if ((fp=fopen(filename,"rt"))!=NULL) {
        while(fgets(line, MAXLINE, fp)!= NULL) {         
            ptr = strtok(line, delim);
            while(ptr != NULL) {
                data[i] = atof(ptr);
                i++;
                ptr = strtok(NULL, delim);
            }
        }
        fclose(fp);
        return 0;
    } else {
        return -2; // No file found
    }
}

int writeResult(int *classMap, int lines, const char* filename) {	
    FILE *fp;
    
    if ((fp=fopen(filename,"wt"))!=NULL) {
        for(int i=0; i<lines; i++) {
            fprintf(fp,"%d\n",classMap[i]);
        }
        fclose(fp);  
        return 0;
    } else {
        return -3; // No file found
    }
}

void initCentroids(const float *data, float* centroids, int* centroidPos, int samples, int K) {
    int i;
    int idx;
    for(i=0; i<K; i++) {
        idx = centroidPos[i];
        memcpy(&centroids[i*samples], &data[idx*samples], (samples*sizeof(float)));
    }
}

float euclideanDistance(float *point, float *center, int samples) {
    float dist=0.0;
    for(int i=0; i<samples; i++) {
        dist+= (point[i]-center[i])*(point[i]-center[i]);
    }
    dist = sqrt(dist);
    return(dist);
}

void zeroFloatMatriz(float *matrix, int rows, int columns) {
    int i,j;
    for (i=0; i<rows; i++)
        for (j=0; j<columns; j++)
            matrix[i*columns+j] = 0.0;	
}

void zeroIntArray(int *array, int size) {
    int i;
    for (i=0; i<size; i++)
        array[i] = 0;	
}

typedef struct {
    int thread_id;              // ID del thread
    int start;                  // Indice di inizio dei punti assegnati al thread
    int end;                    // Indice di fine dei punti assegnati al thread
    int lines;                  // Numero totale di punti
    int samples;                // Numero di dimensioni (attributi) di ciascun punto
    int K;                      // Numero di cluster
    float *data;                // Puntatore ai dati (punti)
    float *centroids;           // Puntatore ai centroidi
    int *classMap;              // Puntatore alla mappa dei cluster
    int *pointsPerClass;        // Puntatore al numero di punti per cluster (locale)
    float *auxCentroids;        // Puntatore alle somme parziali per aggiornare i centroidi (locale)
} ThreadData;

void *assign_points(void *arg) {
    ThreadData *t_data = (ThreadData *)arg;
    int i, j, class;
    float dist, minDist;

    // Inizializza le strutture dati locali
    int *local_pointsPerClass = (int *)calloc(t_data->K, sizeof(int));
    float *local_auxCentroids = (float *)calloc(t_data->K * t_data->samples, sizeof(float));

    for (i = t_data->start; i < t_data->end; i++) {
        class = 1;
        minDist = FLT_MAX;

        for (j = 0; j < t_data->K; j++) {
            dist = euclideanDistance(&t_data->data[i * t_data->samples], &t_data->centroids[j * t_data->samples], t_data->samples);
            if (dist < minDist) {
                minDist = dist;
                class = j + 1;
            }
        }

        t_data->classMap[i] = class;

        // Aggiorna le strutture dati locali
        local_pointsPerClass[class - 1]++;
        for (j = 0; j < t_data->samples; j++) {
            local_auxCentroids[(class - 1) * t_data->samples + j] += t_data->data[i * t_data->samples + j];
        }
    }

    // Copia i risultati locali nelle strutture dati globali
    for (i = 0; i < t_data->K; i++) {
        t_data->pointsPerClass[i] += local_pointsPerClass[i];
        for (j = 0; j < t_data->samples; j++) {
            t_data->auxCentroids[i * t_data->samples + j] += local_auxCentroids[i * t_data->samples + j];
        }
    }

    free(local_pointsPerClass);
    free(local_auxCentroids);

    return NULL;
}

int main(int argc, char *argv[]) {
    if(argc !=  7) {
        fprintf(stderr,"EXECUTION ERROR K-MEANS: Parameters are not correct.\n");
        fprintf(stderr,"./KMEANS [Input Filename] [Number of clusters] [Number of iterations] [Number of changes] [Threshold] [Output data file]\n");
        fflush(stderr);
        exit(EXIT_FAILURE);
    }
    
    int lines = 0, samples = 0;  
    int error = readInput(argv[1], &lines, &samples);
    if(error != 0) {
        showFileError(error,argv[1]);
        exit(EXIT_FAILURE);
    }
    
    float *data = (float*)calloc(lines*samples,sizeof(float));
    if (data == NULL) {
        fprintf(stderr,"Memory allocation error.\n");
        exit(EXIT_FAILURE);
    }
    error = readInput2(argv[1], data);
    if(error != 0) {
        showFileError(error,argv[1]);
        exit(EXIT_FAILURE);
    }

    int K = atoi(argv[2]); 
    int maxIterations = atoi(argv[3]);
    int minChanges = (int)(lines * atof(argv[4]) / 100.0);
    float maxThreshold = atof(argv[5]);

    int *centroidPos = (int*)calloc(K,sizeof(int));
    float *centroids = (float*)calloc(K*samples,sizeof(float));
    int *classMap = (int*)calloc(lines,sizeof(int));
    int *prevClassMap = (int *)calloc(lines, sizeof(int));

    if (centroidPos == NULL || centroids == NULL || classMap == NULL || prevClassMap == NULL) {
        fprintf(stderr,"Memory allocation error.\n");
        exit(EXIT_FAILURE);
    }
    
    srand(0);
    for (int i = 0; i < K; i++) {
        centroidPos[i] = rand() % lines;
    }

    initCentroids(data, centroids, centroidPos, samples, K);
    printf("\n\tData file: %s \n\tPoints: %d\n\tDimensions: %d\n", argv[1], lines, samples);
    printf("\tNumber of clusters: %d\n", K);
    printf("\tMaximum number of iterations: %d\n", maxIterations);
    printf("\tMinimum number of changes: %d [%g%% of %d points]\n", minChanges, atof(argv[4]), lines);
    printf("\tMaximum centroid precision: %f\n", maxThreshold);

    pthread_t threads[THREADS];
    ThreadData thread_data[THREADS];

    int *pointsPerClass = (int *)calloc(K, sizeof(int));
    float *auxCentroids = (float *)calloc(K * samples, sizeof(float));

    if (pointsPerClass == NULL || auxCentroids == NULL) {
        fprintf(stderr, "Memory allocation error for pointsPerClass or auxCentroids.\n");
        exit(EXIT_FAILURE);
    }

    int points_per_thread = lines / THREADS;
    int remainder = lines % THREADS;
    int changes = 0;
    int it = 0;
    float maxDist;

    do {
        changes = 0;
        it++;
        memcpy(prevClassMap, classMap, lines * sizeof(int));

        int current_start = 0;
        for (int i = 0; i < THREADS; i++) {
            thread_data[i].thread_id = i;
            thread_data[i].start = current_start;
            thread_data[i].end = current_start + points_per_thread + (i < remainder ? 1 : 0);
            thread_data[i].lines = lines;
            thread_data[i].samples = samples;
            thread_data[i].K = K;
            thread_data[i].data = data;
            thread_data[i].centroids = centroids;
            thread_data[i].classMap = classMap;
            thread_data[i].pointsPerClass = pointsPerClass;
            thread_data[i].auxCentroids = auxCentroids;

            current_start = thread_data[i].end;
        }

        for (int i = 0; i < THREADS; i++) {
            pthread_create(&threads[i], NULL, assign_points, (void *)&thread_data[i]);
        }

        for (int i = 0; i < THREADS; i++) {
            pthread_join(threads[i], NULL);
        }

        for (int i = 0; i < lines; i++) {
            if (classMap[i] != prevClassMap[i]) {
                changes++;
            }
        }

        for (int i = 0; i < K; i++) {
            if (pointsPerClass[i] > 0) {
                for (int j = 0; j < samples; j++) {
                    centroids[i * samples + j] = auxCentroids[i * samples + j] / pointsPerClass[i];
                }
            }
        }

        maxDist = FLT_MIN;
        for (int i = 0; i < K; i++) {
            float dist = euclideanDistance(&centroids[i * samples], &auxCentroids[i * samples], samples);
            if (dist > maxDist) {
                maxDist = dist;
            }
        }

        zeroIntArray(pointsPerClass, K);
        zeroFloatMatriz(auxCentroids, K, samples);

    } while ((changes > minChanges) && (it < maxIterations) && (maxDist > maxThreshold));

    error = writeResult(classMap, lines, argv[6]);
    if (error != 0) {
        showFileError(error, argv[6]);
        exit(EXIT_FAILURE);
    }

    free(data);
    free(centroidPos);
    free(centroids);
    free(classMap);
    free(pointsPerClass);
    free(auxCentroids);
    free(prevClassMap);

    exit(EXIT_SUCCESS);
}

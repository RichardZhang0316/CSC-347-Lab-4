/**
 * This program is used for test. Please neglect.
 **/

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <curand_kernel.h>
#include <math.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define N 9999     // number of bodies
#define MASS 0     // row in array for mass
#define X_POS 1    // row in array for x position
#define Y_POS 2    // row in array for y position
#define Z_POS 3    // row in array for z position
#define X_VEL 4    // row in array for x velocity
#define Y_VEL 5    // row in array for y velocity
#define Z_VEL 6    // row in array for z velocity
#define G 200      // "gravitational constant" (not really)
#define MU 0.001   // "frictional coefficient"
#define BOXL 100.0 // periodic boundary box length

float dt = 0.05; // time interval

float body[10000][7]; // data array of bodies

// The function calculates the cross product of 3-dimensional vectors
void crossProduct(float vect_A[], float vect_B[], float cross_P[])
{
    cross_P[0] = vect_A[1] * vect_B[2] - vect_A[2] * vect_B[1];
    cross_P[1] = vect_A[2] * vect_B[0] - vect_A[0] * vect_B[2];
    cross_P[2] = vect_A[0] * vect_B[1] - vect_A[1] * vect_B[0];
}

void norm(float &x, float &y, float &z)
{
    float mag = sqrt(x * x + y * y + z * z);
    x /= mag;
    y /= mag;
    z /= mag;
}

// Kernel function that calculates the forces between bodies and update the positions & velocities of bodies
__global__ void calculateForces(float *body, float dt, int bodiesPerThread)
{

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    x *= bodiesPerThread; // Each thread will process bodiesPerThread of bodies

    for (int j = 0; j < bodiesPerThread; j++) // Loop for processing bodiesPerThread of bodies
    {
        int current_x = x + j;
        if (current_x < N)
        {
            // Initialize forces to zero
            float Fx = 0.0f;
            float Fy = 0.0f;
            float Fz = 0.0f;
            for (int i = 0; i < N; i++)
            {
                // calculate the force between body current_x and body i
                if (i != current_x)
                {
                    float x_diff = body[i * 7 + X_POS] - body[current_x * 7 + X_POS];
                    float y_diff = body[i * 7 + Y_POS] - body[current_x * 7 + Y_POS];
                    float z_diff = body[i * 7 + Z_POS] - body[current_x * 7 + Z_POS];

                    float rr = (x_diff * x_diff + y_diff * y_diff + z_diff * z_diff);
                    float r = sqrt(rr);

                    if (r > 50.0)
                    {
                        float F = -1.0 * (G * body[i * 7 + MASS] * body[current_x * 7 + MASS]) / rr;
                        // We cannot call CPU functions in a kernel function, so we have to normalize them here without calling norm()
                        float mag = sqrt(x_diff * x_diff + y_diff * y_diff + z_diff * z_diff);
                        x_diff /= mag;
                        y_diff /= mag;
                        z_diff /= mag;

                        Fx += x_diff * F;
                        Fy += y_diff * F;
                        Fz += z_diff * F;
                    }
                }
            }
            // Update velocities
            body[current_x * 7 + X_VEL] += Fx * dt / body[current_x * 7 + MASS];
            body[current_x * 7 + Y_VEL] += Fy * dt / body[current_x * 7 + MASS];
            body[current_x * 7 + Z_VEL] += Fz * dt / body[current_x * 7 + MASS];
            // Update positions
            body[current_x * 7 + X_POS] += body[current_x * 7 + X_VEL] * dt;
            body[current_x * 7 + Y_POS] += body[current_x * 7 + Y_VEL] * dt;
            body[current_x * 7 + Z_POS] += body[current_x * 7 + Z_VEL] * dt;

            // // Update velocities
            // atomicAdd(&body[current_x * 7 + X_VEL], Fx * dt / body[current_x * 7 + MASS]);
            // atomicAdd(&body[current_x * 7 + Y_VEL], Fy * dt / body[current_x * 7 + MASS]);
            // atomicAdd(&body[current_x * 7 + Z_VEL], Fz * dt / body[current_x * 7 + MASS]);

            // // Update positions
            // atomicAdd(&body[current_x * 7 + X_POS], body[current_x * 7 + X_VEL] * dt);
            // atomicAdd(&body[current_x * 7 + Y_POS], body[current_x * 7 + Y_VEL] * dt);
            // atomicAdd(&body[current_x * 7 + Z_POS], body[current_x * 7 + Z_VEL] * dt);
        }
    }
}

int main(int argc, char **argv)
{
    int tmax = 0;
    float serialTimeSpent = 0.0;
    float parallelTimeSpent = 0.0;

    cudaEvent_t serialStart, serialStop, parallelStart, parallelStop;
    cudaEventCreate(&serialStart);
    cudaEventCreate(&serialStop);
    cudaEventCreate(&parallelStart);
    cudaEventCreate(&parallelStop);

    // Record start time for serial operations
    cudaEventRecord(serialStart, 0);

    // Determine if there are two arguments on the command line
    if (argc != 2)
    {
        printf("Command line arguments are not enough: %s \n", argv[0]);
        return 1;
    }

    tmax = atoi(argv[1]);

    // Determine if the number of timesteps entered by users is legitamate
    if (tmax <= 0)
    {
        printf("Number of timesteps should not less than 1\n");
        return 2;
    }

    // assign each body a random initial positions and velocities
    srand48(time(NULL));
    float vect_A[3];
    float vect_B[3];
    float cross_P[3];

    // black hole at the center
    body[0][MASS] = 4000.0;
    body[0][X_POS] = 0.0;
    body[0][Y_POS] = 0.0;
    body[0][Z_POS] = 0.0;
    body[0][X_VEL] = 0.0;
    body[0][Y_VEL] = 0.0;
    body[0][Z_VEL] = 0.0;

    float *body_d;

    cudaMalloc((void **)&body_d, N * 7 * sizeof(float));

    for (int i = 1; i < N; i++)
    {
        body[i][MASS] = 0.001;

        // TODO: initial coordinates centered on origin, ranging -150.0 to +150.0
        body[i][X_POS] = drand48() * 300 - 150;
        body[i][Y_POS] = drand48() * 300 - 150;
        body[i][Z_POS] = drand48() * 300 - 150;

        // initial velocities directions around z-axis
        vect_A[0] = body[i][X_POS];
        vect_A[1] = body[i][Y_POS];
        vect_A[2] = body[i][Z_POS];

        norm(vect_A[0], vect_A[1], vect_A[2]);
        vect_B[0] = 0.0;
        vect_B[1] = 0.0;
        vect_B[2] = 1.0;
        cross_P[0] = 0.0;
        cross_P[1] = 0.0;
        cross_P[2] = 0.0;
        crossProduct(vect_A, vect_B, cross_P);

        // random initial velocities magnitudes
        body[i][X_VEL] = drand48() * 100 * cross_P[0];
        body[i][Y_VEL] = drand48() * 100 * cross_P[1];
        body[i][Z_VEL] = drand48() * 100 * cross_P[2];
    }

    // Copy the body array from the CPU to GPU
    cudaMemcpy(body_d, body, N * 7 * sizeof(float), cudaMemcpyHostToDevice);

    // Used for test!!! Open the file "NBody.pdb" for writing
    FILE *output_file = fopen("NBody.pdb", "w");
    if (output_file == NULL)
    {
        fprintf(stderr, "Error opening file for writing.\n");
        exit(-1);
    }
    // Used for test!!! Print out initial positions in PDB format
    fprintf(output_file, "MODEL %8d\n", 0);
    for (int i = 0; i < N; i++)
    {
        fprintf(output_file, "%s%7d  %s %s %s%4d    %8.3f%8.3f%8.3f  %4.2f  %4.3f\n",
                "ATOM", i + 1, "CA ", "GLY", "A", i + 1, body[i][X_POS], body[i][Y_POS], body[i][Z_POS], 1.00, 0.00);
    }
    fprintf(output_file, "TER\nENDMDL\n");

    // print out initial positions in PDB format
    printf("MODEL %8d\n", 0);
    for (int i = 0; i < N; i++)
    {
        printf("%s%7d  %s %s %s%4d    %8.3f%8.3f%8.3f  %4.2f  %4.3f\n",
               "ATOM", i + 1, "CA ", "GLY", "A", i + 1, body[i][X_POS], body[i][Y_POS], body[i][Z_POS], 1.00, 0.00);
    }
    printf("TER\nENDMDL\n");

    // step through each time step
    for (int t = 0; t < tmax; t++)
    {
        // Determine the size of grid & block as well as the number of threads we have
        // We need to change the number of threads to to calculate a speedup factor and plot it with respect to different
        // number of processors (threads).
        int threads = 1000;
        int bodiesPerThread = (N + threads - 1) / threads;
        dim3 blockDim(32, 32);
        dim3 gridDim((threads + 1024) / 1024);
        // TODO: initialize forces to zero (no longer need to do it here. We did this in the kernel function)

        // Record start time for parallel operations
        cudaEventRecord(parallelStart, 0);
        calculateForces<<<gridDim, blockDim>>>(body_d, dt, bodiesPerThread);
        cudaDeviceSynchronize();
        cudaMemcpy(body, body_d, N * 7 * sizeof(float), cudaMemcpyDeviceToHost);
        // Record stop time for parallel operations
        cudaEventRecord(parallelStop, 0);
        cudaEventSynchronize(parallelStop);

        float elapsedTime;
        cudaEventElapsedTime(&elapsedTime, parallelStart, parallelStop);
        parallelTimeSpent += elapsedTime;

        // print out positions in PDB format
        printf("MODEL %8d\n", t + 1);
        for (int i = 0; i < N; i++)
        {
            printf("%s%7d  %s %s %s%4d    %8.3f%8.3f%8.3f  %4.2f  %4.3f\n", "ATOM", i + 1, "CA ", "GLY", "A", i + 1, body[i][X_POS], body[i][Y_POS], body[i][Z_POS], 1.00, 0.00);
        }
        printf("TER\nENDMDL tmax: %d\n", tmax);

        // Used for test!!!
        fprintf(output_file, "MODEL %8d\n", t + 1);
        for (int i = 0; i < N; i++)
        {
            fprintf(output_file, "%s%7d  %s %s %s%4d    %8.3f%8.3f%8.3f  %4.2f  %4.3f\n", "ATOM", i + 1, "CA ", "GLY", "A", i + 1, body[i][X_POS], body[i][Y_POS], body[i][Z_POS], 1.00, 0.00);
        }
        fprintf(output_file, "TER\nENDMDL\n");

    } // end of time period loop
    // Record stop time for serial operations.
    cudaEventRecord(serialStop, 0);

    // Wait for all events to complete
    cudaEventSynchronize(serialStop);

    // Calculate and print the time spent for serial portion and parallel portion
    cudaEventElapsedTime(&serialTimeSpent, serialStart, serialStop);
    printf("Serial time (seconds):   %f \n", (serialTimeSpent - parallelTimeSpent) / 1000.0);
    printf("Parallel time (seconds): %f \n", parallelTimeSpent / 1000.0);

    // Clean up memory and events
    cudaEventDestroy(serialStart);
    cudaEventDestroy(serialStop);
    cudaEventDestroy(parallelStart);
    cudaEventDestroy(parallelStop);
    cudaFree(body_d);
}

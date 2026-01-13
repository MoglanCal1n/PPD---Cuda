#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <string>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace std;

const string INPUT_FILE = "date.txt";
const string OUTPUT_FILE = "output.txt";

const int K = 3;
const int RADIUS = K / 2;

__constant__ int d_C_const[K * K];

#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void computeRowConvolution(int* F, const int* prevRow, const int* currRow, int rowIdx, int M, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < N) {
        int sum = 0;

        for (int u = -RADIUS; u <= RADIUS; u++) {
            for (int v = -RADIUS; v <= RADIUS; v++) {

                int n_col = col + v;
                if (n_col < 0) n_col = 0;
                if (n_col >= N) n_col = N - 1;

                int valPixel = 0;


                if (u == -1) {
                    valPixel = prevRow[n_col];
                }
                else if (u == 0) {
                    valPixel = currRow[n_col];
                }
                else {
                    if (rowIdx == M - 1) {
                        valPixel = currRow[n_col];
                    } else {
                        valPixel = F[(rowIdx + 1) * N + n_col];
                    }
                }

                sum += valPixel * d_C_const[(u + RADIUS) * K + (v + RADIUS)];
            }
        }

        F[rowIdx * N + col] = sum;
    }
}

void readInput(int* &h_F, int* &h_C, int M, int N) {
    ifstream fin(INPUT_FILE);
    if (!fin) {
        cerr << "Error: Cannot open " << INPUT_FILE << ". Please generate it first." << endl; exit(1);
    }
    h_F = new int[M * N];
    h_C = new int[K * K];
    for (int i = 0; i < M * N; i++) fin >> h_F[i];
    for (int i = 0; i < K * K; i++) fin >> h_C[i];
    fin.close();
}

void writeOutput(const int* h_F, int M, int N) {
    ofstream fout(OUTPUT_FILE);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            fout << h_F[i * N + j] << " ";
        }
        fout << "\n";
    }
    fout.close();
}

int main(int argc, char* argv[]) {
    int M = 10;
    int N = 10;
    int threadsPerBlock = 256;

    if (argc >= 4) {
        M = stoi(argv[1]);
        N = stoi(argv[2]);
        threadsPerBlock = stoi(argv[3]);
    } else if (argc == 3) {
        M = stoi(argv[1]);
        N = stoi(argv[2]);
    } else {
        cout << "Running with default 10x10 and 256 threads per block. Usage: ./program <M> <N> <threadsPerBlock>" << endl;
    }

    int *h_F = nullptr, *h_C = nullptr;
    readInput(h_F, h_C, M, N);

    int *d_F;
    int *d_prevRow, *d_currRow;

    size_t sizeImage = M * N * sizeof(int);
    size_t sizeRow = N * sizeof(int);

    cudaCheckError(cudaMalloc(&d_F, sizeImage));
    cudaCheckError(cudaMalloc(&d_prevRow, sizeRow));
    cudaCheckError(cudaMalloc(&d_currRow, sizeRow));

    cudaCheckError(cudaMemcpy(d_F, h_F, sizeImage, cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpyToSymbol(d_C_const, h_C, K * K * sizeof(int)));

    cudaCheckError(cudaMemcpy(d_prevRow, d_F, sizeRow, cudaMemcpyDeviceToDevice));
    cudaCheckError(cudaMemcpy(d_currRow, d_F, sizeRow, cudaMemcpyDeviceToDevice));

    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    cout << "Starting convolution on " << M << "x" << N << " matrix..." << endl;
    auto start = chrono::high_resolution_clock::now();

    for (int i = 0; i < M; i++) {

        computeRowConvolution<<<blocksPerGrid, threadsPerBlock>>>(d_F, d_prevRow, d_currRow, i, M, N);
        cudaCheckError(cudaGetLastError());

        if (i < M - 1) {
            cudaCheckError(cudaMemcpy(d_prevRow, d_currRow, sizeRow, cudaMemcpyDeviceToDevice));

            cudaCheckError(cudaMemcpy(d_currRow, d_F + (i + 1) * N, sizeRow, cudaMemcpyDeviceToDevice));
        }
    }

    cudaCheckError(cudaDeviceSynchronize());

    auto end = chrono::high_resolution_clock::now();
    double duration = chrono::duration<double, milli>(end - start).count();
    cout << "Execution Time: " << duration << " ms" << endl;

    cudaCheckError(cudaMemcpy(h_F, d_F, sizeImage, cudaMemcpyDeviceToHost));

    writeOutput(h_F, M, N);

    delete[] h_F; delete[] h_C;
    cudaFree(d_F); cudaFree(d_prevRow); cudaFree(d_currRow);

    return 0;
}
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

#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void computeRowConvolution(int* F, const int* C, const int* prevRow, const int* currRow, int rowIdx, int M, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < N) {
        int sum = 0;
        int half_k = K / 2; 

        for (int u = -half_k; u <= half_k; u++) {     
            for (int v = -half_k; v <= half_k; v++) { 
                
                int valPixel;
                int n_col = col + v;

                if (n_col < 0) n_col = 0;
                if (n_col >= N) n_col = N - 1;

                
                if (u == -1) { 
                    if (rowIdx == 0) valPixel = currRow[n_col];
                    else valPixel = prevRow[n_col];
                } 
                else if (u == 0) { 
                    valPixel = currRow[n_col];
                } 
                else { 
                    if (rowIdx == M - 1) valPixel = currRow[n_col];
                    else valPixel = F[(rowIdx + 1) * N + n_col];
                }

                sum += valPixel * C[(u + 1) * K + (v + 1)];
            }
        }
        
        F[rowIdx * N + col] = sum;
    }
}

void readInput(int* &h_F, int* &h_C, int M, int N) {
    ifstream fin(INPUT_FILE);
    if (!fin) { 
        cerr << "Eroare: Nu pot deschide " << INPUT_FILE << endl; exit(1); 
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
    if (argc < 3) { cout << "Usage: ./program <M> <N>" << endl; return 1; }
    int M = stoi(argv[1]);
    int N = stoi(argv[2]);

    int *h_F = nullptr, *h_C = nullptr;
    readInput(h_F, h_C, M, N);

    int *d_F, *d_C;
    int *d_prevRow, *d_currRow;

    size_t sizeImage = M * N * sizeof(int);
    
    cudaCheckError(cudaMalloc(&d_F, sizeImage));
    cudaCheckError(cudaMalloc(&d_C, K * K * sizeof(int)));
    cudaCheckError(cudaMalloc(&d_prevRow, N * sizeof(int)));
    cudaCheckError(cudaMalloc(&d_currRow, N * sizeof(int)));

    cudaCheckError(cudaMemcpy(d_F, h_F, sizeImage, cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_C, h_C, K * K * sizeof(int), cudaMemcpyHostToDevice));

    cudaCheckError(cudaMemcpy(d_prevRow, d_F, N * sizeof(int), cudaMemcpyDeviceToDevice)); 
    cudaCheckError(cudaMemcpy(d_currRow, d_F, N * sizeof(int), cudaMemcpyDeviceToDevice));

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    auto start = chrono::high_resolution_clock::now();

    for (int i = 0; i < M; i++) {
        
        computeRowConvolution<<<blocksPerGrid, threadsPerBlock>>>(d_F, d_C, d_prevRow, d_currRow, i, M, N);
        cudaCheckError(cudaGetLastError()); // Verificare erori lansare

        if (i < M - 1) {
            cudaCheckError(cudaMemcpy(d_prevRow, d_currRow, N * sizeof(int), cudaMemcpyDeviceToDevice));
            
            cudaCheckError(cudaMemcpy(d_currRow, d_F + (i + 1) * N, N * sizeof(int), cudaMemcpyDeviceToDevice));
        }
    }

    cudaCheckError(cudaDeviceSynchronize());
    auto end = chrono::high_resolution_clock::now();
    double duration = chrono::duration<double, milli>(end - start).count();
    cout << "Timp executie: " << duration << " ms" << endl;

    cudaCheckError(cudaMemcpy(h_F, d_F, sizeImage, cudaMemcpyDeviceToHost));

    writeOutput(h_F, M, N);

    delete[] h_F; delete[] h_C;
    cudaFree(d_F); cudaFree(d_C); cudaFree(d_prevRow); cudaFree(d_currRow);

    return 0;
}
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <string>

using namespace std;

const string INPUT_FILE = "date.txt";
const string OUTPUT_FILE = "output_seq.txt"; 

const int K = 3;
const int RADIUS = K / 2;

// Kernelul de convolutie
int C_const[K * K];

void readInput(int* &h_F, int* &h_C, int M, int N) {
    ifstream fin(INPUT_FILE);
    if (!fin) {
        cerr << "Error: Cannot open " << INPUT_FILE << endl; exit(1);
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
    
    if (argc >= 3) {
        M = stoi(argv[1]);
        N = stoi(argv[2]);
    } else {
        cout << "Usage: ./sequential <M> <N>" << endl;
    }

    int *F = nullptr, *C_temp = nullptr;
    readInput(F, C_temp, M, N);
    
    // Initializare kernel
    for(int i=0; i<K*K; ++i) C_const[i] = C_temp[i];
    delete[] C_temp;

    // ALOCARE O(N) - Respectam constrangerea de spatiu
    // Avem nevoie de linia anterioara si curenta salvate, 
    // pentru a nu citi valori deja modificate.
    vector<int> prevRow(N);
    vector<int> currRow(N);
    vector<int> tempRow(N); // Buffer pentru rezultatul curent

    // Initializare buffer: Prima linie se comporta ca si cum ar avea padding sus
    for(int j=0; j<N; ++j) {
        prevRow[j] = F[j];
        currRow[j] = F[j];
    }

    cout << "Starting SEQUENTIAL convolution on " << M << "x" << N << "..." << endl;
    auto start = chrono::high_resolution_clock::now();

    for (int i = 0; i < M; ++i) {
        // Procesam fiecare coloana de pe linia i
        for (int col = 0; col < N; ++col) {
            int sum = 0;

            for (int u = -RADIUS; u <= RADIUS; u++) {
                for (int v = -RADIUS; v <= RADIUS; v++) {
                    
                    int n_col = col + v;
                    // Clamp to border (Padding stanga/dreapta)
                    if (n_col < 0) n_col = 0;
                    if (n_col >= N) n_col = N - 1;

                    int valPixel = 0;

                    if (u == -1) {
                        valPixel = prevRow[n_col]; // Luam din bufferul liniei anterioare
                    }
                    else if (u == 0) {
                        valPixel = currRow[n_col]; // Luam din bufferul liniei curente (nemodificat)
                    }
                    else { // u == 1
                        if (i == M - 1) {
                            valPixel = currRow[n_col]; // Ultima linie: padding jos
                        } else {
                            valPixel = F[(i + 1) * N + n_col]; // Citim din matricea F (inca nemodificata)
                        }
                    }

                    sum += valPixel * C_const[(u + RADIUS) * K + (v + RADIUS)];
                }
            }
            tempRow[col] = sum;
        }

        // Scriem rezultatul inapoi in matricea F (doar linia i)
        for(int j=0; j<N; ++j) {
            F[i * N + j] = tempRow[j];
        }

        // Pregatim bufferele pentru iteratia urmatoare (i+1)
        if (i < M - 1) {
            // Vechiul currRow devine prevRow
            prevRow = currRow;
            // Noul currRow il citim din matricea F (care contine valorile originale pt linia i+1)
            for(int j=0; j<N; ++j) {
                currRow[j] = F[(i + 1) * N + j];
            }
        }
    }

    auto end = chrono::high_resolution_clock::now();
    double duration = chrono::duration<double, milli>(end - start).count();
    
    // Acest format este necesar pentru ca benchmark.bat sa citeasca timpul
    cout << "Execution Time: " << duration << " ms" << endl;

    // Scrierea outputului e optionala la benchmark ca sa nu pierdem timp cu I/O,
    // dar e utila pentru verificare. Poti comenta linia daca vrei.
    // writeOutput(F, M, N);

    delete[] F;
    return 0;
}
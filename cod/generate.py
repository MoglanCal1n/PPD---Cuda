import random
import sys

def generate_data(M, N, K=3):
    filename = "date.txt"
    
    image_pixels = [random.randint(0, 255) for _ in range(M * N)]
    
    kernel_values = [random.randint(0, 5) for _ in range(K * K)]
    
    try:
        with open(filename, "w") as f:
            f.write(" ".join(map(str, image_pixels)))
            f.write("\n")
            
            f.write(" ".join(map(str, kernel_values)))
            f.write("\n")
            
        print(f"Succes! Fișierul {filename} a fost generat pentru M={M}, N={N}.")
    except Exception as e:
        print(f"Eroare la scrierea fișierului: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Utilizare: python gen_date.py <M> <N>")
    else:
        m_val = int(sys.argv[1])
        n_val = int(sys.argv[2])
        generate_data(m_val, n_val)
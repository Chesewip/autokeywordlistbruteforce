# Quagmire III Wordlist Bruteforcer (CLI)

It runs a lot of vigenere ciphers dawg. 
NOTE : This tool is optimized entirely for the GK cipher, it will not work with a cipher of another text length at the moment due to vectorization. So if you test something, it has to be 85 characters

## C++ command-line solver

The new `quag_bruteforce.cpp` binary is designed for fast, multi-threaded searches.

### Build

Compile with any C++17-compatible compiler. Example using `g++`:

Linux
```bash
g++ -std=c++17 -O3 -mavx2 -mfma -pthread -o quag_bruteforce_cpp quag_bruteforce.cpp

```

Windows

```
cl /std:c++17 /O2 /EHsc /arch:AVX2 /Fe:quag_bruteforce_cpp.exe quag_bruteforce.cpp

```


### CUDA build (optional)

To enable the GPU accelerated scoring path, compile the CUDA translation unit and link it with the main executable. Example commands on Linux:

```bash
nvcc -std=c++17 -O3 -c gpu_quag.cu -o gpu_quag.o
g++ -std=c++17 -O3 -mavx2 -mfma -pthread quag_bruteforce.cpp gpu_quag.o -lcudart -L"${CUDA_HOME:-/usr/local/cuda}/lib64" -o quag_bruteforce_cpp
```

On Windows with the Visual Studio toolchain:

```cmd
nvcc -std=c++17 -O3 -c gpu_quag.cu -o gpu_quag.obj
cl /std:c++17 /O2 /EHsc /arch:AVX2 quag_bruteforce.cpp gpu_quag.obj /link cudart.lib
```

At runtime pass `--use-cuda` to request GPU execution. The solver automatically falls back to the CPU path when no CUDA device is available.

To confirm parity between the CPU and GPU implementations, run a sample search with `--use-cuda` and then with `--no-cuda`, and compare the ranked results.

### Usage

Something like this. Ended up having a lot of word lists options that need to be set. 

Linux
```bash
./quag_bruteforce_cpp \
   --ciphertext-file "ciphertext.txt" \
   --wordlist "./Word Lists/370kwords.txt" \
   --two-letter-list "./Word Lists/2wordharsh.txt" \
   --alphabet-wordlist "./Word Lists/370kwords.txt" \
   --spacing-guide-file "spacingguideline.txt"  \
   --spacing-wordlist "./Word Lists/370kwords.txt" \
   --threads 48 --max-results 50 --include-autokey --preview-length 150
```

Windows
```
quag_bruteforce_cpp.exe --ciphertext-file "ciphertext.txt" --wordlist "Word Lists\370kwords.txt" --two-letter-list "Word Lists\2wordharsh.txt" --alphabet-wordlist "Word Lists\370kwords.txt" --spacing-guide-file "spacingguideline.txt"  --spacing-wordlist "Word Lists\370kwords.txt" --threads 20 --max-results 50 --include-autokey --preview-length 150

```

Key flags:

| Flag | Purpose |
| ---- | ------- |
| `--ciphertext` / `--ciphertext-file` | Provide the ciphertext inline or via file. |
| `--wordlist-inline` / `--wordlist` | Poorly worded, but this is the cipher key wordlist |
| `--two-letter-inline` / `--two-letter-list` | This list is for filtering the results quickly without having to fully decode the cipher  |
| `--alphabet-wordlist` | This list is used to create the alphabet keys|
| `--spacing-guide-file` | This file is for the spacing guide lines of the cipher. Used later when we rank real words in a result |
| `--spacing-wordlist` | This file is used for deciding if output words are real words |
| `--threads` | Number of worker threads (default: hardware concurrency). |
| `--include-autokey` | Enables searching autokey as well. Drastically adds compute time |
| `--max-results` | Maximum ranked candidates to keep (default: 50). |
| `--preview-length` | Number of plaintext characters to display for each candidate (default: 80). |
| `--progress-interval` | Seconds between progress updates (set `--quiet` to suppress). |
| `--no-keyword-front` | Skip key prefix on normal alphabet |
| `--no-reversed-keyword-front` |  Skip reversed key prefix variant |
| `--no-keyword-back` | Skip key suffix on normal alphabet |
| `--no-keyword-front-reversed` |  Skip key prefix on reversed alphabet |
| `--no-keyword-back-reversed` |  Skip key suffix on reversed alphabet |


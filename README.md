# Quagmire III Wordlist Bruteforcer (CLI)

This repository packages a terminal version of the Quagmire III wordlist solver that originally shipped as a browser tool (`quagmire_3_wordlist_solver (1).html`).
Both a Python script (`quag_bruteforce.py`) and a standalone C++ implementation (`quag_bruteforce.cpp`) reproduce the keyed alphabet enumeration,
cipher family decryptors (Vigenère, Beaufort, variant Beaufort, and their autokey variants), and the scoring heuristics used by the HTML worker.
Use them to explore candidate plaintexts directly from your shell or deploy them on bare-bones machines.

## C++ command-line solver

The new `quag_bruteforce.cpp` binary is designed for fast, multi-threaded searches on systems where Python is undesirable or unavailable.

### Build

Compile with any C++17-compatible compiler. Example using `g++`:

```bash
g++ -std=c++17 -O3 -pthread -o quag_bruteforce_cpp quag_bruteforce.cpp
```

### Usage

```bash
./quag_bruteforce_cpp \
    --ciphertext RIJVSUYVJN \
    --wordlist-inline $'KEY\nLEMON\nALPHA' \
    --two-letter-inline HE \
    --threads 4 \
    --max-results 25
```

Key flags:

| Flag | Purpose |
| ---- | ------- |
| `--ciphertext` / `--ciphertext-file` | Provide the ciphertext inline or via file. |
| `--wordlist-inline` / `--wordlist` | Main wordlist (required). |
| `--two-letter-inline` / `--two-letter-list` | Optional two-letter filter list. |
| `--threads` | Number of worker threads (default: hardware concurrency). |
| `--include-autokey` | Enable autokey variants in addition to the repeating-key modes. |
| `--max-results` | Maximum ranked candidates to keep (default: 50). |
| `--preview-length` | Number of plaintext characters to display for each candidate (default: 80). |
| `--progress-interval` | Seconds between progress updates (set `--quiet` to suppress). |

The tool mirrors the browser worker’s behaviour: it enumerates keyed alphabets in both forward and reversed forms,
tests the Vigenère, Beaufort, and variant Beaufort families, applies optional structural filters (two-letter start and 4-letter second word),
and ranks candidates using an IoC/χ² blend. During the search a live dashboard continuously clears and redraws on stdout,
showing elapsed time, work counters, and the current top 50 plaintext candidates (including cipher family, autokey usage, IoC, χ²,
and plaintext previews). When the run completes you also receive a final summary followed by the same ranked table for easy capture.

## Requirements

* Python 3.8 or newer
* A wordlist of candidate keys (one per line, uppercase recommended but not required)

The Python script has no third-party dependencies.

## Getting Started

1. Clone or download this repository.
2. (Optional) Make the script executable:
   ```bash
   chmod +x quag_bruteforce.py
   ```
3. Run the solver with Python:
   ```bash
   python quag_bruteforce.py --ciphertext "RIJVSUYVJN" \
       --wordlist-inline $'KEY\nABCDEFGHIJKLMNOPQRSTUVWXYZ'
   ```

The example above decrypts a short ciphertext using two inline key candidates.

## Providing Inputs

You may supply the ciphertext and wordlists either inline or via files. The script cleans each entry to uppercase A–Z letters.

| Input | Inline flag | File flag | Notes |
| ----- | ----------- | --------- | ----- |
| Ciphertext | `--ciphertext` | `--ciphertext-file` | Provide exactly one of these. |
| Key wordlist | `--wordlist-inline` | `--wordlist` | Required (at least one key). |
| Two-letter list | `--two-letter-inline` | `--two-letter-list` | Optional filter; only candidates whose plaintext starts with any pair in this list are kept. |

When both inline and file variants are provided for the same input, the inline value takes precedence.

## Core Options

```
usage: quag_bruteforce.py [options]

optional arguments:
  --ciphertext TEXT          Ciphertext as a string.
  --ciphertext-file PATH     Read ciphertext from a file (used if --ciphertext is omitted).
  --wordlist PATH            Path to the main wordlist (one word per line).
  --wordlist-inline TEXT     Inline wordlist string (lines separated by \n).
  --two-letter-list PATH     Path to an optional 2-letter list (one per line).
  --two-letter-inline TEXT   Inline 2-letter list string.
  --max-results N            Maximum number of candidates to keep (default: 50).
  --update-interval SEC      Emit progress updates every SEC seconds (default: 0.75).
  --start-key-index N        Start index within the wordlist (inclusive, default: 0).
  --end-key-index N          End index within the wordlist (exclusive).
  --preview-length N         Number of plaintext characters to show in the preview (default: 120).
  --no-autokey               Disable autokey-family decryptions.
  --workers N                Number of worker processes to spawn (default: 1).
```

### Progress and Output

* While the search runs the solver continuously refreshes a progress dashboard on stderr. It includes a progress bar, live statistics, and the current top 50 candidates (or fewer if `--max-results` is lower).
* Adjust `--update-interval` to control how frequently the dashboard refreshes. Setting it to `0` forces an update after every chunk of work.
* When multiprocessing is enabled the dashboard shows how many keys are currently "in flight" across workers (for example `(+16 in-flight)`). The scheduler automatically breaks the wordlist into bite-sized chunks so progress updates continue to arrive even when keyed alphabet enumeration is extremely large.
* Once the search completes, a final summary is printed to stderr and the same ranked candidate table is emitted to stdout so you can pipe it to other tools.
  Columns include the cipher mode, key word, alphabet word (and whether it was reversed), the first two plaintext letters, and a preview of the plaintext.

## Worked Examples

### Search a small inline wordlist

```bash
python quag_bruteforce.py \
    --ciphertext RIJVSUYVJN \
    --wordlist-inline $'KEY\nLEMON\nALPHA' \
    --two-letter-inline HE \
    --max-results 5
```

### Use wordlists stored on disk

```bash
python quag_bruteforce.py \
    --ciphertext-file ciphertext.txt \
    --wordlist "370-thousand-wordlist_english_unknown_unknown.txt" \
    --two-letter-list "2 letter word list.txt" \
    --start-key-index 0 \
    --end-key-index 10000 \
    --update-interval 5
```

### Skip autokey modes

```bash
python quag_bruteforce.py --ciphertext RIJVSUYVJN --wordlist-inline KEY --no-autokey
```

Disabling autokey can speed up searches when you know the cipher was strictly Quagmire/Beaufort.

## Debugging without the CLI

If you prefer to step through the solver inside an IDE, use the companion script
`vscode_debug.py`.  It imports the same `run_search` function as the CLI but
pulls its inputs from a small configuration dictionary instead of parsing
arguments.

1. Open `vscode_debug.py` and adjust the paths or inline strings in
   `DEBUG_CONFIG` to match your ciphertext and word lists.  By default it reads
   the bundled 370k wordlist and evaluates the first 200 entries.
2. Launch the script under the Visual Studio Code debugger (or run
   `python vscode_debug.py` from a terminal).  Set breakpoints anywhere inside
   `quag_bruteforce.py`; they will trigger exactly as they do for the CLI.

The script automatically enables `multiprocessing.freeze_support()` so spawned
workers behave on Windows the same way they do when the CLI runs from the
command line.

## Tips

* The solver filters candidates whose plaintext preview starts with any entry from the optional two-letter list.
  If you do not provide a list, all candidates are considered.
* Use `--start-key-index` and `--end-key-index` to split large wordlists across multiple runs or machines.
* Increase `--preview-length` to inspect more plaintext without re-running the solver.
* Bump `--workers` to take advantage of multiple CPU cores (values above your physical core count may slow things down).

## Troubleshooting

* **"Error loading ..."** – ensure exactly one of the inline or file flags is provided for each input.
* **No results** – try removing the two-letter filter, expanding the wordlist, or enabling autokey modes.
* **Slow progress** – increase the update interval to reduce console noise, bump `--workers` to parallelise the search, or narrow the key index range.  On Windows, the solver now uses the `spawn` multiprocessing context and preloads shared data so multi-worker runs keep updating the dashboard instead of stalling at zero combos.

For deeper inspection or to integrate the solver into your own tooling, read through `quag_bruteforce.py`.

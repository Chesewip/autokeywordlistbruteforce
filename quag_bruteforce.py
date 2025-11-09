#!/usr/bin/env python3
"""Command-line Quagmire III wordlist solver.

This script replicates the search behaviour of the browser-based
"quagmire_3_wordlist_solver" tool.  It enumerates candidate Vigenère keys
and keyed alphabets derived from a supplied wordlist, scores the resulting
plaintexts, and prints the best-scoring matches.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import itertools
import math
import sys
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

BASE_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
ENGLISH_FREQ = [
    0.08167,
    0.01492,
    0.02782,
    0.04253,
    0.12702,
    0.02228,
    0.02015,
    0.06094,
    0.06966,
    0.00153,
    0.00772,
    0.04025,
    0.02406,
    0.06749,
    0.07507,
    0.01929,
    0.00095,
    0.05987,
    0.06327,
    0.09056,
    0.02758,
    0.00978,
    0.02360,
    0.00150,
    0.01974,
    0.00074,
]
IOC_TARGET = 0.066
TARGET_COMBOS_PER_CHUNK = 250_000


def clean_text_letters(text: str) -> str:
    """Uppercase the text and keep only characters in the base alphabet."""

    return "".join(ch for ch in text.upper() if ch in BASE_ALPHABET)


def parse_wordlist(text: str) -> List[str]:
    words: List[str] = []
    for line in text.splitlines():
        cleaned = clean_text_letters(line)
        if cleaned:
            words.append(cleaned)
    return words


def parse_two_letter_list(text: str) -> Set[str]:
    pairs: Set[str] = set()
    for line in text.splitlines():
        cleaned = clean_text_letters(line)
        if len(cleaned) == 2:
            pairs.add(cleaned)
    return pairs


def build_four_letter_set_from_words(words: Sequence[str]) -> Set[str]:
    return {w for w in words if len(w) == 4}


def build_keyed_alphabet(word: str) -> str:
    seen: Set[str] = set()
    head_chars: List[str] = []
    for ch in word:
        if ch not in seen:
            seen.add(ch)
            head_chars.append(ch)
    for ch in BASE_ALPHABET:
        if ch not in seen:
            head_chars.append(ch)
    return "".join(head_chars)


def build_alphabet_candidates(words: Sequence[str]) -> List[Dict[str, object]]:
    candidates: Dict[str, Tuple[str, bool]] = {}
    for word in words:
        forward = build_keyed_alphabet(word)
        if forward not in candidates:
            candidates[forward] = (word, False)
        reversed_word = word[::-1]
        reverse = build_keyed_alphabet(reversed_word)
        if reverse not in candidates:
            candidates[reverse] = (word, True)
    return [
        {
            "alphabet": alphabet,
            "base_word": base_word,
            "is_reversed": is_reversed,
        }
        for alphabet, (base_word, is_reversed) in candidates.items()
    ]


def decrypt_vigenere(cipher: str, key: str, alphabet: str) -> str:
    if not cipher or not key:
        return ""
    result_chars: List[str] = []
    key_len = len(key)
    for idx, c in enumerate(cipher):
        c_idx = alphabet.find(c)
        k_idx = alphabet.find(key[idx % key_len])
        if c_idx == -1 or k_idx == -1:
            result_chars.append(c)
            continue
        p_idx = (c_idx - k_idx) % 26
        result_chars.append(alphabet[p_idx])
    return "".join(result_chars)


def decrypt_beaufort(cipher: str, key: str, alphabet: str) -> str:
    if not cipher or not key:
        return ""
    result_chars: List[str] = []
    key_len = len(key)
    for idx, c in enumerate(cipher):
        c_idx = alphabet.find(c)
        k_idx = alphabet.find(key[idx % key_len])
        if c_idx == -1 or k_idx == -1:
            result_chars.append(c)
            continue
        p_idx = (k_idx - c_idx) % 26
        result_chars.append(alphabet[p_idx])
    return "".join(result_chars)


def decrypt_variant_beaufort(cipher: str, key: str, alphabet: str) -> str:
    if not cipher or not key:
        return ""
    result_chars: List[str] = []
    key_len = len(key)
    for idx, c in enumerate(cipher):
        c_idx = alphabet.find(c)
        k_idx = alphabet.find(key[idx % key_len])
        if c_idx == -1 or k_idx == -1:
            result_chars.append(c)
            continue
        p_idx = (c_idx + k_idx) % 26
        result_chars.append(alphabet[p_idx])
    return "".join(result_chars)


def decrypt_autokey_vigenere(cipher: str, key: str, alphabet: str) -> str:
    if not cipher or not key:
        return ""
    result_chars: List[str] = []
    key_len = len(key)
    for idx, c in enumerate(cipher):
        c_idx = alphabet.find(c)
        if idx < key_len:
            k_char = key[idx]
        else:
            k_char = result_chars[idx - key_len]
        k_idx = alphabet.find(k_char)
        if c_idx == -1 or k_idx == -1:
            result_chars.append(c)
            continue
        p_idx = (c_idx - k_idx) % 26
        result_chars.append(alphabet[p_idx])
    return "".join(result_chars)


def decrypt_autokey_beaufort(cipher: str, key: str, alphabet: str) -> str:
    if not cipher or not key:
        return ""
    result_chars: List[str] = []
    key_len = len(key)
    for idx, c in enumerate(cipher):
        c_idx = alphabet.find(c)
        if idx < key_len:
            k_char = key[idx]
        else:
            k_char = result_chars[idx - key_len]
        k_idx = alphabet.find(k_char)
        if c_idx == -1 or k_idx == -1:
            result_chars.append(c)
            continue
        p_idx = (k_idx - c_idx) % 26
        result_chars.append(alphabet[p_idx])
    return "".join(result_chars)


def decrypt_autokey_variant_beaufort(cipher: str, key: str, alphabet: str) -> str:
    if not cipher or not key:
        return ""
    result_chars: List[str] = []
    key_len = len(key)
    for idx, c in enumerate(cipher):
        c_idx = alphabet.find(c)
        if idx < key_len:
            k_char = key[idx]
        else:
            k_char = result_chars[idx - key_len]
        k_idx = alphabet.find(k_char)
        if c_idx == -1 or k_idx == -1:
            result_chars.append(c)
            continue
        p_idx = (c_idx + k_idx) % 26
        result_chars.append(alphabet[p_idx])
    return "".join(result_chars)


def index_of_coincidence(text: str) -> float:
    n = len(text)
    if n <= 1:
        return 0.0
    counts = [0] * 26
    for ch in text:
        code = ord(ch) - 65
        if 0 <= code < 26:
            counts[code] += 1
    numerator = sum(c * (c - 1) for c in counts)
    return numerator / (n * (n - 1))


def chi_square(text: str) -> float:
    n = len(text)
    if n == 0:
        return math.inf
    counts = [0] * 26
    for ch in text:
        code = ord(ch) - 65
        if 0 <= code < 26:
            counts[code] += 1
    chi = 0.0
    for idx, expected_freq in enumerate(ENGLISH_FREQ):
        expected = n * expected_freq
        if expected <= 0:
            continue
        diff = counts[idx] - expected
        chi += (diff * diff) / expected
    return chi


def score_plaintext(text: str) -> Dict[str, float]:
    ioc = index_of_coincidence(text)
    chi = chi_square(text)
    ioc_delta = abs(ioc - IOC_TARGET)
    ioc_score = max(0.0, 1.0 - ioc_delta / 0.02)
    chi_clamped = min(chi, 400.0)
    chi_score = max(0.0, 1.0 - chi_clamped / 400.0)
    quality_factor = 0.1 + 0.9 * chi_score
    score = ioc_score * quality_factor
    return {"score": score, "ioc": ioc, "chi": chi}


def maintain_top_results(results: List[Dict[str, object]], candidate: Dict[str, object], max_results: int) -> None:
    if max_results <= 0:
        return
    if len(results) < max_results:
        results.append(candidate)
        return
    worst_index = min(range(len(results)), key=lambda idx: results[idx]["score"])
    if candidate["score"] > results[worst_index]["score"]:
        results[worst_index] = candidate


def load_text_arg(value: Optional[str], file_value: Optional[str], description: str) -> str:
    if value and file_value:
        raise ValueError(f"Cannot specify both inline {description} and {description} file")
    if value:
        return value
    if file_value:
        return Path(file_value).read_text(encoding="utf-8")
    raise ValueError(f"Missing {description}; provide either --{description} or --{description}-file")


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    remainder = seconds - minutes * 60
    return f"{minutes}m {remainder:.1f}s"


class LiveRenderer:
    """Continuously refreshes progress and top candidate details."""

    def __init__(self, max_results: int, stream=None) -> None:
        self.max_results = max_results
        self.stream = stream or sys.stderr
        self._active = False
        self._supports_clear = hasattr(self.stream, "isatty") and self.stream.isatty()
        self._bar_width = 40

    def _build_header(self) -> str:
        return (
            f"{'#':>3}  {'Score':>7}  {'IoC':>8}  {'Chi^2':>8}  {'Mode':<20}  "
            f"{'Key':<15}  {'Alphabet word':<15}  {'Rev':<3}  {'First2':<6}  Preview"
        )

    def render(self, stats: Dict[str, float | int], results: Sequence[Dict[str, object]]) -> None:
        self._active = True
        total_combos = int(stats.get("total_combos", 0) or 0)
        combos_tried = int(stats.get("combos_tried", 0) or 0)
        keys_processed = int(stats.get("keys_processed", 0) or 0)
        keys_considered = int(stats.get("keys_considered", 0) or 0)
        autokey_attempts = int(stats.get("autokey_attempts", 0) or 0)
        worker_count = int(stats.get("worker_count", 1) or 1)
        elapsed_seconds = float(stats.get("elapsed_seconds", 0.0) or 0.0)

        progress_ratio = 0.0
        if total_combos > 0:
            progress_ratio = min(1.0, combos_tried / total_combos)
        filled = int(round(progress_ratio * self._bar_width))
        progress_bar = "[" + "#" * filled + "-" * (self._bar_width - filled) + "]"
        best_score = max((cand["score"] for cand in results), default=0.0)

        lines = []
        keys_inflight = int(stats.get("keys_inflight", 0))
        keys_line = (
            "Workers: {workers}  Keys: {processed}/{total_keys}  Combos: {combos}/{total_combos}  "
            "Autokey attempts: {autokey}".format(
                workers=worker_count,
                processed=keys_processed,
                total_keys=keys_considered,
                combos=combos_tried,
                total_combos=total_combos,
                autokey=autokey_attempts,
            )
        )
        if keys_inflight:
            keys_line += f" (+{keys_inflight} in-flight)"
        lines.append(keys_line)
        lines.append(
            f"{progress_bar}  {progress_ratio * 100:5.1f}%  elapsed {format_duration(elapsed_seconds)}"
        )
        lines.append(f"Best score: {best_score:0.3f}")
        lines.append("")

        display_limit = min(50, self.max_results)
        header = self._build_header()
        lines.append(f"Top candidates (showing up to {display_limit}):")
        lines.append(header)
        lines.append("-" * len(header))

        sorted_results = sorted(results, key=lambda c: c["score"], reverse=True)
        for idx, candidate in enumerate(sorted_results[:display_limit], start=1):
            rev_flag = "Y" if candidate["alphabet_reversed"] else "N"
            lines.append(
                f"{idx:>3}  "
                f"{candidate['score']:7.3f}  "
                f"{candidate['ioc']:8.4f}  "
                f"{candidate['chi']:8.2f}  "
                f"{str(candidate['mode']):<20}  "
                f"{str(candidate['key']):<15}  "
                f"{str(candidate['alphabet_word']):<15}  "
                f"{rev_flag:<3}  "
                f"{candidate['first2']:<6}  "
                f"{candidate['plaintext_preview']}"
            )

        output = "\n".join(lines)
        if self._supports_clear:
            self.stream.write("\033[2J\033[H")
        self.stream.write(output + "\n")
        self.stream.flush()

    def finish(self, stats: Dict[str, float | int], results: Sequence[Dict[str, object]]) -> None:
        self.render(stats, results)
        if self._supports_clear:
            self.stream.write("\n")
            self.stream.flush()
        self._active = False


def evaluate_key_word(
    key_word: str,
    cipher: str,
    alph_candidates: Sequence[Dict[str, object]],
    families: Sequence[
        Tuple[
            str,
            Callable[[str, str, str], str],
            Callable[[str, str, str], str],
        ]
    ],
    two_letter_set: Set[str],
    four_letter_set: Set[str],
    preview_length: int,
    include_autokey: bool,
    max_results: int,
    have_first2_filter: bool,
    have_second4_filter: bool,
) -> Tuple[List[Dict[str, object]], int, int]:
    """Evaluate all alphabet/mode combinations for a single key word."""

    local_results: List[Dict[str, object]] = []
    combos_tried = 0
    autokey_attempts = 0

    for alph_info in alph_candidates:
        alphabet = str(alph_info["alphabet"])
        base_word = str(alph_info["base_word"])
        is_reversed = bool(alph_info["is_reversed"])

        for mode_name, decrypt_fn, autokey_fn in families:
            plaintext_std = decrypt_fn(cipher, key_word, alphabet)
            combos_tried += 1

            if len(plaintext_std) < 2:
                continue

            first2 = plaintext_std[:2]
            second_word4 = plaintext_std[2:6] if len(plaintext_std) >= 6 else ""

            if have_first2_filter and first2 not in two_letter_set:
                continue
            if have_second4_filter and second_word4 not in four_letter_set:
                continue

            scored_std = score_plaintext(plaintext_std)
            candidate_std = {
                "score": scored_std["score"],
                "chi": scored_std["chi"],
                "ioc": scored_std["ioc"],
                "key": key_word,
                "mode": mode_name,
                "alphabet_word": base_word,
                "alphabet_reversed": is_reversed,
                "alphabet_string": alphabet,
                "first2": first2,
                "plaintext_preview": plaintext_std[:preview_length],
            }
            maintain_top_results(local_results, candidate_std, max_results)

            if include_autokey:
                plaintext_auto = autokey_fn(cipher, key_word, alphabet)
                autokey_attempts += 1

                if len(plaintext_auto) < 2:
                    continue

                first2_auto = plaintext_auto[:2]
                second_word4_auto = plaintext_auto[2:6] if len(plaintext_auto) >= 6 else ""

                passes = True
                if have_first2_filter and first2_auto not in two_letter_set:
                    passes = False
                if have_second4_filter and second_word4_auto not in four_letter_set:
                    passes = False

                if passes:
                    scored_auto = score_plaintext(plaintext_auto)
                    autokey_label = {
                        "Vig": "Vig autokey",
                        "Beaufort": "Beaufort autokey",
                        "Beaufort var": "Beaufort var autokey",
                    }[mode_name]
                    candidate_auto = {
                        "score": scored_auto["score"],
                        "chi": scored_auto["chi"],
                        "ioc": scored_auto["ioc"],
                        "key": key_word,
                        "mode": autokey_label,
                        "alphabet_word": base_word,
                        "alphabet_reversed": is_reversed,
                        "alphabet_string": alphabet,
                        "first2": first2_auto,
                        "plaintext_preview": plaintext_auto[:preview_length],
                    }
                    maintain_top_results(local_results, candidate_auto, max_results)

    return local_results, combos_tried, autokey_attempts


def _process_key_chunk(
    key_chunk: Sequence[str],
    cipher: str,
    alph_candidates: Sequence[Dict[str, object]],
    families: Sequence[
        Tuple[
            str,
            Callable[[str, str, str], str],
            Callable[[str, str, str], str],
        ]
    ],
    two_letter_set: Set[str],
    four_letter_set: Set[str],
    preview_length: int,
    include_autokey: bool,
    max_results: int,
    have_first2_filter: bool,
    have_second4_filter: bool,
) -> Tuple[List[Dict[str, object]], int, int, int]:
    """Worker helper that evaluates a chunk of keys."""

    chunk_results: List[Dict[str, object]] = []
    chunk_combos = 0
    chunk_autokey_attempts = 0

    for key_word in key_chunk:
        key_results, combos, autokey_attempts = evaluate_key_word(
            key_word=key_word,
            cipher=cipher,
            alph_candidates=alph_candidates,
            families=families,
            two_letter_set=two_letter_set,
            four_letter_set=four_letter_set,
            preview_length=preview_length,
            include_autokey=include_autokey,
            max_results=max_results,
            have_first2_filter=have_first2_filter,
            have_second4_filter=have_second4_filter,
        )
        chunk_combos += combos
        chunk_autokey_attempts += autokey_attempts
        for candidate in key_results:
            maintain_top_results(chunk_results, candidate, max_results)

    return chunk_results, chunk_combos, chunk_autokey_attempts, len(key_chunk)


def _iter_chunks(seq: Sequence[str], chunk_size: int) -> Iterable[Sequence[str]]:
    iterator = iter(seq)
    while True:
        chunk = list(itertools.islice(iterator, chunk_size))
        if not chunk:
            break
        yield chunk


def run_search(
    ciphertext: str,
    words: Sequence[str],
    two_letter_set: Set[str],
    max_results: int,
    update_interval: float,
    start_index: int,
    end_index: int,
    preview_length: int,
    include_autokey: bool,
    workers: int,
) -> Tuple[List[Dict[str, object]], Dict[str, float | int]]:
    cipher = clean_text_letters(ciphertext)
    if not cipher:
        raise ValueError("Ciphertext has no A-Z letters after cleaning.")

    if not words:
        raise ValueError("Wordlist is empty after cleaning.")

    four_letter_set = build_four_letter_set_from_words(words)
    alph_candidates = build_alphabet_candidates(words)
    if not alph_candidates:
        raise ValueError("No unique keyed alphabets could be derived from the wordlist.")

    total_keys = len(words)
    start_index = max(0, start_index)
    end_index = max(start_index, min(end_index, total_keys))
    if start_index >= end_index:
        raise ValueError("Invalid key range: start index must be less than end index.")

    key_subset = words[start_index:end_index]

    families = (
        ("Vig", decrypt_vigenere, decrypt_autokey_vigenere),
        ("Beaufort", decrypt_beaufort, decrypt_autokey_beaufort),
        ("Beaufort var", decrypt_variant_beaufort, decrypt_autokey_variant_beaufort),
    )

    total_combos = len(key_subset) * len(alph_candidates) * len(families)

    have_first2_filter = bool(two_letter_set)
    have_second4_filter = bool(four_letter_set)

    results: List[Dict[str, object]] = []
    worker_count = max(1, workers)

    stats: Dict[str, float | int] = {
        "total_keys": total_keys,
        "keys_considered": len(key_subset),
        "total_alphabets": len(alph_candidates),
        "total_combos": total_combos,
        "combos_tried": 0,
        "autokey_attempts": 0,
        "worker_count": worker_count,
        "keys_processed": 0,
        "keys_inflight": 0,
    }

    last_update = time.perf_counter()
    start_time = last_update
    renderer = LiveRenderer(max_results=max_results)

    if worker_count == 1:
        for key_word in key_subset:
            key_results, combos, autokey_attempts = evaluate_key_word(
                key_word=key_word,
                cipher=cipher,
                alph_candidates=alph_candidates,
                families=families,
                two_letter_set=two_letter_set,
                four_letter_set=four_letter_set,
                preview_length=preview_length,
                include_autokey=include_autokey,
                max_results=max_results,
                have_first2_filter=have_first2_filter,
                have_second4_filter=have_second4_filter,
            )
            stats["combos_tried"] += combos
            stats["autokey_attempts"] += autokey_attempts
            stats["keys_processed"] += 1
            for candidate in key_results:
                maintain_top_results(results, candidate, max_results)

            now = time.perf_counter()
            if update_interval == 0 or now - last_update >= update_interval:
                elapsed = now - start_time
                stats["elapsed_seconds"] = elapsed
                renderer.render(stats, results)
                last_update = now
    else:
        combos_per_key = len(alph_candidates) * len(families)
        approx_work_per_key = combos_per_key * (2 if include_autokey else 1)
        chunk_size_by_work = max(
            1,
            TARGET_COMBOS_PER_CHUNK
            // max(1, approx_work_per_key),
        )
        chunk_size_by_keys = max(1, math.ceil(len(key_subset) / (worker_count * 8)))
        chunk_size = max(1, min(chunk_size_by_work, chunk_size_by_keys))

        chunk_iter = _iter_chunks(key_subset, chunk_size)

        with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as executor:
            pending: Set[concurrent.futures.Future] = set()

            def submit_next_chunk() -> bool:
                try:
                    next_chunk = next(chunk_iter)
                except StopIteration:
                    return False
                future = executor.submit(
                    _process_key_chunk,
                    next_chunk,
                    cipher,
                    alph_candidates,
                    families,
                    two_letter_set,
                    four_letter_set,
                    preview_length,
                    include_autokey,
                    max_results,
                    have_first2_filter,
                    have_second4_filter,
                )
                pending.add(future)
                stats["keys_inflight"] += len(next_chunk)
                return True

            for _ in range(worker_count):
                if not submit_next_chunk():
                    break

            while pending:
                timeout = update_interval if update_interval > 0 else None
                done, not_done = concurrent.futures.wait(
                    pending,
                    timeout=timeout,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )

                if not done:
                    pending = not_done
                    now = time.perf_counter()
                    if update_interval > 0 and now - last_update >= update_interval:
                        elapsed = now - start_time
                        stats["elapsed_seconds"] = elapsed
                        renderer.render(stats, results)
                        last_update = now
                    continue

                pending = not_done
                for future in done:
                    chunk_results, chunk_combos, chunk_autokey_attempts, chunk_keys = future.result()
                    stats["combos_tried"] += chunk_combos
                    stats["autokey_attempts"] += chunk_autokey_attempts
                    stats["keys_inflight"] -= chunk_keys
                    stats["keys_processed"] += chunk_keys
                    for candidate in chunk_results:
                        maintain_top_results(results, candidate, max_results)

                while len(pending) < worker_count and submit_next_chunk():
                    pass

                now = time.perf_counter()
                if update_interval == 0 or now - last_update >= update_interval:
                    elapsed = now - start_time
                    stats["elapsed_seconds"] = elapsed
                    renderer.render(stats, results)
                    last_update = now

    stats["elapsed_seconds"] = time.perf_counter() - start_time
    renderer.finish(stats, results)
    return results, stats


def print_results(results: Sequence[Dict[str, object]]) -> None:
    if not results:
        print("No candidates met the filtering criteria.")
        return

    header = (
        f"{'#':>3}  {'Score':>7}  {'IoC':>8}  {'Chi^2':>8}  {'Mode':<20}  "
        f"{'Key':<15}  {'Alphabet word':<15}  {'Rev':<3}  {'First2':<6}  Preview"
    )
    print(header)
    print("-" * len(header))
    for idx, candidate in enumerate(sorted(results, key=lambda c: c["score"], reverse=True), start=1):
        rev_flag = "Y" if candidate["alphabet_reversed"] else "N"
        print(
            f"{idx:>3}  "
            f"{candidate['score']:7.3f}  "
            f"{candidate['ioc']:8.4f}  "
            f"{candidate['chi']:8.2f}  "
            f"{str(candidate['mode']):<20}  "
            f"{str(candidate['key']):<15}  "
            f"{str(candidate['alphabet_word']):<15}  "
            f"{rev_flag:<3}  "
            f"{candidate['first2']:<6}  "
            f"{candidate['plaintext_preview']}"
        )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quagmire III wordlist bruteforcer")
    parser.add_argument("--ciphertext", help="Ciphertext as a string")
    parser.add_argument("--ciphertext-file", help="Path to a file containing the ciphertext")
    parser.add_argument("--wordlist", help="Path to the main wordlist (one word per line)")
    parser.add_argument("--wordlist-inline", help="Inline wordlist string")
    parser.add_argument("--two-letter-list", help="Path to an optional 2-letter list (one per line)")
    parser.add_argument("--two-letter-inline", help="Inline 2-letter list string")
    parser.add_argument("--max-results", type=int, default=50, help="Maximum number of top candidates to keep (default: 50)")
    parser.add_argument(
        "--update-interval",
        type=float,
        default=0.75,
        help="Progress update interval in seconds (default: 0.75)",
    )
    parser.add_argument("--start-key-index", type=int, default=0, help="Start index within the wordlist (inclusive)")
    parser.add_argument("--end-key-index", type=int, default=None, help="End index within the wordlist (exclusive)")
    parser.add_argument("--preview-length", type=int, default=120, help="Plaintext preview length to display (default: 120)")
    parser.add_argument("--no-autokey", action="store_true", help="Disable autokey family checks")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes to use (default: 1)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    try:
        ciphertext = load_text_arg(args.ciphertext, args.ciphertext_file, "ciphertext")
    except Exception as exc:  # pragma: no cover - user input validation
        print(f"Error loading ciphertext: {exc}", file=sys.stderr)
        return 1

    try:
        wordlist_text = load_text_arg(args.wordlist_inline, args.wordlist, "wordlist")
    except Exception as exc:  # pragma: no cover - user input validation
        print(f"Error loading wordlist: {exc}", file=sys.stderr)
        return 1

    two_letter_text = ""
    try:
        if args.two_letter_list or args.two_letter_inline:
            two_letter_text = load_text_arg(args.two_letter_inline, args.two_letter_list, "two-letter-list")
    except Exception as exc:  # pragma: no cover - user input validation
        print(f"Error loading two-letter list: {exc}", file=sys.stderr)
        return 1

    ciphertext = ciphertext.strip()
    words = parse_wordlist(wordlist_text)
    two_letter_set = parse_two_letter_list(two_letter_text)

    end_index = args.end_key_index if args.end_key_index is not None else len(words)

    try:
        results, stats = run_search(
            ciphertext=ciphertext,
            words=words,
            two_letter_set=two_letter_set,
            max_results=max(0, args.max_results),
            update_interval=max(0.0, args.update_interval),
            start_index=args.start_key_index,
            end_index=end_index,
            preview_length=max(0, args.preview_length),
            include_autokey=not args.no_autokey,
            workers=max(1, args.workers),
        )
    except Exception as exc:  # pragma: no cover - runtime validation
        print(f"Search error: {exc}", file=sys.stderr)
        return 1

    elapsed = stats.get("elapsed_seconds", 0.0)
    print(
        f"Completed in {format_duration(elapsed)}. Keys considered: {stats['keys_considered']}/" f"{stats['total_keys']}, "
        f"alphabets: {stats['total_alphabets']}, combos tried: {stats['combos_tried']}/" f"{stats['total_combos']}.",
        file=sys.stderr,
    )
    if not args.no_autokey:
        print(
            f"Autokey decryptions evaluated: {stats['autokey_attempts']}",
            file=sys.stderr,
        )

    print_results(results)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())

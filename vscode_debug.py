"""Helper harness for debugging :mod:`quag_bruteforce` in an IDE.

Edit ``DEBUG_CONFIG`` below to point at your ciphertext, wordlist and
optional helper lists.  Running this module executes ``run_search`` with the
same engine the CLI uses but avoids argument parsing, which makes it easier to
set breakpoints in Visual Studio Code or other debuggers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from quag_bruteforce import (
    parse_two_letter_list,
    parse_wordlist,
    print_results,
    run_search,
)


def _load_optional(path: Optional[str]) -> str:
    if not path:
        return ""
    return Path(path).read_text(encoding="utf-8")


DEBUG_CONFIG = {
    "ciphertext": "RIJVSUYVJN",
    "ciphertext_path": None,
    "wordlist_inline": "KEY\nABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "wordlist_path": "370-thousand-wordlist_english_unknown_unknown.txt",
    "two_letter_inline": "HE",
    "two_letter_path": "2 letter word list.txt",
    "max_results": 50,
    "update_interval": 0.5,
    "start_key_index": 0,
    "end_key_index": 200,
    "preview_length": 120,
    "include_autokey": True,
    "workers": 2,
}


def main() -> None:
    ciphertext = DEBUG_CONFIG.get("ciphertext") or ""
    if not ciphertext:
        ciphertext = _load_optional(DEBUG_CONFIG.get("ciphertext_path"))
    if not ciphertext:
        raise ValueError("Populate DEBUG_CONFIG['ciphertext'] or 'ciphertext_path'.")

    wordlist_text = DEBUG_CONFIG.get("wordlist_inline") or ""
    if not wordlist_text:
        wordlist_text = _load_optional(DEBUG_CONFIG.get("wordlist_path"))
    if not wordlist_text:
        raise ValueError("Populate DEBUG_CONFIG with a wordlist inline string or path.")

    two_letter_text = DEBUG_CONFIG.get("two_letter_inline") or ""
    if not two_letter_text:
        two_letter_text = _load_optional(DEBUG_CONFIG.get("two_letter_path"))

    words = parse_wordlist(wordlist_text)
    two_letter_set = parse_two_letter_list(two_letter_text)

    results, stats = run_search(
        ciphertext=ciphertext.strip(),
        words=words,
        two_letter_set=two_letter_set,
        max_results=int(DEBUG_CONFIG.get("max_results", 50)),
        update_interval=float(DEBUG_CONFIG.get("update_interval", 0.5)),
        start_index=int(DEBUG_CONFIG.get("start_key_index", 0)),
        end_index=int(DEBUG_CONFIG.get("end_key_index", len(words))),
        preview_length=int(DEBUG_CONFIG.get("preview_length", 120)),
        include_autokey=bool(DEBUG_CONFIG.get("include_autokey", True)),
        workers=int(DEBUG_CONFIG.get("workers", 1)),
    )

    print(
        "Completed debug run in {elapsed:.2f}s (keys processed: {done}/{total}, "
        "combos tried: {combos}/{total_combos}).".format(
            elapsed=float(stats.get("elapsed_seconds", 0.0)),
            done=int(stats.get("keys_processed", 0)),
            total=int(stats.get("keys_considered", 0)),
            combos=int(stats.get("combos_tried", 0)),
            total_combos=int(stats.get("total_combos", 0)),
        )
    )
    print_results(results)


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    main()

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cctype>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace {

constexpr std::array<double, 26> kEnglishFreq = {
    0.08167, 0.01492, 0.02782, 0.04253, 0.12702, 0.02228, 0.02015,
    0.06094, 0.06966, 0.00153, 0.00772, 0.04025, 0.02406, 0.06749,
    0.07507, 0.01929, 0.00095, 0.05987, 0.06327, 0.09056, 0.02758,
    0.00978, 0.02360, 0.00150, 0.01974, 0.00074};

constexpr double kIocTarget = 0.066;

enum class Mode {
  kVigenere,
  kBeaufort,
  kVariantBeaufort,
};

struct AlphabetCandidate {
  std::string alphabet;
  std::array<int, 26> index_map{};
  std::string base_word;
  bool keyword_reversed = false;
  bool alphabet_reversed = false;
  bool keyword_front = true;
};

#if defined(__AVX2__)
struct alignas(32) KeyBlock {
  std::array<std::uint8_t, 32> data{};
};
#endif

struct Candidate {
  double score = 0.0;
  double ioc = 0.0;
  double chi = 0.0;
  std::string key;
  Mode mode = Mode::kVigenere;
  bool autokey = false;
  std::string alphabet_word;
  bool alphabet_keyword_reversed = false;
  bool alphabet_base_reversed = false;
  bool alphabet_keyword_front = true;
  std::string alphabet_string;
  std::string first2;
  std::string plaintext_preview;
  int spacing_matches = -1;
  int spacing_total = 0;
};

struct Options {
  std::string ciphertext;
  std::string wordlist;
  std::string alphabet_wordlist;
  std::string two_letter_list;
  std::string spacing_wordlist;
  std::string spacing_guide;
  std::size_t max_results = 50;
  std::size_t preview_length = 80;
  std::size_t threads = std::max<std::size_t>(1, std::thread::hardware_concurrency());
  bool include_autokey = false;
  double progress_interval_seconds = 1.0;
  bool quiet = false;
  bool include_keyword_front = true;
  bool include_reversed_keyword_front = true;
  bool include_keyword_back = true;
  bool include_keyword_front_reversed_alphabet = true;
  bool include_keyword_back_reversed_alphabet = true;
};

std::string read_file(const std::string &path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("Failed to open file: " + path);
  }
  std::ostringstream buffer;
  buffer << input.rdbuf();
  return buffer.str();
}

std::string clean_letters(const std::string &text) {
  std::string result;
  result.reserve(text.size());
  for (char ch : text) {
    if (ch >= 'A' && ch <= 'Z') {
      result.push_back(ch);
    } else if (ch >= 'a' && ch <= 'z') {
      result.push_back(static_cast<char>('A' + (ch - 'a')));
    }
  }
  return result;
}

std::vector<std::string> parse_wordlist(const std::string &text) {
  std::vector<std::string> words;
  std::string current;
  std::istringstream stream(text);
  while (std::getline(stream, current)) {
    auto cleaned = clean_letters(current);
    if (!cleaned.empty()) {
      words.push_back(std::move(cleaned));
    }
  }
  return words;
}

std::unordered_set<std::string> parse_two_letter_list(const std::string &text) {
  std::unordered_set<std::string> pairs;
  std::string current;
  std::istringstream stream(text);
  while (std::getline(stream, current)) {
    auto cleaned = clean_letters(current);
    if (cleaned.size() == 2) {
      pairs.insert(std::move(cleaned));
    }
  }
  return pairs;
}

std::unordered_set<std::string> build_four_letter_set(const std::vector<std::string> &words) {
  std::unordered_set<std::string> result;
  for (const auto &w : words) {
    if (w.size() == 4) {
      result.insert(w);
    }
  }
  return result;
}

std::unordered_map<int, std::unordered_set<std::string>>
build_words_by_length(const std::vector<std::string> &words) {
  std::unordered_map<int, std::unordered_set<std::string>> grouped;
  for (const auto &word : words) {
    grouped[static_cast<int>(word.size())].insert(word);
  }
  return grouped;
}

std::vector<int> parse_spacing_pattern(const std::string &pattern_text) {
  std::vector<int> pattern;
  int current = 0;
  bool in_number = false;
  for (char ch : pattern_text) {
    if (std::isdigit(static_cast<unsigned char>(ch))) {
      current = current * 10 + (ch - '0');
      in_number = true;
    } else {
      if (in_number) {
        if (current <= 0) {
          throw std::runtime_error("Spacing guide values must be positive integers");
        }
        pattern.push_back(current);
        current = 0;
        in_number = false;
      }
    }
  }
  if (in_number) {
    if (current <= 0) {
      throw std::runtime_error("Spacing guide values must be positive integers");
    }
    pattern.push_back(current);
  }
  return pattern;
}

std::pair<int, int> count_spacing_matches(
    const std::string &plaintext, const std::vector<int> &pattern,
    const std::unordered_map<int, std::unordered_set<std::string>> &words_by_length) {
  if (pattern.empty() || words_by_length.empty()) {
    return {-1, 0};
  }
  int matches = 0;
  int considered = 0;
  std::size_t offset = 0;
  for (int length : pattern) {
    if (length <= 0) {
      continue;
    }
    if (offset + static_cast<std::size_t>(length) > plaintext.size()) {
      break;
    }
    auto it = words_by_length.find(length);
    if (it == words_by_length.end() || it->second.empty()) {
      offset += static_cast<std::size_t>(length);
      continue;
    }
    const std::string word = plaintext.substr(offset, static_cast<std::size_t>(length));
    ++considered;
    if (it->second.find(word) != it->second.end()) {
      ++matches;
    }
    offset += static_cast<std::size_t>(length);
  }
  return {matches, considered};
}

std::string build_keyed_alphabet(const std::string& word, bool keyword_reversed,
    bool alphabet_reversed, bool keyword_front) {
    std::array<bool, 26> seen{};
    std::string ordered_word;
    if (keyword_reversed) {
        ordered_word.assign(word.rbegin(), word.rend());
    }
    else {
        ordered_word = word;
    }

    std::string unique_key;
    unique_key.reserve(ordered_word.size());
    for (char ch : ordered_word) {
        int idx = ch - 'A';
        if (idx >= 0 && idx < 26 && !seen[idx]) {
            seen[idx] = true;
            unique_key.push_back(ch);
        }
    }

    std::string alphabet;
    alphabet.reserve(26);

    if (keyword_front) {
        alphabet.append(unique_key);
    }

    if (alphabet_reversed) {
        for (int i = 25; i >= 0; --i) {
            if (!seen[i]) {
                alphabet.push_back(static_cast<char>('A' + i));
            }
        }
    }
    else {
        for (int i = 0; i < 26; ++i) {
            if (!seen[i]) {
                alphabet.push_back(static_cast<char>('A' + i));
            }
        }
    }

    if (!keyword_front) {
        alphabet.append(unique_key);
    }

    return alphabet;
}

bool same_candidate_identity(const Candidate& a, const Candidate& b)
{
    return a.key == b.key
        && a.mode == b.mode
        && a.autokey == b.autokey
        && a.alphabet_string == b.alphabet_string;
}

AlphabetCandidate make_alphabet_candidate(const std::string &word,
                                          bool keyword_reversed,
                                          bool alphabet_reversed,
                                          bool keyword_front) {
  AlphabetCandidate candidate;
  candidate.base_word = word;
  candidate.keyword_reversed = keyword_reversed;
  candidate.alphabet_reversed = alphabet_reversed;
  candidate.keyword_front = keyword_front;
  candidate.alphabet =
      build_keyed_alphabet(word, keyword_reversed, alphabet_reversed, keyword_front);
  candidate.index_map.fill(-1);
  for (std::size_t i = 0; i < candidate.alphabet.size(); ++i) {
    char ch = candidate.alphabet[i];
    candidate.index_map[ch - 'A'] = static_cast<int>(i);
  }
  return candidate;
}

std::vector<AlphabetCandidate>
build_alphabet_candidates(const std::vector<std::string> &words,
                          const Options &options) {
  std::map<std::string, AlphabetCandidate> unique_map;
  for (const auto &word : words) {
    if (options.include_keyword_front) {
      AlphabetCandidate forward =
          make_alphabet_candidate(word, false, false, true);
      unique_map.emplace(forward.alphabet, forward);
    }
    if (options.include_reversed_keyword_front) {
      AlphabetCandidate reversed_key =
          make_alphabet_candidate(word, true, false, true);
      unique_map.emplace(reversed_key.alphabet, reversed_key);
    }
    if (options.include_keyword_back) {
      AlphabetCandidate back = make_alphabet_candidate(word, false, false, false);
      unique_map.emplace(back.alphabet, back);
    }
    if (options.include_keyword_front_reversed_alphabet) {
      AlphabetCandidate rev_alphabet_front =
          make_alphabet_candidate(word, false, true, true);
      unique_map.emplace(rev_alphabet_front.alphabet, rev_alphabet_front);
    }
    if (options.include_keyword_back_reversed_alphabet) {
      AlphabetCandidate rev_alphabet_back =
          make_alphabet_candidate(word, false, true, false);
      unique_map.emplace(rev_alphabet_back.alphabet, rev_alphabet_back);
    }
    if (true) { //Reversed alphabet and reverse key forgot to add, just always true for now
        AlphabetCandidate rev_alphabet_back =
            make_alphabet_candidate(word, true, true, false);
        unique_map.emplace(rev_alphabet_back.alphabet, rev_alphabet_back);
    }
  }
  std::vector<AlphabetCandidate> result;
  result.reserve(unique_map.size());
  for (auto &entry : unique_map) {
    result.push_back(std::move(entry.second));
  }
  return result;
}

inline int alphabet_index(const AlphabetCandidate &alphabet, char ch) {
  if (ch < 'A' || ch > 'Z') {
    return -1;
  }
  return alphabet.index_map[ch - 'A'];
}

inline std::uint8_t decrypt_symbol(std::uint8_t cipher_idx,
                                   std::uint8_t key_idx, Mode mode) {
  int value = 0;
  switch (mode) {
  case Mode::kVigenere:
    value = static_cast<int>(cipher_idx) - static_cast<int>(key_idx);
    if (value < 0) {
      value += 26;
    }
    break;
  case Mode::kBeaufort:
    value = static_cast<int>(key_idx) - static_cast<int>(cipher_idx);
    if (value < 0) {
      value += 26;
    }
    break;
  case Mode::kVariantBeaufort:
    value = static_cast<int>(cipher_idx) + static_cast<int>(key_idx);
    if (value >= 26) {
      value -= 26;
    }
    break;
  }
  return static_cast<std::uint8_t>(value);
}

std::string decrypt_repeating(
    const std::string& cipher, const std::string& key,
    const AlphabetCandidate& alphabet, Mode mode,
    const std::vector<std::uint8_t>& cipher_indices,   // precomputed per alphabet
    const std::vector<std::uint8_t>& letter_mask,      // precomputed once per worker
    std::vector<std::uint8_t>& key_indices,            // scratch per key
    std::vector<std::uint8_t>& key_valid,              // scratch per key
    std::vector<std::uint8_t>& plaintext_indices       // scratch per decrypt
#if defined(__AVX2__)
    , std::vector<KeyBlock>& key_blocks                // scratch per decrypt (AVX2)
#endif
)
{
    if (cipher.empty() || key.empty()) {
        return {};
    }

    const std::size_t text_len = cipher.size();
    const std::size_t key_len = key.size();
    const std::string& alph = alphabet.alphabet;

    std::string result = cipher;

#if defined(__AVX2__)
    if (text_len >= 32) {


        const __m256i zero = _mm256_setzero_si256();
        const __m256i twenty_six = _mm256_set1_epi8(26);
        const __m256i twenty_five = _mm256_set1_epi8(25);

        std::size_t vec_index = 0;
        std::size_t block_offset = 0;

        switch (mode) {
        case Mode::kVigenere:
            while (vec_index + 32 <= text_len) {
                __m256i cipher_vec = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(cipher_indices.data() + vec_index));
                __m256i key_vec = _mm256_load_si256(
                    reinterpret_cast<const __m256i*>(key_blocks[block_offset].data.data()));
                __m256i plain_vec = _mm256_sub_epi8(cipher_vec, key_vec);
                __m256i mask = _mm256_cmpgt_epi8(zero, plain_vec);
                plain_vec = _mm256_add_epi8(plain_vec,
                    _mm256_and_si256(mask, twenty_six));
                _mm256_storeu_si256(
                    reinterpret_cast<__m256i*>(plaintext_indices.data() + vec_index),
                    plain_vec);
                vec_index += 32;
                block_offset = (block_offset + 32) % key_len;
            }
            break;

        case Mode::kBeaufort:
            while (vec_index + 32 <= text_len) {
                __m256i cipher_vec = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(cipher_indices.data() + vec_index));
                __m256i key_vec = _mm256_load_si256(
                    reinterpret_cast<const __m256i*>(key_blocks[block_offset].data.data()));
                __m256i plain_vec = _mm256_sub_epi8(key_vec, cipher_vec);
                __m256i mask = _mm256_cmpgt_epi8(zero, plain_vec);
                plain_vec = _mm256_add_epi8(plain_vec,
                    _mm256_and_si256(mask, twenty_six));
                _mm256_storeu_si256(
                    reinterpret_cast<__m256i*>(plaintext_indices.data() + vec_index),
                    plain_vec);
                vec_index += 32;
                block_offset = (block_offset + 32) % key_len;
            }
            break;

        case Mode::kVariantBeaufort:
            while (vec_index + 32 <= text_len) {
                __m256i cipher_vec = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(cipher_indices.data() + vec_index));
                __m256i key_vec = _mm256_load_si256(
                    reinterpret_cast<const __m256i*>(key_blocks[block_offset].data.data()));
                __m256i plain_vec = _mm256_add_epi8(cipher_vec, key_vec);
                __m256i mask = _mm256_cmpgt_epi8(plain_vec, twenty_five);
                plain_vec = _mm256_sub_epi8(plain_vec,
                    _mm256_and_si256(mask, twenty_six));
                _mm256_storeu_si256(
                    reinterpret_cast<__m256i*>(plaintext_indices.data() + vec_index),
                    plain_vec);
                vec_index += 32;
                block_offset = (block_offset + 32) % key_len;
            }
            break;
        }

        // scalar tail
        for (std::size_t i = vec_index; i < text_len; ++i) {
            if (!letter_mask[i] || !key_valid[i % key_len]) {
                continue;
            }
            plaintext_indices[i] =
                decrypt_symbol(cipher_indices[i], key_indices[i % key_len], mode);
        }
    }
    else
#endif
    {
        // pure scalar path
        for (std::size_t i = 0; i < text_len; ++i) {
            if (!letter_mask[i] || !key_valid[i % key_len]) {
                continue;
            }
            plaintext_indices[i] =
                decrypt_symbol(cipher_indices[i], key_indices[i % key_len], mode);
        }
    }

     //map plaintext indices back to letters
    for (std::size_t i = 0; i < text_len; ++i) {
        if (!letter_mask[i] || !key_valid[i % key_len]) {
            continue;
        }
        std::uint8_t idx = plaintext_indices[i];
        if (idx < alph.size()) {
            result[i] = alph[idx];
        }
    }

    return result;
}


std::string decrypt_autokey(const std::string &cipher, const std::string &key,
                            const AlphabetCandidate &alphabet, Mode mode) {
  if (cipher.empty() || key.empty()) {
    return {};
  }
  std::string result;
  result.reserve(cipher.size());
  const std::string &alph = alphabet.alphabet;
  const std::size_t key_len = key.size();
  for (std::size_t i = 0; i < cipher.size(); ++i) {
    char c = cipher[i];
    int c_idx = alphabet_index(alphabet, c);
    char k_char;
    if (i < key_len) {
      k_char = key[i];
    } else {
      k_char = result[i - key_len];
    }
    int k_idx = alphabet_index(alphabet, k_char);
    if (c_idx < 0 || k_idx < 0) {
      result.push_back(c);
      continue;
    }
    int p_idx = 0;
    switch (mode) {
    case Mode::kVigenere:
      p_idx = (c_idx - k_idx + 26) % 26;
      break;
    case Mode::kBeaufort:
      p_idx = (k_idx - c_idx + 26) % 26;
      break;
    case Mode::kVariantBeaufort:
      p_idx = (c_idx + k_idx) % 26;
      break;
    }
    result.push_back(alph[p_idx]);
  }
  return result;
}

double index_of_coincidence(const std::string &text) {
  const std::size_t n = text.size();
  if (n <= 1) {
    return 0.0;
  }
  std::array<std::size_t, 26> counts{};
  for (char ch : text) {
    if (ch >= 'A' && ch <= 'Z') {
      counts[ch - 'A']++;
    }
  }
  double numerator = 0.0;
  for (std::size_t count : counts) {
    numerator += static_cast<double>(count) * static_cast<double>(count - 1);
  }
  return numerator / (static_cast<double>(n) * static_cast<double>(n - 1));
}

double chi_square(const std::string &text) {
  const std::size_t n = text.size();
  if (n == 0) {
    return std::numeric_limits<double>::infinity();
  }
  std::array<std::size_t, 26> counts{};
  for (char ch : text) {
    if (ch >= 'A' && ch <= 'Z') {
      counts[ch - 'A']++;
    }
  }
  double chi = 0.0;
  for (std::size_t i = 0; i < 26; ++i) {
    double expected = static_cast<double>(n) * kEnglishFreq[i];
    if (expected <= 0.0) {
      continue;
    }
    double diff = static_cast<double>(counts[i]) - expected;
    chi += (diff * diff) / expected;
  }
  return chi;
}

void compute_stats(const std::string& text, double& out_ioc, double& out_chi) {
    const std::size_t n = text.size();
    if (n <= 1) {
        out_ioc = 0.0;
        out_chi = std::numeric_limits<double>::infinity();
        return;
    }

    std::array<std::size_t, 26> counts{};
    for (char ch : text) {
        if (ch >= 'A' && ch <= 'Z') {
            counts[ch - 'A']++;
        }
    }

    double numerator = 0.0;
    double chi = 0.0;
    for (std::size_t i = 0; i < 26; ++i) {
        std::size_t c = counts[i];
        numerator += static_cast<double>(c) * static_cast<double>(c - 1);

        double expected = static_cast<double>(n) * kEnglishFreq[i];
        if (expected > 0.0) {
            double diff = static_cast<double>(c) - expected;
            chi += (diff * diff) / expected;
        }
    }

    out_ioc = numerator / (static_cast<double>(n) * static_cast<double>(n - 1));
    out_chi = (n == 0 ? std::numeric_limits<double>::infinity() : chi);
}

Candidate make_candidate(
    const std::string &key_word, const AlphabetCandidate &alphabet, Mode mode,
    bool autokey_variant, const std::string &plaintext,
    std::size_t preview_length, const std::vector<int> *spacing_pattern,
    const std::unordered_map<int, std::unordered_set<std::string>>
        *spacing_words_by_length) {
  Candidate cand;
  cand.key = key_word;
  cand.mode = mode;
  cand.autokey = autokey_variant;
  cand.alphabet_word = alphabet.base_word;
  cand.alphabet_keyword_reversed = alphabet.keyword_reversed;
  cand.alphabet_base_reversed = alphabet.alphabet_reversed;
  cand.alphabet_keyword_front = alphabet.keyword_front;
  cand.alphabet_string = alphabet.alphabet;
  //cand.first2 = plaintext.substr(0, std::min<std::size_t>(2, plaintext.size()));
  cand.plaintext_preview = plaintext.substr(0, std::min(preview_length, plaintext.size()));
  compute_stats(plaintext, cand.ioc, cand.chi);

  const double ioc_delta = std::abs(cand.ioc - kIocTarget);
  const double ioc_score = std::max(0.0, 1.0 - ioc_delta / 0.02);
  const double chi_clamped = std::min(400.0, cand.chi);
  const double chi_score = std::max(0.0, 1.0 - chi_clamped / 400.0);
  const double quality_factor = 0.1 + 0.9 * chi_score;
  const double stats_score = ioc_score * quality_factor;
  const double word_weight = 0.8;
  const double stats_weight = 1.0 - word_weight;
  cand.score = stats_score;
  if (cand.ioc > .05 && cand.chi < 160 && spacing_pattern && spacing_words_by_length) 
  {
    auto result = count_spacing_matches(plaintext, *spacing_pattern,
                                        *spacing_words_by_length);
    cand.spacing_matches = result.first;
    cand.spacing_total = result.second;
    if (cand.spacing_matches >= 0 && cand.spacing_total > 0) {
      double word_score = static_cast<double>(cand.spacing_matches) /
                          static_cast<double>(cand.spacing_total);
      cand.score = word_weight * word_score + stats_weight * stats_score;
    }
  }
  return cand;
}




void maintain_top_results(std::vector<Candidate> &results, const Candidate &candidate,
                          std::size_t max_results) {
    if (max_results == 0) {
        return;
    }

    // 1. Check if we already have this identity
    auto existing = std::find_if(results.begin(), results.end(),
        [&](const Candidate& c) {
            return same_candidate_identity(c, candidate);
        });

    if (existing != results.end()) {
        // Optional: upgrade if the new one is better
        if (candidate.score > existing->score) {
            *existing = candidate;
        }
        return; // don't add a duplicate row
    }

    // 2. Normal insert/replace logic
    if (results.size() < max_results) {
        results.push_back(candidate);
        return;
    }

    auto worst_it = std::min_element(
        results.begin(), results.end(),
        [](const Candidate& a, const Candidate& b) { return a.score < b.score; });

    if (worst_it != results.end() && worst_it->score < candidate.score) {
        *worst_it = candidate;
    }
}

Options parse_options(int argc, char *argv[]) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    auto require_value = [&](const std::string &name) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error("Missing value for option " + name);
      }
      return argv[++i];
    };

    if (arg == "--ciphertext") {
      options.ciphertext = require_value(arg);
    } else if (arg == "--ciphertext-file") {
      options.ciphertext = read_file(require_value(arg));
    } else if (arg == "--wordlist") {
      options.wordlist = read_file(require_value(arg));
    } else if (arg == "--wordlist-inline") {
      options.wordlist = require_value(arg);
    } else if (arg == "--alphabet-wordlist") {
      options.alphabet_wordlist = read_file(require_value(arg));
    } else if (arg == "--alphabet-wordlist-inline") {
      options.alphabet_wordlist = require_value(arg);
    } else if (arg == "--two-letter-list") {
      options.two_letter_list = read_file(require_value(arg));
    } else if (arg == "--two-letter-inline") {
      options.two_letter_list = require_value(arg);
    } else if (arg == "--spacing-wordlist") {
      options.spacing_wordlist = read_file(require_value(arg));
    } else if (arg == "--spacing-wordlist-inline") {
      options.spacing_wordlist = require_value(arg);
    } else if (arg == "--spacing-guide") {
      options.spacing_guide = require_value(arg);
    } else if (arg == "--spacing-guide-file") {
      options.spacing_guide = read_file(require_value(arg));
    } else if (arg == "--max-results") {
      options.max_results = static_cast<std::size_t>(std::stoul(require_value(arg)));
    } else if (arg == "--preview-length") {
      options.preview_length = static_cast<std::size_t>(std::stoul(require_value(arg)));
    } else if (arg == "--threads") {
      options.threads = static_cast<std::size_t>(std::stoul(require_value(arg)));
    } else if (arg == "--include-autokey") {
      options.include_autokey = true;
    } else if (arg == "--progress-interval") {
      options.progress_interval_seconds = std::stod(require_value(arg));
    } else if (arg == "--quiet") {
      options.quiet = true;
    } else if (arg == "--no-keyword-front") {
      options.include_keyword_front = false;
    } else if (arg == "--no-reversed-keyword-front") {
      options.include_reversed_keyword_front = false;
    } else if (arg == "--no-keyword-back") {
      options.include_keyword_back = false;
    } else if (arg == "--no-keyword-front-reversed") {
      options.include_keyword_front_reversed_alphabet = false;
    } else if (arg == "--no-keyword-back-reversed") {
      options.include_keyword_back_reversed_alphabet = false;
    } else if (arg == "--help" || arg == "-h") {
      std::cout << "Quagmire III wordlist bruteforcer (C++)\n"
                << "Options:\n"
                << "  --ciphertext <text>             Inline ciphertext string\n"
                << "  --ciphertext-file <path>        File containing ciphertext\n"
                << "  --wordlist <path>               Main wordlist file (required)\n"
                << "  --wordlist-inline <text>        Inline wordlist string\n"
                << "  --alphabet-wordlist <path>      Alphabet wordlist file (defaults to main)\n"
                << "  --alphabet-wordlist-inline <text> Inline alphabet wordlist string\n"
                << "  --two-letter-list <path>        Optional 2-letter filter list\n"
                << "  --two-letter-inline <text>      Inline 2-letter filter string\n"
                << "  --spacing-wordlist <path>       Wordlist used for spacing guide scoring\n"
                << "  --spacing-wordlist-inline <text> Inline spacing wordlist string\n"
                << "  --spacing-guide <pattern>       Word length pattern (e.g. 2-4-3-3)\n"
                << "  --spacing-guide-file <path>     File containing word length pattern\n"
                << "  --max-results <N>               Max candidates to keep (default 50)\n"
                << "  --preview-length <N>            Plaintext preview length (default 80)\n"
                << "  --threads <N>                   Worker threads (default hardware)\n"
                << "  --include-autokey               Try autokey variants too\n"
                << "  --progress-interval <sec>       Progress update interval (default 1.0)\n"
                << "  --quiet                         Suppress periodic progress output\n"
                << "  --no-keyword-front              Skip key prefix on normal alphabet\n"
                << "  --no-reversed-keyword-front     Skip reversed key prefix variant\n"
                << "  --no-keyword-back               Skip key suffix on normal alphabet\n"
                << "  --no-keyword-front-reversed     Skip key prefix on reversed alphabet\n"
                << "  --no-keyword-back-reversed      Skip key suffix on reversed alphabet\n"
                << "  --help                          Show this help message\n";
      std::exit(0);
    } else {
      throw std::runtime_error("Unknown option: " + arg);
    }
  }

  if (options.ciphertext.empty()) {
    throw std::runtime_error("Ciphertext is required (use --ciphertext or --ciphertext-file)");
  }
  if (options.wordlist.empty()) {
    throw std::runtime_error("Wordlist is required (use --wordlist or --wordlist-inline)");
  }
  if (options.alphabet_wordlist.empty()) {
    options.alphabet_wordlist = options.wordlist;
  }
  if (!options.spacing_guide.empty() && options.spacing_wordlist.empty()) {
    throw std::runtime_error(
        "Spacing guide provided but spacing wordlist is missing (use --spacing-wordlist)");
  }
  if (options.threads == 0) {
    options.threads = 1;
  }
  if (!options.include_keyword_front && !options.include_reversed_keyword_front &&
      !options.include_keyword_back &&
      !options.include_keyword_front_reversed_alphabet &&
      !options.include_keyword_back_reversed_alphabet) {
    throw std::runtime_error(
        "All alphabet construction variants are disabled. Enable at least one option.");
  }
  return options;
}

std::string mode_family_name(Mode mode) {
  switch (mode) {
  case Mode::kVigenere:
    return "Vigenere";
  case Mode::kBeaufort:
    return "Beaufort";
  case Mode::kVariantBeaufort:
    return "Beaufort variant";
  }
  return "";
}

std::string format_results_table(const std::vector<Candidate> &candidates,
                                 std::size_t max_rows) {
  std::ostringstream oss;
  if (candidates.empty()) {
    oss << "No candidates met the filtering criteria." << '\n';
    return oss.str();
  }

  std::vector<const Candidate *> sorted;
  sorted.reserve(candidates.size());
  for (const auto &cand : candidates) {
    sorted.push_back(&cand);
  }
  std::sort(sorted.begin(), sorted.end(), [](const Candidate *a, const Candidate *b) {
    return a->score > b->score;
  });

  const std::size_t limit =
      std::min<std::size_t>({sorted.size(), max_rows, static_cast<std::size_t>(50)});

  oss << std::setw(3) << "#" << "  " << std::setw(7) << "Score" << "  "
      << std::setw(7) << "Words" << "  " << std::setw(8) << "IoC" << "  "
      << std::setw(9) << "Chi^2" << "  "
      << std::left << std::setw(16) << "Cipher" << "  " << std::setw(8)
      << "Autokey" << "  " << std::setw(18) << "Key" << "  " << std::setw(15)
      << "Alphabet word" << "  " << std::setw(6) << "KeyRev" << "  "
      << std::setw(6) << "Base" << "  " << std::setw(4) << "Pos" << "  "
      << "Plaintext" << '\n';
  oss << std::string(140, '-') << '\n';
  for (std::size_t i = 0; i < limit; ++i) {
    const Candidate &cand = *sorted[i];
    std::string word_summary;
    if (cand.spacing_matches >= 0 && cand.spacing_total > 0) {
      word_summary = std::to_string(cand.spacing_matches) + "/" +
                     std::to_string(cand.spacing_total);
    } else if (cand.spacing_matches == 0 && cand.spacing_total == 0) {
      word_summary = "0/0";
    } else {
      word_summary = "--";
    }
    oss << std::right << std::setw(3) << (i + 1) << "  " << std::fixed
        << std::setprecision(3) << std::setw(7) << cand.score << "  "
        << std::setw(7) << word_summary << "  "
        << std::setprecision(4) << std::setw(8) << cand.ioc << "  "
        << std::setprecision(2) << std::setw(9) << cand.chi << "  "
        << std::left << std::setw(16) << mode_family_name(cand.mode) << "  "
        << std::setw(8) << (cand.autokey ? "Yes" : "No") << "  "
        << std::setw(18) << cand.key << "  " << std::setw(15)
        << cand.alphabet_word << "  " << std::setw(6)
        << (cand.alphabet_keyword_reversed ? "Yes" : "No") << "  "
        << std::setw(6) << (cand.alphabet_base_reversed ? "Rev" : "Fwd") << "  "
        << std::setw(4) << (cand.alphabet_keyword_front ? "Pre" : "Suf") << "  "
        << cand.plaintext_preview << '\n';
  }
  return oss.str();
}

void print_table(const std::vector<Candidate> &candidates, std::size_t max_rows) {
  std::cout << format_results_table(candidates, max_rows);
}

void write_results_to_file(const std::vector<Candidate> &candidates,
                           std::size_t max_rows,
                           const std::string &path) {
  std::ofstream output(path);
  if (!output) {
    throw std::runtime_error("Failed to open results output file: " + path);
  }
  output << format_results_table(candidates, max_rows);
}

bool passes_front_filters_repeating(
    const std::string& cipher,
    const std::string& key,
    const AlphabetCandidate& alphabet,
    Mode mode,
    bool have_first2_filter,
    bool have_second4_filter,
    const std::unordered_set<std::string>& two_letter_set,
    const std::unordered_set<std::string>& four_letter_set)
{
    const std::size_t text_len = cipher.size();
    if (text_len < 2) {
        return false;
    }

    const std::size_t key_len = key.size();
    if (key_len == 0) {
        return false;
    }

    const std::string& alph = alphabet.alphabet;

    // We only need up to 6 chars: 2 for first word, 4 for second word.
    const std::size_t needed =
        have_second4_filter ? std::min<std::size_t>(6, text_len) : 2;

    char buf[6];
    std::size_t produced = 0;

    for (std::size_t i = 0; i < needed; ++i) {
        char c = cipher[i];
        char out = c;  // default: keep cipher char (matches current behavior
        // when key/alphabet indices are invalid)

        int c_idx = alphabet_index(alphabet, c);
        if (c_idx >= 0) {
            char kch = key[i % key_len];
            int k_idx = alphabet_index(alphabet, kch);
            if (k_idx >= 0) {
                std::uint8_t p_idx = decrypt_symbol(
                    static_cast<std::uint8_t>(c_idx),
                    static_cast<std::uint8_t>(k_idx),
                    mode
                );
                if (p_idx < alph.size()) {
                    out = alph[p_idx];
                }
            }
        }

        buf[produced++] = out;
    }

    if (produced < 2) {
        return false;
    }

    if (have_first2_filter) {
        std::string first2(buf, buf + 2);
        if (two_letter_set.find(first2) == two_letter_set.end()) {
            return false;
        }
    }

    if (have_second4_filter && produced >= 6) {
        std::string second4(buf + 2, buf + 6);
        if (four_letter_set.find(second4) == four_letter_set.end()) {
            return false;
        }
    }

    return true;
}


struct WorkerResult {
  std::vector<Candidate> best;
  std::size_t combos = 0;
  std::size_t autokey_attempts = 0;
  std::size_t keys_processed = 0;
};

WorkerResult process_keys(
    const std::vector<std::string>& keys, std::size_t begin,
    std::size_t end, const std::string& cipher,
    const std::vector<AlphabetCandidate>& alphabets,
    const std::vector<Mode>& modes,
    const std::unordered_set<std::string>& two_letter_set,
    const std::unordered_set<std::string>& four_letter_set,
    bool have_first2_filter, bool have_second4_filter,
    const std::vector<int>* spacing_pattern,
    const std::unordered_map<int, std::unordered_set<std::string>>* spacing_words_by_length,
    std::size_t max_results, std::size_t preview_length,
    bool include_autokey, std::atomic<std::size_t>& combos_counter,
    std::atomic<std::size_t>& autokey_counter,
    std::atomic<std::size_t>& keys_counter,
    std::mutex& results_mutex,
    std::vector<Candidate>& global_results)
{
    WorkerResult result;
    result.best.reserve(max_results);

    const std::size_t text_len = cipher.size();
    const std::size_t alphabet_count = alphabets.size();

    // per-worker scratch buffers
    std::vector<std::uint8_t> letter_mask(text_len);
    std::vector<std::uint8_t> cipher_indices(text_len);
    std::vector<std::uint8_t> plaintext_indices(text_len);

    std::vector<std::uint8_t> key_indices;
    std::vector<std::uint8_t> key_valid;
    key_indices.reserve(64);
    key_valid.reserve(64);
#if defined(__AVX2__)
    std::vector<KeyBlock> key_blocks;
#endif

    // build letter_mask once (A–Z only)
    for (std::size_t i = 0; i < text_len; ++i) {
        char c = cipher[i];
        letter_mask[i] = (c >= 'A' && c <= 'Z') ? 1u : 0u;
    }

    // Outer: alphabets
    for (std::size_t a = 0; a < alphabet_count; ++a) {
        const auto& alphabet = alphabets[a];

        // build cipher_indices for this alphabet once
        for (std::size_t i = 0; i < text_len; ++i) {
            if (!letter_mask[i]) {
                cipher_indices[i] = 0;
                continue;
            }
            int idx = alphabet_index(alphabet, cipher[i]);
            cipher_indices[i] = (idx >= 0)
                ? static_cast<std::uint8_t>(idx)
                : 0;
        }

        const bool last_alphabet = (a + 1 == alphabet_count);

        // Inner: keys assigned to this worker
        for (std::size_t idx = begin; idx < end; ++idx) {
            const std::string& key_word = keys[idx];


            key_indices.resize(key_word.size());
            key_valid.resize(key_word.size());
            for (std::size_t i = 0; i < key_word.size(); ++i) {
                int idx = alphabet_index(alphabet, key_word[i]);
                if (idx < 0) {
                    key_valid[i] = 0;
                    key_indices[i] = 0;
                }
                else {
                    key_valid[i] = 1;
                    key_indices[i] = static_cast<std::uint8_t>(idx);
                }
            }

            bool keyblockBuilt = false;

            for (Mode mode : modes) 
            {

                ++result.combos;
                ++combos_counter;

                if ((have_first2_filter || have_second4_filter) &&
                    !passes_front_filters_repeating(
                        cipher,
                        key_word,
                        alphabet,
                        mode,
                        have_first2_filter,
                        have_second4_filter,
                        two_letter_set,
                        four_letter_set))
                {
                    continue; // reject this (key, alphabet, mode) without full decrypt
                }

#if defined(__AVX2__)

                if (!keyblockBuilt)
                {
                    // build per-decrypt key_blocks, but reuse the vector
                    const size_t key_len = key_word.size();
                    key_blocks.resize(key_len);
                    for (std::size_t start = 0; start < key_len; ++start) {
                        auto& block = key_blocks[start];
                        for (std::size_t j = 0; j < block.data.size(); ++j) {
                            block.data[j] = key_indices[(start + j) % key_len];
                        }
                    }
                    keyblockBuilt = true;
                }
#endif

                std::string plaintext =
                    decrypt_repeating(cipher, key_word, alphabet, mode,
                        cipher_indices, letter_mask,
                        key_indices, key_valid,
                        plaintext_indices
#if defined(__AVX2__)
                        , key_blocks
#endif
                    );

                Candidate cand = make_candidate(
                    key_word, alphabet, mode, false,
                    plaintext, preview_length,
                    spacing_pattern, spacing_words_by_length);
                maintain_top_results(result.best, cand, max_results);

                if (include_autokey) {
                    std::string plaintext_auto =
                        decrypt_autokey(cipher, key_word, alphabet, mode);
                    ++result.autokey_attempts;
                    ++autokey_counter;

                    Candidate cand_auto = make_candidate(
                        key_word, alphabet, mode, true,
                        plaintext_auto, preview_length,
                        spacing_pattern, spacing_words_by_length);
                    maintain_top_results(result.best, cand_auto, max_results);
                }
            }


        }

        // count each key once (not once per alphabet)
        ++result.keys_processed;
        ++keys_counter;

        //merge local best into global (you can make this periodic later)
        if (a % 1000 == 0)
        {
            std::lock_guard<std::mutex> lock(results_mutex);
            for (const auto& cand : result.best) {
                maintain_top_results(global_results, cand, max_results);
            }
        }
    }

    return result;
}


void progress_loop(const std::atomic<bool> &done, double interval_seconds,
                   std::atomic<std::size_t> &combos_counter,
                   std::atomic<std::size_t> &keys_counter,
                   std::atomic<std::size_t> &autokey_counter,
                   std::size_t total_combos, std::mutex &results_mutex,
                   std::vector<Candidate> &results, std::size_t max_rows) {
  using clock = std::chrono::steady_clock;
  const auto start = clock::now();
  while (!done.load(std::memory_order_relaxed)) {
    std::this_thread::sleep_for(std::chrono::duration<double>(interval_seconds));
    const auto now = clock::now();
    const double elapsed =
        std::chrono::duration_cast<std::chrono::duration<double>>(now - start)
            .count();
    std::size_t combos = combos_counter.load();
    std::size_t keys = keys_counter.load();
    std::size_t autokeys = autokey_counter.load();
    double pct = total_combos == 0
                     ? 0.0
                     : (static_cast<double>(combos) /
                        static_cast<double>(total_combos)) * 100.0;
    std::vector<Candidate> snapshot;
    {
      std::lock_guard<std::mutex> lock(results_mutex);
      snapshot = results;
    }
    std::ostringstream oss;
#if defined(_WIN32)
    // Simple version: delegate to the shell.
    std::system("cls");
#else
    // ANSI clear + home
    std::cout << "\033[2J\033[H";
#endif
    oss << std::fixed << std::setprecision(1) << "Elapsed: " << elapsed
        << "s  Alphabets: " << keys << "  Combos: " << combos;
    if (total_combos > 0) {
      oss << " (" << std::setprecision(2) << pct << "%)";
    }
    oss << "  Autokey attempts: " << autokeys << '\n';
    oss << '\n';
    oss << format_results_table(snapshot, max_rows);
    std::cout << oss.str() << std::flush;
  }
}

} // namespace

int main(int argc, char *argv[]) {
  try {
    Options options = parse_options(argc, argv);

#ifdef __AVX2__
    std::cout << "__AVX2__ is defined\n";
#else
    std::cout << "__AVX2__ is NOT defined\n";
#endif

    const std::string cipher = clean_letters(options.ciphertext);
    const std::vector<std::string> key_words = parse_wordlist(options.wordlist);
    if (key_words.empty()) {
      throw std::runtime_error("Wordlist is empty after cleaning");
    }
    const std::vector<std::string> alphabet_words =
        parse_wordlist(options.alphabet_wordlist);
    if (alphabet_words.empty()) {
      throw std::runtime_error(
          "Alphabet wordlist is empty after cleaning");
    }
    std::vector<std::string> spacing_words;
    if (!options.spacing_wordlist.empty()) {
      spacing_words = parse_wordlist(options.spacing_wordlist);
      if (spacing_words.empty()) {
        throw std::runtime_error(
            "Spacing wordlist is empty after cleaning");
      }
    }
    std::vector<int> spacing_pattern;
    std::unordered_map<int, std::unordered_set<std::string>>
        spacing_words_by_length;
    if (!options.spacing_guide.empty()) {
      spacing_pattern = parse_spacing_pattern(options.spacing_guide);
      if (spacing_pattern.empty()) {
        throw std::runtime_error(
            "Spacing guide did not contain any valid word lengths");
      }
      if (spacing_words.empty()) {
        throw std::runtime_error(
            "Spacing guide requires a spacing wordlist");
      }
      spacing_words_by_length = build_words_by_length(spacing_words);
    }
    const bool spacing_scoring_enabled =
        !spacing_pattern.empty() && !spacing_words_by_length.empty();
    const std::unordered_set<std::string> two_letter_set =
        parse_two_letter_list(options.two_letter_list);
    const std::unordered_set<std::string> four_letter_set = !spacing_words.empty()
                                                                ? build_four_letter_set(spacing_words)
                                                                : build_four_letter_set(key_words);
    const bool have_first2_filter = !two_letter_set.empty();
    const bool have_second4_filter = !four_letter_set.empty();

    std::vector<AlphabetCandidate> alphabets =
        build_alphabet_candidates(alphabet_words, options);
    if (alphabets.empty()) {
      throw std::runtime_error("No alphabet candidates generated from wordlist");
    }

    std::vector<Mode> modes = {Mode::kVigenere, Mode::kBeaufort,
                               Mode::kVariantBeaufort};

    const std::size_t total_combos =
        key_words.size() * alphabets.size() * modes.size();

    std::cout << "Cipher length: " << cipher.size() << '\n';
    std::cout << "Key candidates: " << key_words.size() << '\n';
    std::cout << "Alphabet candidates: " << alphabets.size() << '\n';
    std::cout << "Modes per key: " << modes.size() << '\n';
    std::cout << "Total combinations: " << total_combos << '\n';
    if (options.include_autokey) {
      std::cout << "Autokey variants enabled" << '\n';
    }
    std::cout << "Using " << options.threads << " worker threads" << '\n' << std::endl;

    std::atomic<std::size_t> combos_counter{0};
    std::atomic<std::size_t> autokey_counter{0};
    std::atomic<std::size_t> keys_counter{0};

    const std::size_t results_limit =
        std::max<std::size_t>(options.max_results, static_cast<std::size_t>(50));
    const std::vector<int> *spacing_pattern_ptr =
        spacing_scoring_enabled ? &spacing_pattern : nullptr;
    const std::unordered_map<int, std::unordered_set<std::string>>
        *spacing_words_ptr =
            spacing_scoring_enabled ? &spacing_words_by_length : nullptr;

    std::vector<Candidate> global_results;
    global_results.reserve(results_limit);
    std::mutex results_mutex;

    std::vector<std::thread> workers;
    const std::size_t thread_count = std::max<std::size_t>(1, options.threads);
    const std::size_t base_chunk = key_words.size() / thread_count;
    const std::size_t remainder = key_words.size() % thread_count;

    std::atomic<bool> done{false};
    std::thread progress_thread;
    if (!options.quiet && options.progress_interval_seconds > 0.0) {
      progress_thread = std::thread(progress_loop, std::cref(done),
                                    options.progress_interval_seconds,
                                    std::ref(combos_counter),
                                    std::ref(keys_counter),
                                    std::ref(autokey_counter), total_combos,
                                    std::ref(results_mutex),
                                    std::ref(global_results), results_limit);
    }

    auto start_time = std::chrono::steady_clock::now();

    std::size_t current = 0;
    for (std::size_t t = 0; t < thread_count; ++t) {
      std::size_t chunk = base_chunk + (t < remainder ? 1 : 0);
      std::size_t begin = current;
      std::size_t end = begin + chunk;
      current = end;
      if (begin >= end) {
        continue;
      }
      workers.emplace_back([&, begin, end]() {
        WorkerResult worker_result = process_keys(
            key_words, begin, end, cipher, alphabets, modes, two_letter_set,
            four_letter_set, have_first2_filter, have_second4_filter,
            spacing_pattern_ptr, spacing_words_ptr,
            results_limit, options.preview_length, options.include_autokey,
            combos_counter, autokey_counter, keys_counter, results_mutex, global_results);
        std::lock_guard<std::mutex> lock(results_mutex);
        for (const auto &cand : worker_result.best) {
          maintain_top_results(global_results, cand, results_limit);
        }
      });
    }

    for (auto &thread : workers) {
      if (thread.joinable()) {
        thread.join();
      }
    }

    done.store(true, std::memory_order_relaxed);
    if (progress_thread.joinable()) {
      progress_thread.join();
    }

    auto end_time = std::chrono::steady_clock::now();
    double elapsed_seconds =
        std::chrono::duration_cast<std::chrono::duration<double>>(end_time -
                                                                  start_time)
            .count();

    std::cout << std::fixed << std::setprecision(2)
              << "Elapsed time: " << elapsed_seconds << "s" << '\n';
    std::cout << "Combos tried: " << combos_counter.load() << '/' << total_combos
              << '\n';
    if (options.include_autokey) {
      std::cout << "Autokey attempts: " << autokey_counter.load() << '\n';
    }
    std::cout << "Keys processed: " << keys_counter.load() << '/' << key_words.size()
              << '\n' << std::endl;

    print_table(global_results, results_limit);
    const std::string results_path = "bruteforce_results.txt";
    try {
      write_results_to_file(global_results, results_limit, results_path);
      std::cout << "Results written to " << results_path << '\n';
    } catch (const std::exception &file_ex) {
      std::cerr << "Failed to write results file: " << file_ex.what() << '\n';
    }
#if defined(_WIN32)
    std::cout << "Press Enter to exit..." << std::endl;
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
#endif
    return 0;
  } catch (const std::exception &ex) {
    std::cerr << "Error: " << ex.what() << std::endl;

#if defined(_WIN32)
    std::cout << "Press Enter to exit..." << std::endl;
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
#endif

    return 1;
  }
}


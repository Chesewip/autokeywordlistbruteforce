#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdlib>
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
#include <unordered_set>
#include <utility>
#include <vector>

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
  bool reversed = false;
};

struct Candidate {
  double score = 0.0;
  double ioc = 0.0;
  double chi = 0.0;
  std::string key;
  Mode mode = Mode::kVigenere;
  bool autokey = false;
  std::string alphabet_word;
  bool alphabet_reversed = false;
  std::string alphabet_string;
  std::string first2;
  std::string plaintext_preview;
};

struct Options {
  std::string ciphertext;
  std::string wordlist;
  std::string two_letter_list;
  std::size_t max_results = 50;
  std::size_t preview_length = 80;
  std::size_t threads = std::max<std::size_t>(1, std::thread::hardware_concurrency());
  bool include_autokey = false;
  double progress_interval_seconds = 1.0;
  bool quiet = false;
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

std::string build_keyed_alphabet(const std::string &word) {
  std::array<bool, 26> seen{};
  std::string alphabet;
  alphabet.reserve(26);
  for (char ch : word) {
    int idx = ch - 'A';
    if (idx >= 0 && idx < 26 && !seen[idx]) {
      seen[idx] = true;
      alphabet.push_back(ch);
    }
  }
  for (int i = 0; i < 26; ++i) {
    if (!seen[i]) {
      alphabet.push_back(static_cast<char>('A' + i));
    }
  }
  return alphabet;
}

AlphabetCandidate make_alphabet_candidate(const std::string &word, bool reversed) {
  AlphabetCandidate candidate;
  candidate.base_word = word;
  candidate.reversed = reversed;
  if (reversed) {
    std::string reversed_word(word.rbegin(), word.rend());
    candidate.alphabet = build_keyed_alphabet(reversed_word);
  } else {
    candidate.alphabet = build_keyed_alphabet(word);
  }
  candidate.index_map.fill(-1);
  for (std::size_t i = 0; i < candidate.alphabet.size(); ++i) {
    char ch = candidate.alphabet[i];
    candidate.index_map[ch - 'A'] = static_cast<int>(i);
  }
  return candidate;
}

std::vector<AlphabetCandidate> build_alphabet_candidates(const std::vector<std::string> &words) {
  std::map<std::string, AlphabetCandidate> unique_map;
  for (const auto &word : words) {
    AlphabetCandidate forward = make_alphabet_candidate(word, false);
    unique_map.emplace(forward.alphabet, forward);
    AlphabetCandidate reversed = make_alphabet_candidate(word, true);
    unique_map.emplace(reversed.alphabet, reversed);
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

std::string decrypt_repeating(const std::string &cipher, const std::string &key,
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
    int k_idx = alphabet_index(alphabet, key[i % key_len]);
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

Candidate make_candidate(const std::string &key_word,
                         const AlphabetCandidate &alphabet, Mode mode,
                         bool autokey_variant, const std::string &plaintext,
                         std::size_t preview_length) {
  Candidate cand;
  cand.key = key_word;
  cand.mode = mode;
  cand.autokey = autokey_variant;
  cand.alphabet_word = alphabet.base_word;
  cand.alphabet_reversed = alphabet.reversed;
  cand.alphabet_string = alphabet.alphabet;
  cand.first2 = plaintext.substr(0, std::min<std::size_t>(2, plaintext.size()));
  cand.plaintext_preview = plaintext.substr(0, std::min(preview_length, plaintext.size()));
  cand.ioc = index_of_coincidence(plaintext);
  cand.chi = chi_square(plaintext);
  const double ioc_delta = std::abs(cand.ioc - kIocTarget);
  const double ioc_score = std::max(0.0, 1.0 - ioc_delta / 0.02);
  const double chi_clamped = std::min(400.0, cand.chi);
  const double chi_score = std::max(0.0, 1.0 - chi_clamped / 400.0);
  const double quality_factor = 0.1 + 0.9 * chi_score;
  cand.score = ioc_score * quality_factor;
  return cand;
}

void maintain_top_results(std::vector<Candidate> &results, const Candidate &candidate,
                          std::size_t max_results) {
  if (max_results == 0) {
    return;
  }
  if (results.size() < max_results) {
    results.push_back(candidate);
    return;
  }
  auto worst_it = std::min_element(
      results.begin(), results.end(),
      [](const Candidate &a, const Candidate &b) { return a.score < b.score; });
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
    } else if (arg == "--two-letter-list") {
      options.two_letter_list = read_file(require_value(arg));
    } else if (arg == "--two-letter-inline") {
      options.two_letter_list = require_value(arg);
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
    } else if (arg == "--help" || arg == "-h") {
      std::cout << "Quagmire III wordlist bruteforcer (C++)\n"
                << "Options:\n"
                << "  --ciphertext <text>             Inline ciphertext string\n"
                << "  --ciphertext-file <path>        File containing ciphertext\n"
                << "  --wordlist <path>               Main wordlist file (required)\n"
                << "  --wordlist-inline <text>        Inline wordlist string\n"
                << "  --two-letter-list <path>        Optional 2-letter filter list\n"
                << "  --two-letter-inline <text>      Inline 2-letter filter string\n"
                << "  --max-results <N>               Max candidates to keep (default 50)\n"
                << "  --preview-length <N>            Plaintext preview length (default 80)\n"
                << "  --threads <N>                   Worker threads (default hardware)\n"
                << "  --include-autokey               Try autokey variants too\n"
                << "  --progress-interval <sec>       Progress update interval (default 1.0)\n"
                << "  --quiet                         Suppress periodic progress output\n"
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
  if (options.threads == 0) {
    options.threads = 1;
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
      << std::setw(8) << "IoC" << "  " << std::setw(9) << "Chi^2" << "  "
      << std::left << std::setw(16) << "Cipher" << "  " << std::setw(8)
      << "Autokey" << "  " << std::setw(18) << "Key" << "  " << std::setw(15)
      << "Alphabet word" << "  " << std::setw(3) << "Rev" << "  "
      << "Plaintext" << '\n';
  oss << std::string(120, '-') << '\n';
  for (std::size_t i = 0; i < limit; ++i) {
    const Candidate &cand = *sorted[i];
    oss << std::right << std::setw(3) << (i + 1) << "  " << std::fixed
        << std::setprecision(3) << std::setw(7) << cand.score << "  "
        << std::setprecision(4) << std::setw(8) << cand.ioc << "  "
        << std::setprecision(2) << std::setw(9) << cand.chi << "  "
        << std::left << std::setw(16) << mode_family_name(cand.mode) << "  "
        << std::setw(8) << (cand.autokey ? "Yes" : "No") << "  "
        << std::setw(18) << cand.key << "  " << std::setw(15)
        << cand.alphabet_word << "  " << std::setw(3)
        << (cand.alphabet_reversed ? "Y" : "N") << "  "
        << cand.plaintext_preview << '\n';
  }
  return oss.str();
}

void print_table(const std::vector<Candidate> &candidates, std::size_t max_rows) {
  std::cout << format_results_table(candidates, max_rows);
}

struct WorkerResult {
  std::vector<Candidate> best;
  std::size_t combos = 0;
  std::size_t autokey_attempts = 0;
  std::size_t keys_processed = 0;
};

WorkerResult process_keys(const std::vector<std::string> &keys, std::size_t begin,
                          std::size_t end, const std::string &cipher,
                          const std::vector<AlphabetCandidate> &alphabets,
                          const std::vector<Mode> &modes,
                          const std::unordered_set<std::string> &two_letter_set,
                          const std::unordered_set<std::string> &four_letter_set,
                          bool have_first2_filter, bool have_second4_filter,
                          std::size_t max_results, std::size_t preview_length,
                          bool include_autokey, std::atomic<std::size_t> &combos_counter,
                          std::atomic<std::size_t> &autokey_counter,
                          std::atomic<std::size_t> &keys_counter) {
  WorkerResult result;
  result.best.reserve(max_results);
  for (std::size_t idx = begin; idx < end; ++idx) {
    const std::string &key_word = keys[idx];
    for (const auto &alphabet : alphabets) {
      for (Mode mode : modes) {
        std::string plaintext =
            decrypt_repeating(cipher, key_word, alphabet, mode);
        ++result.combos;
        ++combos_counter;
        if (plaintext.size() < 2) {
          continue;
        }
    std::string first2 = plaintext.substr(0, 2);
    bool apply_second4 = have_second4_filter && plaintext.size() >= 6;
    std::string second4 = apply_second4 ? plaintext.substr(2, 4) : std::string();
    if (have_first2_filter && two_letter_set.find(first2) == two_letter_set.end()) {
      continue;
    }
    if (apply_second4 && four_letter_set.find(second4) == four_letter_set.end()) {
      continue;
    }
        Candidate cand = make_candidate(key_word, alphabet, mode, false,
                                        plaintext, preview_length);
        maintain_top_results(result.best, cand, max_results);

        if (include_autokey) {
          std::string plaintext_auto =
              decrypt_autokey(cipher, key_word, alphabet, mode);
          ++result.autokey_attempts;
          ++autokey_counter;
          if (plaintext_auto.size() < 2) {
            continue;
          }
          std::string first2_auto = plaintext_auto.substr(0, 2);
          bool apply_second4_auto = have_second4_filter && plaintext_auto.size() >= 6;
          std::string second4_auto =
              apply_second4_auto ? plaintext_auto.substr(2, 4) : std::string();
          if (have_first2_filter &&
              two_letter_set.find(first2_auto) == two_letter_set.end()) {
            continue;
          }
          if (apply_second4_auto &&
              four_letter_set.find(second4_auto) == four_letter_set.end()) {
            continue;
          }
          Candidate cand_auto = make_candidate(key_word, alphabet, mode, true,
                                               plaintext_auto, preview_length);
          maintain_top_results(result.best, cand_auto, max_results);
        }
      }
    }
    ++result.keys_processed;
    ++keys_counter;
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
    oss << "\033[2J\033[H";
    oss << std::fixed << std::setprecision(1) << "Elapsed: " << elapsed
        << "s  Keys: " << keys << "  Combos: " << combos;
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

    const std::string cipher = clean_letters(options.ciphertext);
    const std::vector<std::string> words = parse_wordlist(options.wordlist);
    if (words.empty()) {
      throw std::runtime_error("Wordlist is empty after cleaning");
    }
    const std::unordered_set<std::string> two_letter_set =
        parse_two_letter_list(options.two_letter_list);
    const std::unordered_set<std::string> four_letter_set =
        build_four_letter_set(words);
    const bool have_first2_filter = !two_letter_set.empty();
    const bool have_second4_filter = !four_letter_set.empty();

    std::vector<AlphabetCandidate> alphabets = build_alphabet_candidates(words);
    if (alphabets.empty()) {
      throw std::runtime_error("No alphabet candidates generated from wordlist");
    }

    std::vector<Mode> modes = {Mode::kVigenere, Mode::kBeaufort,
                               Mode::kVariantBeaufort};

    const std::size_t total_combos = words.size() * alphabets.size() * modes.size();

    std::cout << "Cipher length: " << cipher.size() << '\n';
    std::cout << "Key candidates: " << words.size() << '\n';
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

    std::vector<Candidate> global_results;
    global_results.reserve(results_limit);
    std::mutex results_mutex;

    std::vector<std::thread> workers;
    const std::size_t thread_count = std::max<std::size_t>(1, options.threads);
    const std::size_t base_chunk = words.size() / thread_count;
    const std::size_t remainder = words.size() % thread_count;

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
            words, begin, end, cipher, alphabets, modes, two_letter_set,
            four_letter_set, have_first2_filter, have_second4_filter,
            results_limit, options.preview_length, options.include_autokey,
            combos_counter, autokey_counter, keys_counter);
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
    std::cout << "Keys processed: " << keys_counter.load() << '/' << words.size()
              << '\n' << std::endl;

    print_table(global_results, results_limit);
    return 0;
  } catch (const std::exception &ex) {
    std::cerr << "Error: " << ex.what() << std::endl;
    return 1;
  }
}


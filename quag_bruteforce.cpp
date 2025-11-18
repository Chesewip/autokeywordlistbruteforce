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
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <numeric>

#include "Source/gpu_quag.h"
#include "Source/WordListParser.h"
#include "Source/AlphabetBuilder.h"
#include "Source/Options.h"
#include "Source/CPUDecode.h"
#include "Source/GPUDecode.h"

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace {





struct PhraseNode
{
    std::vector<std::uint32_t> word_ids;  // indices into keys_slice
    std::uint32_t alphabet_id{ 0 };
    std::uint8_t  mode{ 0 };             // Mode enum value
    std::uint16_t key_len{ 0 };          // total key length in letters
    double        prefix_score{ 0.0 };   // e.g., IoC or any heuristic
};


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

std::string format_results_table(const std::vector<Candidate> &candidates, std::size_t max_rows) 
{
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

void print_table(const std::vector<Candidate> &candidates, std::size_t max_rows) 
{
  std::cout << format_results_table(candidates, max_rows);
}

void write_results_to_file(const std::vector<Candidate> &candidates,
                           std::size_t max_rows,
                           const std::string &path) 
{
  std::ofstream output(path);
  if (!output) {
    throw std::runtime_error("Failed to open results output file: " + path);
  }
  output << format_results_table(candidates, max_rows);
}

static bool validate_covered_words(
    const std::string& plaintext,
    const std::vector<int>& spacing_pattern,
    const std::unordered_map<int, std::unordered_set<std::string>>& words_by_length,
    int key_len)
{
    if (plaintext.empty() || key_len <= 0)
        return false;

    int offset = 0;
    const int max_len = std::min<int>((int)plaintext.size(), key_len);
    bool saw_any_word = false;

    for (int len : spacing_pattern)
    {
        int end = offset + len;
        if (end > key_len || end > max_len)
            break; // this word is not fully covered yet

        if (end > (int)plaintext.size())
            return false;

        auto it = words_by_length.find(len);
        if (it == words_by_length.end())
            return false;

        std::string word = plaintext.substr(offset, len);
        if (it->second.find(word) == it->second.end())
            return false; // covered word is not in dictionary

        saw_any_word = true;
        offset = end;
    }

    // We at least want one fully covered, valid word for this to be interesting
    return saw_any_word;
}

void progress_loop(const std::atomic<bool> &done, double interval_seconds,
                   std::atomic<std::size_t> &combos_counter,
                   std::atomic<std::size_t> &keys_counter,
                   std::atomic<std::size_t> &autokey_counter,
                   std::size_t total_combos, std::mutex &results_mutex,
                   std::vector<Candidate> &results, std::size_t max_rows) 
{
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

int main(int argc, char* argv[]) {
    try {
        Options options = Options::parse_options(argc, argv);

#ifdef __AVX2__
        std::cout << "__AVX2__ is defined\n";
#else
        std::cout << "__AVX2__ is NOT defined\n";
#endif

        const std::string cipher = WordlistParser::clean_letters(options.ciphertext);

        // --- spacing words (unchanged) ---
        std::vector<std::string> spacing_words;
        if (!options.spacing_wordlist.empty()) {
            spacing_words = WordlistParser::parse_wordlist(options.spacing_wordlist);
            if (spacing_words.empty()) {
                throw std::runtime_error(
                    "Spacing wordlist is empty after cleaning");
            }
        }

        // --- spacing pattern / guide (unchanged semantics) ---
        std::vector<int> spacing_pattern;
        std::unordered_map<int, std::unordered_set<std::string>>
            spacing_words_by_length;
        if (!options.spacing_guide.empty()) {
            spacing_pattern = WordlistParser::parse_spacing_pattern(options.spacing_guide);
            if (spacing_pattern.empty()) {
                throw std::runtime_error(
                    "Spacing guide did not contain any valid word lengths");
            }
            if (spacing_words.empty()) {
                throw std::runtime_error(
                    "Spacing guide requires a spacing wordlist");
            }
            spacing_words_by_length = WordlistParser::build_words_by_length(spacing_words);
        }

        const bool spacing_scoring_enabled =
            !spacing_pattern.empty() && !spacing_words_by_length.empty();

        // --- KEY WORDS: now we add interrupted variants here ---
        std::vector<std::string> key_words =
            WordlistParser::parse_wordlist(options.wordlist,
                spacing_pattern.empty() ? nullptr : &spacing_pattern);

        if (key_words.empty()) {
            throw std::runtime_error("Wordlist is empty after cleaning");
        }

        // --- alphabet words stay as plain words ---
        const std::vector<std::string> alphabet_words =
            WordlistParser::parse_wordlist(options.alphabet_wordlist);
        if (alphabet_words.empty()) {
            throw std::runtime_error(
                "Alphabet wordlist is empty after cleaning");
        }

        const std::unordered_set<std::uint16_t> two_letter_set =
            WordlistParser::parse_two_letter_list(options.two_letter_list);
        const std::unordered_set<std::uint32_t> four_letter_set = !spacing_words.empty()
            ? WordlistParser::build_four_letter_set(spacing_words)
            : WordlistParser::build_four_letter_set(key_words);

        const bool have_first2_filter = !two_letter_set.empty();
        const bool have_second4_filter = !four_letter_set.empty();

        std::vector<AlphabetCandidate> alphabets = AlphabetBuilder::build_alphabet_candidates(alphabet_words, options);
        if (alphabets.empty()) 
        {
            throw std::runtime_error("No alphabet candidates generated from wordlist");
        }

        std::vector<Mode> modes = { Mode::kVigenere, Mode::kBeaufort,
                                   Mode::kVariantBeaufort };

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
    std::cout << "CUDA execution: " << (options.use_cuda ? "enabled" : "disabled")
              << '\n';

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
    const std::size_t thread_count = options.use_cuda ? 1 : std::max<std::size_t>(1, options.threads);
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
    for (std::size_t t = 0; t < thread_count; ++t) 
    {
      std::size_t chunk = base_chunk + (t < remainder ? 1 : 0);
      std::size_t begin = current;
      std::size_t end = begin + chunk;
      current = end;
      if (begin >= end) 
      {
        continue;
      }

      if (options.use_cuda)
      {
          workers.emplace_back([&, begin, end]()
              {
                  WorkerResult worker_result = GPUDecode::process_keys(
                      key_words,
                      begin,
                      end,
                      cipher,
                      alphabets,
                      modes,
                      two_letter_set,
                      four_letter_set,
                      have_first2_filter,
                      have_second4_filter,
                      spacing_pattern_ptr,
                      spacing_words_ptr,
                      results_limit,
                      options.preview_length,
                      options.include_autokey,
                      options.use_cuda,
                      combos_counter,
                      autokey_counter,
                      keys_counter,
                      results_mutex,
                      global_results
                  );

                  std::lock_guard<std::mutex> lock(results_mutex);
                  for (const auto& cand : worker_result.best)
                  {
                      CPUDecode::maintain_top_results(global_results, cand, results_limit);
                  }
              });
      }
      else
      {
          workers.emplace_back([&, begin, end]()
              {
                  WorkerResult worker_result = CPUDecode::process_keys_CPU(
                      key_words,
                      begin,
                      end,
                      cipher,
                      alphabets,
                      modes,
                      two_letter_set,
                      four_letter_set,
                      have_first2_filter,
                      have_second4_filter,
                      spacing_pattern_ptr,
                      spacing_words_ptr,
                      results_limit,
                      options.preview_length,
                      options.include_autokey,
                      options.use_cuda,
                      combos_counter,
                      autokey_counter,
                      keys_counter,
                      results_mutex,
                      global_results
                  );

                  std::lock_guard<std::mutex> lock(results_mutex);
                  for (const auto& cand : worker_result.best)
                  {
                      CPUDecode::maintain_top_results(global_results, cand, results_limit);
                  }
              });
      }
    }

    for (auto &thread : workers) 
    {
      if (thread.joinable()) 
      {
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


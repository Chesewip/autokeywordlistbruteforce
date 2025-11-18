#pragma once
#include "AlphabetBuilder.h"
#include "WordListParser.h"
#include "gpu_quag.h"
#include <mutex>

#if defined(__AVX2__)
#include <immintrin.h>
#endif


namespace English 
{
    constexpr double kIocTarget = 0.066;

    constexpr std::array<double, 26> kEnglishFreq = 
    {
        0.08167, 0.01492, 0.02782, 0.04253, 0.12702, 0.02228, 0.02015,
        0.06094, 0.06966, 0.00153, 0.00772, 0.04025, 0.02406, 0.06749,
        0.07507, 0.01929, 0.00095, 0.05987, 0.06327, 0.09056, 0.02758,
        0.00978, 0.02360, 0.00150, 0.01974, 0.00074 
    };
};

enum class Mode
{
    kVigenere,
    kBeaufort,
    kVariantBeaufort,
};

#if defined(__AVX2__)
struct alignas(32) KeyBlock 
{
    std::array<std::uint8_t, 32> data{};
};
#endif

struct Candidate
{

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

    static Candidate make_candidate(
        const std::string& key_word,
        const AlphabetCandidate& alphabet, Mode mode,
        bool autokey_variant,
        const std::string& plaintext,
        std::size_t preview_length,
        const std::vector<int>* spacing_pattern,
        const std::unordered_map<int, std::unordered_set<std::string>>* spacing_words_by_length,
        double ioc,
        double chi2)
    {

        Candidate cand;
        cand.key = key_word;
        cand.mode = mode;
        cand.autokey = autokey_variant;
        cand.alphabet_word = alphabet.base_word;
        cand.alphabet_keyword_reversed = alphabet.keyword_reversed;
        cand.alphabet_base_reversed = alphabet.alphabet_reversed;
        cand.alphabet_keyword_front = alphabet.keyword_front;
        cand.alphabet_string = alphabet.alphabet;
        cand.ioc = ioc;
        cand.chi = chi2;
        cand.plaintext_preview = plaintext.substr(0, std::min(preview_length, plaintext.size()));

        
        const double ioc_delta = std::abs(cand.ioc - English::kIocTarget);
        const double ioc_score = std::max(0.0, 1.0 - ioc_delta / 0.02);
        const double chi_clamped = std::min(400.0, cand.chi);
        const double chi_score = std::max(0.0, 1.0 - chi_clamped / 400.0);
        const double quality_factor = 0.1 + 0.9 * chi_score;
        const double stats_score = ioc_score * quality_factor;
        const double word_weight = 0.8;
        const double stats_weight = 1.0 - word_weight;
        cand.score = stats_score;

        auto result = WordlistParser::count_spacing_matches(plaintext, *spacing_pattern, *spacing_words_by_length);
        cand.spacing_matches = result.first;
        cand.spacing_total = result.second;

        if (cand.spacing_matches >= 0 && cand.spacing_total > 0)
        {
            double word_score = static_cast<double>(cand.spacing_matches) /
                static_cast<double>(cand.spacing_total);
            cand.score = word_weight * word_score + stats_weight * stats_score;
        }

        return cand;
    }

    static bool same_candidate_identity(const Candidate& a, const Candidate& b)
    {
        return a.key == b.key
            && a.mode == b.mode
            && a.autokey == b.autokey
            && a.alphabet_string == b.alphabet_string;
    }

};


struct WorkerResult
{
    std::vector<Candidate> best;
    std::size_t combos = 0;
    std::size_t autokey_attempts = 0;
    std::size_t keys_processed = 0;
};

struct CPUDecode
{


    static std::string build_plaintext_string(const AlphabetCandidate& alphabet, const std::uint8_t* plaintext_indices, std::size_t text_len)
    {
        std::string result;
        result.resize(text_len);
        const std::string& alph = alphabet.alphabet;

        for (std::size_t i = 0; i < text_len; ++i) {
            std::uint8_t idx = plaintext_indices[i];
            // assume idx < 26 for valid positions
            result[i] = alph[idx];
        }
        return result;
    }

    static inline std::uint8_t decrypt_symbol(std::uint8_t cipher_idx, std::uint8_t key_idx, Mode mode) \
    {
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

    static void decrypt_repeating(
        const AlphabetCandidate& alphabet,
        Mode mode,
        const std::uint8_t* cipher_indices,
        const std::uint8_t* letter_mask,
        const std::uint8_t* key_indices,
        const std::uint8_t* key_valid,
        std::size_t key_len,
        std::uint8_t* plaintext_indices,
        std::size_t text_len,
        std::size_t padded_len
#if defined(__AVX2__)
        , const KeyBlock* key_blocks
#endif
    )
    {
        (void)alphabet;
        if (padded_len == 0 || key_len == 0 || plaintext_indices == nullptr) {
            return;
        }

#if defined(__AVX2__)
        constexpr std::size_t kVecWidth = 32;
        const std::size_t vec_blocks = padded_len / kVecWidth;

        const __m256i zero = _mm256_setzero_si256();
        const __m256i twenty_six = _mm256_set1_epi8(26);
        const __m256i twenty_five = _mm256_set1_epi8(25);

        std::size_t vec_index = 0;
        std::size_t block_offset = 0;

        switch (mode)
        {
        case Mode::kVigenere:
            for (std::size_t b = 0; b < vec_blocks; ++b) {
                __m256i cipher_vec = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(cipher_indices + vec_index));
                __m256i key_vec = _mm256_load_si256(
                    reinterpret_cast<const __m256i*>(key_blocks[block_offset].data.data()));

                __m256i plain_vec = _mm256_sub_epi8(cipher_vec, key_vec);
                __m256i mask = _mm256_cmpgt_epi8(zero, plain_vec);
                plain_vec = _mm256_add_epi8(
                    plain_vec, _mm256_and_si256(mask, twenty_six));

                _mm256_storeu_si256(
                    reinterpret_cast<__m256i*>(plaintext_indices + vec_index),
                    plain_vec);

                vec_index += kVecWidth;
                block_offset = (block_offset + kVecWidth) % key_len;
            }
            break;

        case Mode::kBeaufort:
            for (std::size_t b = 0; b < vec_blocks; ++b) {
                __m256i cipher_vec = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(cipher_indices + vec_index));
                __m256i key_vec = _mm256_load_si256(
                    reinterpret_cast<const __m256i*>(key_blocks[block_offset].data.data()));

                __m256i plain_vec = _mm256_sub_epi8(key_vec, cipher_vec);
                __m256i mask = _mm256_cmpgt_epi8(zero, plain_vec);
                plain_vec = _mm256_add_epi8(
                    plain_vec, _mm256_and_si256(mask, twenty_six));

                _mm256_storeu_si256(
                    reinterpret_cast<__m256i*>(plaintext_indices + vec_index),
                    plain_vec);

                vec_index += kVecWidth;
                block_offset = (block_offset + kVecWidth) % key_len;
            }
            break;

        case Mode::kVariantBeaufort:
            for (std::size_t b = 0; b < vec_blocks; ++b) {
                __m256i cipher_vec = _mm256_loadu_si256(
                    reinterpret_cast<const __m256i*>(cipher_indices + vec_index));
                __m256i key_vec = _mm256_load_si256(
                    reinterpret_cast<const __m256i*>(key_blocks[block_offset].data.data()));

                __m256i plain_vec = _mm256_add_epi8(cipher_vec, key_vec);
                __m256i mask = _mm256_cmpgt_epi8(plain_vec, twenty_five);
                plain_vec = _mm256_sub_epi8(
                    plain_vec, _mm256_and_si256(mask, twenty_six));

                _mm256_storeu_si256(
                    reinterpret_cast<__m256i*>(plaintext_indices + vec_index),
                    plain_vec);

                vec_index += kVecWidth;
                block_offset = (block_offset + kVecWidth) % key_len;
            }
            break;
        }
#else
        for (std::size_t i = 0; i < text_len; ++i) {
            if (!letter_mask[i] || !key_valid[i % key_len]) {
                plaintext_indices[i] = 0;
                continue;
            }
            plaintext_indices[i] =
                decrypt_symbol(cipher_indices[i], key_indices[i % key_len], mode);
        }
#endif

    }

    static void decrypt_autokey(
        const std::uint8_t* cipher_indices,
        std::size_t cipher_len,
        const std::uint8_t* key_indices,
        std::size_t key_len,
        std::uint8_t* plaintext_indices,
        Mode mode)
    {
        if (cipher_len == 0 || key_len == 0 || plaintext_indices == nullptr) {
            return;
        }

        for (std::size_t i = 0; i < cipher_len; ++i) {
            std::uint8_t c_idx = cipher_indices[i];
            std::uint8_t k_idx = (i < key_len)
                ? key_indices[i]
                : plaintext_indices[i - key_len];
            plaintext_indices[i] = decrypt_symbol(c_idx, k_idx, mode);
        }
    }

    static void compute_stats_indices(const std::uint8_t* plain,
        std::size_t n,
        const AlphabetCandidate& alphabet,
        double& out_ioc,
        double& out_chi)
    {
        if (n <= 1) {
            out_ioc = 0.0;
            out_chi = std::numeric_limits<double>::infinity();
            return;
        }

        const std::string& alph = alphabet.alphabet; // keyed alphabet

        std::array<std::size_t, 26> counts{};
        for (std::size_t i = 0; i < n; ++i) {
            std::uint8_t idx = plain[i];
            if (idx < 26) {
                char ch = alph[idx];  // map keyed index  actual letter
                if (ch >= 'A' && ch <= 'Z') {
                    counts[static_cast<std::size_t>(ch - 'A')]++;
                }
            }
        }

        double numerator = 0.0;
        double chi = 0.0;

        for (std::size_t i = 0; i < 26; ++i) {
            std::size_t c = counts[i];
            numerator += static_cast<double>(c) * static_cast<double>(c - 1);

            double expected = static_cast<double>(n) * English::kEnglishFreq[i];
            if (expected > 0.0) {
                double diff = static_cast<double>(c) - expected;
                chi += (diff * diff) / expected;
            }
        }

        out_ioc = numerator / (static_cast<double>(n) * static_cast<double>(n - 1));
        out_chi = chi;
    }


    static bool passes_front_filters_repeating(
        const std::string& cipher,
        const std::string& key,
        const AlphabetCandidate& alphabet,
        Mode mode,
        bool have_first2_filter,
        bool have_second4_filter,
        const std::unordered_set<std::uint16_t>& two_letter_set,
        const std::unordered_set<std::uint32_t>& four_letter_set,
        std::unordered_set<std::uint16_t>& trash_letter_set)
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

            int c_idx = AlphabetBuilder::alphabet_index(alphabet, c);
            if (c_idx >= 0) {
                char kch = key[i % key_len];
                int k_idx = AlphabetBuilder::alphabet_index(alphabet, kch);
                if (k_idx >= 0) {
                    std::uint8_t p_idx = CPUDecode::decrypt_symbol(
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
            char a = buf[0];
            char b = buf[1];
            if (a < 'A' || a > 'Z' || b < 'A' || b > 'Z') {
                // definitely not in two_letter_codes
                return false;
            }
            std::uint16_t code = WordlistParser::encode_bigram(a, b);
            if (two_letter_set.find(code) == two_letter_set.end())
            {
                trash_letter_set.insert(code);
                return false;
            }
        }

        if (have_second4_filter && produced >= 6) {
            char a = buf[2];
            char b = buf[3];
            char c = buf[4];
            char d = buf[5];

            if (a < 'A' || a > 'Z' ||
                b < 'A' || b > 'Z' ||
                c < 'A' || c > 'Z' ||
                d < 'A' || d > 'Z') {
                return false;
            }

            std::uint32_t code4 = WordlistParser::encode_quad(a, b, c, d);
            if (four_letter_set.find(code4) == four_letter_set.end()) {
                return false;
            }
        }

        return true;
    }

    static void maintain_top_results(std::vector<Candidate>& results, const Candidate& candidate, std::size_t max_results)
    {
        if (max_results == 0) {
            return;
        }

        // 1. Check if we already have this identity
        auto existing = std::find_if(results.begin(), results.end(),
            [&](const Candidate& c) {
                return Candidate::same_candidate_identity(c, candidate);
            });

        if (existing != results.end()) {
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

    static WorkerResult process_keys_CPU(
        const std::vector<std::string>& keys,
        std::size_t begin,
        std::size_t end,
        const std::string& cipher,
        const std::vector<AlphabetCandidate>& alphabets,
        const std::vector<Mode>& modes,
        const std::unordered_set<std::uint16_t>& two_letter_set,
        const std::unordered_set<std::uint32_t>& four_letter_set,
        bool have_first2_filter,
        bool have_second4_filter,
        const std::vector<int>* spacing_pattern,
        const std::unordered_map<int, std::unordered_set<std::string>>* spacing_words_by_length,
        std::size_t max_results,
        std::size_t preview_length,
        bool include_autokey,
        bool use_cuda,
        std::atomic<std::size_t>& combos_counter,
        std::atomic<std::size_t>& autokey_counter,
        std::atomic<std::size_t>& keys_counter,
        std::mutex& results_mutex,
        std::vector<Candidate>& global_results)
    {
        WorkerResult result;
        result.best.reserve(max_results);

        const std::size_t alphabet_count = alphabets.size();
        const std::size_t text_len = cipher.size();    // e.g. 85
        constexpr std::size_t kVecWidth = 32;
        const std::size_t padded_len =
            ((text_len + kVecWidth - 1) / kVecWidth) * kVecWidth; // e.g. 96

        // per-worker scratch buffers (padded)
        std::unique_ptr<std::uint8_t[]> letter_mask(new std::uint8_t[padded_len]());
        std::unique_ptr<std::uint8_t[]> cipher_indices(new std::uint8_t[padded_len]());
        std::unique_ptr<std::uint8_t[]> plaintext_indices(new std::uint8_t[padded_len]());

        std::unique_ptr<std::uint8_t[]> key_indices;
        std::size_t key_indices_capacity = 0;
        std::unique_ptr<std::uint8_t[]> key_valid;
        std::size_t key_valid_capacity = 0;
#if defined(__AVX2__)
        std::unique_ptr<KeyBlock[]> key_blocks;
        std::size_t key_block_capacity = 0;
#endif
        std::unique_ptr<std::uint8_t[]> autokey_plaintext;
        std::size_t autokey_capacity = 0;

        auto ensure_u8_capacity = [](std::unique_ptr<std::uint8_t[]>& buffer, std::size_t& capacity, std::size_t required) {
            if (capacity < required) {
                std::unique_ptr<std::uint8_t[]> new_buffer(new std::uint8_t[required]);
                buffer.swap(new_buffer);
                capacity = required;
            }
        };
#if defined(__AVX2__)
        auto ensure_keyblock_capacity = [](std::unique_ptr<KeyBlock[]>& buffer, std::size_t& capacity, std::size_t required) {
            if (capacity < required) {
                std::unique_ptr<KeyBlock[]> new_buffer(new KeyBlock[required]);
                buffer.swap(new_buffer);
                capacity = required;
            }
        };
#endif

        // build letter_mask once (A-Z only)
        std::fill(letter_mask.get(), letter_mask.get() + padded_len, 0);
        for (std::size_t i = 0; i < text_len; ++i) {
            char c = cipher[i];
            letter_mask[i] = (c >= 'A' && c <= 'Z') ? 1u : 0u;
        }

        // ---- CPU PATH (unchanged) ----
        {
            // Outer: alphabets
            for (std::size_t a = 0; a < alphabet_count; ++a) {
                const auto& alphabet = alphabets[a];

                // build cipher_indices for this alphabet once
                for (std::size_t i = 0; i < text_len; ++i) {
                    if (!letter_mask[i]) {
                        cipher_indices[i] = 0;
                        continue;
                    }
                    int idx = AlphabetBuilder::alphabet_index(alphabet, cipher[i]);
                    cipher_indices[i] = (idx >= 0)
                        ? static_cast<std::uint8_t>(idx)
                        : 0;
                }
                std::fill(cipher_indices.get() + text_len, cipher_indices.get() + padded_len, 0);

                std::unordered_map<Mode, std::unordered_set<std::uint16_t>> trashBigramMap;

                // Inner: keys assigned to this worker
                for (std::size_t idx_k = begin; idx_k < end; ++idx_k) {
                    const std::string& key_word = keys[idx_k];

                    const std::size_t key_len = key_word.size();
                    ensure_u8_capacity(key_indices, key_indices_capacity, key_len);
                    ensure_u8_capacity(key_valid, key_valid_capacity, key_len);
                    for (std::size_t i = 0; i < key_len; ++i)
                    {
                        int idx = AlphabetBuilder::alphabet_index(alphabet, key_word[i]);
                        if (idx < 0) {
                            key_valid[i] = 0;
                            key_indices[i] = 0;
                        }
                        else {
                            key_valid[i] = 1;
                            key_indices[i] = static_cast<std::uint8_t>(idx);
                        }
                    }

                    if (key_len == 0) {
                        continue;
                    }

                    std::uint16_t key_bigram_code = 0;
                    bool has_bigram = key_len >= 2;
                    if (has_bigram) {
                        char a0 = key_word[0];
                        char b0 = key_word[1];
                        key_bigram_code = WordlistParser::encode_bigram(a0, b0);
                    }

                    bool keyblockBuilt = false;

                    for (Mode mode : modes)
                    {
                        ++result.combos;
                        ++combos_counter;

                        auto& trashSet = trashBigramMap[mode];

                        if (has_bigram && trashSet.find(key_bigram_code) != trashSet.end()) {
                            continue;
                        }

                        if ((have_first2_filter || have_second4_filter) &&
                            !CPUDecode::passes_front_filters_repeating(
                                cipher,
                                key_word,
                                alphabet,
                                mode,
                                have_first2_filter,
                                have_second4_filter,
                                two_letter_set,
                                four_letter_set,
                                trashSet))
                        {
                            continue; // reject this (key, alphabet, mode) without full decrypt
                        }

#if defined(__AVX2__)
                        if (!keyblockBuilt)
                        {
                            ensure_keyblock_capacity(key_blocks, key_block_capacity, key_len);
                            for (std::size_t start = 0; start < key_len; ++start) {
                                auto& block = key_blocks[start];
                                for (std::size_t j = 0; j < block.data.size(); ++j) {
                                    block.data[j] = key_indices[(start + j) % key_len];
                                }
                            }
                            keyblockBuilt = true;
                        }
#endif

                        CPUDecode::decrypt_repeating(
                            alphabet,
                            mode,
                            cipher_indices.get(),
                            letter_mask.get(),
                            key_indices.get(),
                            key_valid.get(),
                            key_len,
                            plaintext_indices.get(),
                            text_len,
                            padded_len
#if defined(__AVX2__)
                            , key_blocks.get()
#endif
                        );

                        double ioc, chi;
                        CPUDecode::compute_stats_indices(plaintext_indices.get(), text_len, alphabet, ioc, chi);

                        if (ioc > 0.05 && chi < 160.0)
                        {
                            std::string plaintext = CPUDecode::build_plaintext_string(alphabet, plaintext_indices.get(), text_len);

                            Candidate cand = Candidate::make_candidate(
                                key_word, alphabet, mode, false,
                                plaintext, preview_length,
                                spacing_pattern, spacing_words_by_length, ioc, chi);
                            CPUDecode::maintain_top_results(result.best, cand, 10);
                        }

                        if (include_autokey)
                        {
                            ++result.autokey_attempts;
                            ++autokey_counter;

                            ensure_u8_capacity(autokey_plaintext, autokey_capacity, text_len);

                            CPUDecode::decrypt_autokey(
                                cipher_indices.get(),
                                text_len,
                                key_indices.get(),
                                key_len,
                                autokey_plaintext.get(),
                                mode);

                            double ioc_auto = 0.0, chi_auto = 0.0;

                            CPUDecode::compute_stats_indices(
                                autokey_plaintext.get(),
                                text_len,
                                alphabet,
                                ioc_auto,
                                chi_auto);

                            if (ioc_auto > 0.05 && chi_auto < 160.0)
                            {
                                std::string plaintext_auto = CPUDecode::build_plaintext_string(alphabet, autokey_plaintext.get(), text_len);

                                Candidate cand_auto = Candidate::make_candidate(
                                    key_word,
                                    alphabet,
                                    mode,
                                    /*autokey_variant=*/true,
                                    plaintext_auto,
                                    preview_length,
                                    spacing_pattern,
                                    spacing_words_by_length,
                                    ioc_auto,
                                    chi_auto);

                                CPUDecode::maintain_top_results(result.best, cand_auto, 10);
                            }
                        }
                    }
                }

                // count each key once (not once per alphabet)
                ++result.keys_processed;
                ++keys_counter;

                // merge local best into global periodically
                if (a % 1000 == 0)
                {
                    std::lock_guard<std::mutex> lock(results_mutex);
                    for (const auto& cand : result.best) {
                        CPUDecode::maintain_top_results(global_results, cand, max_results);
                    }
                }
            }

            return result;
        }
    }

    // CPU-based equivalent of launch_q3_gpu_megabatch, specialized for
// a *single* (alphabet, mode, tile) used in the phrase builder path.
    static inline GpuBatchResult launch_q3_cpu_phrase_megabatch(
        const DeviceAlphabetTile& tile,
        const AlphabetCandidate& alphabet,
        std::uint32_t alphabet_id,
        Mode mode,
        const PackedKeysHostLetters& packed,
        float IOC_GATE,
        float CHI_GATE,
        std::size_t full_cap,
        std::size_t front_cap)
    {
        GpuBatchResult out;
        out.full_hits.reserve(full_cap);
        out.front_hits.reserve(front_cap);

        const std::size_t num_keys = packed.num_keys();
        if (num_keys == 0)
            return out;

        const std::size_t text_len = static_cast<std::size_t>(tile.text_len);
        if (text_len == 0)
            return out;

        std::vector<std::uint8_t> key_indices;
        key_indices.reserve(64);

        // Compact plaintext: only positions where tile.mask[i] != 0
        std::vector<std::uint8_t> plain_compact;
        plain_compact.reserve(text_len);

        constexpr std::uint8_t HIT_FRONT_OK = 0x01;
        constexpr std::uint8_t HIT_STATS_OK = 0x02;
        constexpr int PREVIEW_MAX = 80;

        const char* alph_str = alphabet.alphabet.c_str();

        for (std::size_t k = 0; k < num_keys; ++k)
        {
            const std::uint32_t key_off = packed.key_offsets[k];
            const std::size_t   key_len = packed.key_lengths[k];
            if (key_len == 0)
                continue;

            // --------------------------------------------------------
            // Build key_indices in keyed space from packed key chars
            // --------------------------------------------------------
            key_indices.resize(key_len);
            bool key_ok = true;
            for (std::size_t i = 0; i < key_len; ++i)
            {
                char ch = packed.key_chars_flat[key_off + i];
                int idx = AlphabetBuilder::alphabet_index(alphabet, ch);
                if (idx < 0)
                {
                    key_ok = false;
                    break;
                }
                key_indices[i] = static_cast<std::uint8_t>(idx);
            }
            if (!key_ok)
                continue;

            // --------------------------------------------------------
            // Decrypt into compact plaintext (logical positions only)
            // --------------------------------------------------------
            plain_compact.clear();
            plain_compact.reserve(text_len);

            std::size_t logical_pos = 0;
            for (std::size_t i = 0; i < text_len; ++i)
            {
                if (!tile.mask[i])
                    continue;

                const std::uint8_t c_idx = tile.cipher_idx[i];
                const std::uint8_t k_idx = key_indices[logical_pos % key_len];

                const std::uint8_t p_idx =
                    CPUDecode::decrypt_symbol(c_idx, k_idx, mode);

                plain_compact.push_back(p_idx);
                ++logical_pos;
            }

            const std::size_t n = plain_compact.size();
            if (n == 0)
                continue;

            // --------------------------------------------------------
            // Compute stats on compact plaintext
            // --------------------------------------------------------
            double ioc = 0.0;
            double chi = 0.0;
            CPUDecode::compute_stats_indices(
                plain_compact.data(), n, alphabet, ioc, chi);

            // --------------------------------------------------------
            // Build HitRecord (ALWAYS goes to front_hits, so autokey runs)
            // --------------------------------------------------------
            HitRecord h{};
            h.key_id = static_cast<std::uint32_t>(k);
            h.alphabet_id = alphabet_id;
            h.mode = static_cast<std::uint8_t>(mode);
            h.flags = HIT_FRONT_OK;         // front hit by definition here
            h.ioc = static_cast<float>(ioc);
            h.chi = static_cast<float>(chi);

            const std::size_t preview_len =
                std::min<std::size_t>(n, PREVIEW_MAX);
            h.preview_len = static_cast<std::uint8_t>(preview_len);

            // preview: canonical [0..25] representing A..Z
            for (std::size_t i = 0; i < preview_len; ++i)
            {
                const std::uint8_t p_idx = plain_compact[i];
                char ch = alph_str[p_idx]; // keyed -> canonical

                if (ch >= 'A' && ch <= 'Z')
                    h.preview[i] = static_cast<std::uint8_t>(ch - 'A');
                else
                    h.preview[i] = 0u;
            }

            // Push to front_hits unconditionally (until cap) so process_hits_for_keyset
            // can run static+autokey checks on it.
            if (out.front_hits.size() < front_cap)
                out.front_hits.push_back(h);

            // If the static stats are strong, also mark as full hit
            if (ioc >= static_cast<double>(IOC_GATE) &&
                chi <= static_cast<double>(CHI_GATE) &&
                out.full_hits.size() < full_cap)
            {
                h.flags |= HIT_STATS_OK;
                out.full_hits.push_back(h);
            }
        }

        return out;
    }



};
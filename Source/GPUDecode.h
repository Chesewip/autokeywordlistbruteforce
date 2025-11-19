#pragma once

#include <numeric>

#include "gpu_quag.h"
#include "CPUDecode.h"
#include "PhraseBuilder.h"
#include "ScopedTimer.h"

struct GPUDecode
{
    struct PhraseNode
    {
        std::vector<std::uint32_t> word_ids; // indices into keys_slice
        std::uint32_t alphabet_id{ 0 };
        std::uint8_t  mode{ 0 };           // Mode enum value
        std::uint16_t key_len{ 0 };        // total key length in letters
        double        prefix_score{ 0.0 }; // heuristic (IoC etc.)
    };

    // ------------------------------
    // Small helpers used elsewhere
    // ------------------------------


    static inline std::once_flag g_gpu_filter_init_flag;

    static uint32_t encode_word_base26(const std::string& w)
    {
        uint32_t code = 0;
        for (char ch : w)
        {
            int idx = ch - 'A';
            if (idx < 0 || idx >= 26) return UINT32_MAX; // skip weird stuff
            code = code * 26u + static_cast<uint32_t>(idx);
        }
        return code;
    }

    static WordCodeTable build_word_code_table(
        const std::unordered_map<int, std::unordered_set<std::string>>& words_by_length)
    {
        WordCodeTable table;
        table.codes.clear();
        table.offsets.fill(0);

        int running = 0;
        table.offsets[0] = 0;
        table.offsets[1] = 0;

        // We care about lengths 2..6 (per your sentence)
        for (int len = 2; len <= 6; ++len)
        {
            table.offsets[len] = running;

            auto it = words_by_length.find(len);
            if (it == words_by_length.end())
                continue;

            std::vector<uint32_t> tmp;
            tmp.reserve(it->second.size());
            for (const auto& w : it->second)
            {
                uint32_t code = encode_word_base26(w);
                if (code != UINT32_MAX)
                    tmp.push_back(code);
            }

            std::sort(tmp.begin(), tmp.end());
            tmp.erase(std::unique(tmp.begin(), tmp.end()), tmp.end());

            running += static_cast<int>(tmp.size());
            table.codes.insert(table.codes.end(), tmp.begin(), tmp.end());
        }

        table.offsets[7] = running; // sentinel for len=6 range end
        return table;
    }

    static WordBitsetTable build_word_bitset_table(
        const std::unordered_map<int, std::unordered_set<std::string>>& words_by_length,
        int max_len = 6)
    {
        WordBitsetTable t;
        t.offsets.fill(0);

        // 1) Compute how many bits we need per length
        uint32_t bit_cursor = 0;
        for (int len = 2; len <= max_len; ++len)
        {
            t.offsets[len] = bit_cursor;

            auto it = words_by_length.find(len);
            if (it == words_by_length.end())
                continue;

            // full space is 26^len (but most bits will be 0)
            uint32_t space = 1;
            for (int i = 0; i < len; ++i)
                space *= 26u;               // 26^len

            bit_cursor += space;
        }
        t.offsets[max_len + 1] = bit_cursor;

        // 2) Allocate bits
        uint32_t num_u32 = (bit_cursor + 31u) / 32u;
        t.bits.assign(num_u32, 0u);

        // 3) Fill bits
        for (int len = 2; len <= max_len; ++len)
        {
            auto it = words_by_length.find(len);
            if (it == words_by_length.end())
                continue;

            uint32_t base = t.offsets[len];

            for (const auto& w : it->second)
            {
                uint32_t code = GPUDecode::encode_word_base26(w);
                if (code == UINT32_MAX) continue; // bad word
                uint32_t bit_index = base + code;
                uint32_t word_idx = bit_index >> 5;
                uint32_t bit = bit_index & 31u;
                t.bits[word_idx] |= (1u << bit);
            }
        }

        return t;
    }

    static void init_gpu_filters(
        const std::unordered_set<std::uint16_t>& two_letter_set,
        const std::unordered_set<std::uint32_t>& four_letter_set,
        const std::vector<int>* spacing_pattern,
        const std::unordered_map<int, std::unordered_set<std::string>>* spacing_words_by_length,
        const WordlistParser::TrigramTable* triTable_ptr)
    {
        std::call_once(g_gpu_filter_init_flag, [&]()
            {
                // ---- 2-gram / 4-gram bitsets -> constant memory ----
                {
                    std::vector<std::uint16_t> bigram_codes;
                    bigram_codes.reserve(two_letter_set.size());
                    for (auto v : two_letter_set)
                        bigram_codes.push_back(v);

                    std::vector<std::uint32_t> quad_codes;
                    quad_codes.reserve(four_letter_set.size());
                    for (auto v : four_letter_set)
                        quad_codes.push_back(v);

                    std::vector<std::uint32_t> two_bits, four_bits;
                    build_bigram_bitset(bigram_codes, two_bits);
                    build_quadgram_bitset(quad_codes, four_bits);
                    gpu_upload_gram_bitsets(two_bits, four_bits);
                }

                // ---- Spacing dictionary -> device global + consts ----
                if (spacing_pattern && spacing_words_by_length)
                {
                    constexpr int MAX_WORD_LEN_FOR_BITSET = 6;

                    WordBitsetTable wtable =
                        build_word_bitset_table(*spacing_words_by_length,
                            MAX_WORD_LEN_FOR_BITSET);

                    gpu_upload_word_bitsets(*spacing_pattern,
                        wtable,
                        MAX_WORD_LEN_FOR_BITSET);
                }

                if (triTable_ptr && !triTable_ptr->empty())
                {
                    gpu_upload_q3_trigram_table(*triTable_ptr);
                }

            });
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

    // ==========================================================
    //  Phrase-building / phrase GPU re-check logic
    //  (extracted from process_keys for sanity)
    // ==========================================================

    template <typename ProcessHitsForKeysetFn>
    static void run_phrase_search_for_batch(
        const std::vector<std::string>& keys_slice,
        const std::vector<AlphabetCandidate>& alphabets,
        const std::vector<DeviceAlphabetTile>& tiles,
        const GpuBatchResult& gres,
        std::size_t a0,
        std::size_t a1,
        std::size_t text_len,
        const std::vector<int>* spacing_pattern,
        const std::unordered_map<int, std::unordered_set<std::string>>* spacing_words_by_length,
        const WordlistParser::SpacingPrefixIndex* spacing_prefix_map,
        const WordlistParser::GlobalPrefixIndex* key_prefix_map,
        const WordlistParser::TrigramTable* triTable_ptr,
        float IOC_GATE,
        float CHI_GATE,
        ProcessHitsForKeysetFn&& process_hits_for_keyset)
    {
        if (gres.front_hits.empty())
            return;

        PhraseBuilder::Q3PhraseSearchContext ctx{
            keys_slice,
            alphabets,
            tiles,
            gres,
            a0,
            a1,
            text_len,
            spacing_pattern,
            spacing_words_by_length,
            spacing_prefix_map,
            key_prefix_map,
            triTable_ptr,
            IOC_GATE,
            CHI_GATE
        };

        std::unique_ptr<DebugScopeTimer> t1 = std::make_unique<DebugScopeTimer>("Root Builder");
        // Stage 1: build roots
        auto root_states = PhraseBuilder::q3_build_initial_root_states(ctx);
        if (root_states.empty())
            return;
        t1.reset();


        std::unique_ptr<DebugScopeTimer> t2 = std::make_unique<DebugScopeTimer>("Word list builder");
        // Stage 2: precompute all-word IDs
        auto all_word_ids = PhraseBuilder::q3_build_all_word_ids(ctx);
        t2.reset();

        {
            DebugScopeTimer timer("Depth Loop");
            // Stage 3: depth loop
            for (int depth = 2; depth <= PhraseBuilder::Q3_MAX_PHRASE_WORDS; ++depth)
            {
                bool any_frontier = PhraseBuilder::q3_process_depth_level(
                    depth,
                    root_states,
                    ctx,
                    all_word_ids,
                    process_hits_for_keyset);

                if (!any_frontier)
                    break;
            }
        }
        

        auto a = 0;
    }



    // ==========================================================
    //  Main worker entry: GPU path + phrase expansion
    // ==========================================================
    static WorkerResult process_keys(
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
        const WordlistParser::SpacingPrefixIndex* spacing_prefix_map,
        const WordlistParser::GlobalPrefixIndex* key_prefix_map,
        const WordlistParser::TrigramTable* triTable_ptr,
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

        auto ensure_u8_capacity = [](std::unique_ptr<std::uint8_t[]>& buffer,
            std::size_t& capacity,
            std::size_t required)
        {
            if (capacity < required)
            {
                std::unique_ptr<std::uint8_t[]> new_buffer(new std::uint8_t[required]);
                buffer.swap(new_buffer);
                capacity = required;
            }
        };

#if defined(__AVX2__)
        auto ensure_keyblock_capacity = [](std::unique_ptr<KeyBlock[]>& buffer,
            std::size_t& capacity,
            std::size_t required)
        {
            if (capacity < required)
            {
                std::unique_ptr<KeyBlock[]> new_buffer(new KeyBlock[required]);
                buffer.swap(new_buffer);
                capacity = required;
            }
        };
#endif

        // build letter_mask once (A-Z only)
        std::fill(letter_mask.get(), letter_mask.get() + padded_len, 0);
        for (std::size_t i = 0; i < text_len; ++i)
        {
            char c = cipher[i];
            letter_mask[i] = (c >= 'A' && c <= 'Z') ? 1u : 0u;
        }

        // ---- GPU MEGA-BATCH PATH ----
        if (!use_cuda)
        {
            return result;
        }

        // device gates
        constexpr float IOC_GATE = 0.052f;
        constexpr float CHI_GATE = 100.0f;

        // Slice of keys for this worker [begin,end)
        std::vector<std::string> keys_slice;
        keys_slice.reserve(end - begin);
        for (std::size_t i = begin; i < end; ++i)
            keys_slice.push_back(keys[i]);

        // Pack keys ONCE as letters (device will map via index_map per alphabet)
        PackedKeysHostLetters packed_letters;
        {
            packed_letters.key_offsets.resize(keys_slice.size());
            packed_letters.key_lengths.resize(keys_slice.size());
            packed_letters.key_chars_flat.reserve(
                std::accumulate(keys_slice.begin(), keys_slice.end(), std::size_t(0),
                    [](std::size_t s, const std::string& k) { return s + k.size(); }));

            std::size_t off = 0;
            for (std::size_t i = 0; i < keys_slice.size(); ++i)
            {
                packed_letters.key_offsets[i] = static_cast<std::uint32_t>(off);
                packed_letters.key_lengths[i] = static_cast<std::uint16_t>(keys_slice[i].size());
                for (char ch : keys_slice[i])
                    packed_letters.key_chars_flat.push_back(static_cast<std::uint8_t>(ch));

                off += keys_slice[i].size();
            }
        }

        // Helper: rebuild cipher indices for a given alphabet
        auto rebuild_cipher_indices = [&](const AlphabetCandidate& alphabet)
        {
            for (std::size_t i = 0; i < text_len; ++i)
            {
                if (!letter_mask[i])
                {
                    cipher_indices[i] = 0;
                    continue;
                }
                int idx = AlphabetBuilder::alphabet_index(alphabet, cipher[i]);
                cipher_indices[i] = (idx >= 0) ? static_cast<std::uint8_t>(idx) : 0;
            }
            std::fill(cipher_indices.get() + text_len,
                cipher_indices.get() + padded_len, 0);
        };

        // Shared helper: given a GPU result set and a way to map key_id -> key string,
        // perform CPU static decrypt + autokey for all hits.
        auto process_hits_for_keyset =
            [&](const GpuBatchResult& gres_local, const auto& key_lookup)
        {
            // FULL hits: static decrypt, score, store candidate
            for (const auto& h : gres_local.full_hits)
            {
                if (h.alphabet_id >= alphabets.size())
                    continue;

                const auto& alphabet = alphabets[h.alphabet_id];

                const std::string* key_ptr = key_lookup(h.key_id);
                if (!key_ptr)
                    continue;

                const std::string& key_str = *key_ptr;
                const std::size_t key_len = key_str.size();
                if (!key_len)
                    continue;

                // Prepare key indices
                ensure_u8_capacity(key_indices, key_indices_capacity, key_len);
                ensure_u8_capacity(key_valid, key_valid_capacity, key_len);

                for (std::size_t i = 0; i < key_len; ++i)
                {
                    int idx = AlphabetBuilder::alphabet_index(alphabet, key_str[i]);
                    if (idx < 0) { key_valid[i] = 0; key_indices[i] = 0; }
                    else { key_valid[i] = 1; key_indices[i] = static_cast<std::uint8_t>(idx); }
                }

                rebuild_cipher_indices(alphabet);

#if defined(__AVX2__)
                ensure_keyblock_capacity(key_blocks, key_block_capacity, key_len);
                for (std::size_t start = 0; start < key_len; ++start)
                {
                    auto& block = key_blocks[start];
                    for (std::size_t j = 0; j < block.data.size(); ++j)
                        block.data[j] = key_indices[(start + j) % key_len];
                }
#endif

                CPUDecode::decrypt_repeating(
                    alphabet,
                    static_cast<Mode>(h.mode),
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


                std::string plaintext = CPUDecode::build_plaintext_string(alphabet, plaintext_indices.get(), text_len);

                const double ioc = static_cast<double>(h.ioc);
                const double chi = static_cast<double>(h.chi);

                Candidate cand = Candidate::make_candidate(
                    key_str, alphabet, static_cast<Mode>(h.mode),
                    /*autokey=*/false,
                    plaintext, preview_length,
                    spacing_pattern, spacing_words_by_length,
                    ioc, chi);

                CPUDecode::maintain_top_results(result.best, cand, 10);
            }

            // Deduplicate autokey on (key_id, mode) that already had a full hit
            struct KM { std::uint32_t k; std::uint8_t m; };
            struct H {
                std::size_t operator()(const KM& x) const noexcept
                {
                    return (std::size_t(x.k) << 2) ^ x.m;
                }
            };
            struct E {
                bool operator()(const KM& a, const KM& b) const noexcept
                {
                    return a.k == b.k && a.m == b.m;
                }
            };

            std::unordered_set<KM, H, E> seen_full;
            seen_full.reserve(gres_local.full_hits.size() * 2);
            for (const auto& h : gres_local.full_hits)
                seen_full.insert({ h.key_id, h.mode });

            if (!include_autokey)
                return;

            // AUTOKEY on front_hits not already covered by full_hits
            for (const auto& h : gres_local.front_hits)
            {
                if (seen_full.find({ h.key_id, h.mode }) != seen_full.end())
                    continue;

                if (h.alphabet_id >= alphabets.size())
                    continue;

                const auto& alphabet = alphabets[h.alphabet_id];

                const std::string* key_ptr = key_lookup(h.key_id);
                if (!key_ptr)
                    continue;

                const std::string& key_str = *key_ptr;
                const std::size_t key_len = key_str.size();
                if (!key_len)
                    continue;

                ensure_u8_capacity(key_indices, key_indices_capacity, key_len);
                ensure_u8_capacity(key_valid, key_valid_capacity, key_len);
                for (std::size_t i = 0; i < key_len; ++i)
                {
                    int idx = AlphabetBuilder::alphabet_index(alphabet, key_str[i]);
                    if (idx < 0) { key_valid[i] = 0; key_indices[i] = 0; }
                    else { key_valid[i] = 1; key_indices[i] = static_cast<std::uint8_t>(idx); }
                }

                rebuild_cipher_indices(alphabet);

                ++result.autokey_attempts;
                ++autokey_counter;

                ensure_u8_capacity(autokey_plaintext, autokey_capacity, text_len);

                CPUDecode::decrypt_autokey(
                    cipher_indices.get(),
                    text_len,
                    key_indices.get(),
                    key_len,
                    autokey_plaintext.get(),
                    static_cast<Mode>(h.mode));

                double ioc_auto = 0.0, chi_auto = 0.0;
                CPUDecode::compute_stats_indices(
                    autokey_plaintext.get(), text_len, alphabet,
                    ioc_auto, chi_auto);

                if (ioc_auto > IOC_GATE && chi_auto <= CHI_GATE)
                {
                    std::string plaintext_auto =
                        CPUDecode::build_plaintext_string(alphabet, autokey_plaintext.get(), text_len);

                    Candidate cand_auto = Candidate::make_candidate(
                        key_str, alphabet, static_cast<Mode>(h.mode),
                        /*autokey=*/true,
                        plaintext_auto, preview_length,
                        spacing_pattern, spacing_words_by_length,
                        ioc_auto, chi_auto);

                    CPUDecode::maintain_top_results(result.best, cand_auto, 10);
                }
            }
        }; // process_hits_for_keyset

        // Mega-batch alphabets per launch (tune). Aim to keep the GPU busy.
        constexpr std::size_t A_BATCH = 4096; // try 2k–8k depending on VRAM

        // Process alphabets in chunks of A_BATCH (each with 3 modes per alphabet)
        for (std::size_t a0 = 0; a0 < alphabet_count; a0 += A_BATCH)
        {
            const std::size_t a1 = std::min(a0 + A_BATCH, alphabet_count);
            const std::size_t batch_alphas = a1 - a0;

            // Build all tiles for this mega-batch
            std::vector<DeviceAlphabetTile> tiles;
            tiles.reserve(batch_alphas * 3);

            for (std::size_t a = a0; a < a1; ++a)
            {
                const auto& alphabet = alphabets[a];

                DeviceAlphabetTile base{};
                // cipher_idx/mask per alphabet
                for (std::size_t i = 0; i < text_len; ++i)
                {
                    if (!letter_mask[i])
                    {
                        base.cipher_idx[i] = 0;
                        base.mask[i] = 0;
                    }
                    else
                    {
                        int idx = AlphabetBuilder::alphabet_index(alphabet, cipher[i]);
                        base.cipher_idx[i] = (idx >= 0) ? static_cast<std::uint8_t>(idx) : 0;
                        base.mask[i] = 1;
                    }
                }
                // pad to padded_len
                for (std::size_t i = text_len; i < padded_len; ++i)
                {
                    base.cipher_idx[i] = 0;
                    base.mask[i] = 0;
                }
                // keyed alphabet + index map
                for (int j = 0; j < 26; ++j)
                {
                    base.alph[j] = alphabet.alphabet[j];
                    base.index_map[j] = static_cast<std::int8_t>(alphabet.index_map[j]); // -1 if absent
                }
                base.text_len = static_cast<int>(text_len);
                base.alphabet_id = static_cast<std::uint32_t>(a);

                // three modes
                for (int m = 0; m < 3; ++m)
                {
                    DeviceAlphabetTile t = base;
                    t.mode = static_cast<std::uint8_t>(m);
                    tiles.push_back(t);
                }
            }

            // progress accounting (approximate): all key×mode within this batch
            const std::size_t combos_for_batch = (end - begin) * 3 * batch_alphas;
            result.combos += combos_for_batch;
            combos_counter += combos_for_batch;

            // choose caps (heuristic). You can make these CLI knobs.
            const std::size_t max_full_hits =
                std::max<std::size_t>(packed_letters.num_keys() * tiles.size() / 1000, 8192);
            const std::size_t max_front_hits =
                std::max<std::size_t>(packed_letters.num_keys() * tiles.size() / 500, 16384);

            // one big launch for this batch
            GpuBatchResult gres =
                launch_q3_gpu_megabatch(
                    tiles, packed_letters, IOC_GATE, CHI_GATE,
                    max_full_hits, max_front_hits/*,
                    quadTable_ptr, false*/);

            keys_counter += gres.front_hits.size();

            // Base key lookup (single word)
            auto base_key_lookup =
                [&](std::uint32_t id) -> const std::string*
            {
                if (id >= keys_slice.size())
                    return nullptr;
                return &keys_slice[id];
            };

            // Handle base keys: GPU static -> CPU decrypt -> autokey
            process_hits_for_keyset(gres, base_key_lookup);

            // Phrase search (extracted)
            if (false)
            {
                run_phrase_search_for_batch(
                    keys_slice,
                    alphabets,
                    tiles,
                    gres,
                    a0,
                    a1,
                    text_len,
                    spacing_pattern,
                    spacing_words_by_length,
                    spacing_prefix_map,
                    key_prefix_map,
                    triTable_ptr,
                    IOC_GATE,
                    CHI_GATE,
                    process_hits_for_keyset);
            }

            // count each key once per processed batch (parity with your prior accounting)
            ++result.keys_processed;

            // periodically merge local best into global
            if ((a0 / A_BATCH) % 2 == 0)
            {
                std::lock_guard<std::mutex> lock(results_mutex);
                for (const auto& cand : result.best)
                    CPUDecode::maintain_top_results(global_results, cand, max_results);
            }
        } // for a0

        return result;
    }
};

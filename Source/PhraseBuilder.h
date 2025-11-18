#pragma once
#include <vector>
#include <array>
#include <string>
#include <unordered_map>
#include <unordered_set>


#include "AlphabetBuilder.h"
#include "gpu_quag.h"
#include "ScopedTimer.h"
#include "CPUDecode.h"

namespace PhraseBuilder
{
    using RootKeyPacked = std::uint64_t;
    using KeyedWord = std::vector<int>;

    constexpr int Q3_PREVIEW_MAX = 80;
    constexpr int Q3_MAX_PHRASE_WORDS = 4;
    constexpr std::size_t Q3_MAX_PHRASE_GLOBAL_TILE = 5000000;
    constexpr std::size_t MIN_GPU_KEYS_FOR_PHRASES = 16;   // e.g. don’t GPU < 256 keys
    constexpr std::size_t MIN_GPU_TOTAL_CHARS = 64;  // or skip if less than 4k chars
    constexpr int Q3_MAX_MISSING = 5;

    

    // Phrase node in the phrase search tree
    struct Q3PhraseNode
    {
        std::uint32_t word_ids[Q3_MAX_PHRASE_WORDS]{}; // fixed small array
        std::uint8_t  word_count{ 0 };

        std::uint32_t alphabet_id{ 0 };
        std::uint8_t  mode{ 0 };
        std::uint16_t key_len{ 0 };
        float         prefix_score{ 0.0f };            // float is enough here
        std::uint8_t  preview_len{ 0 };
        std::array<std::uint8_t, Q3_PREVIEW_MAX> preview{};
    };


    // State per "root" (unique (key, alphabet, mode) triple)
    struct Q3RootState
    {
        std::uint32_t            hit_index; // index into gres.front_hits
        std::vector<Q3PhraseNode> frontier;
    };

    // Tile grouping key
    struct Q3TileKey
    {
        std::uint32_t a;
        std::uint8_t  m;
    };

    struct Q3TileKeyHash
    {
        std::size_t operator()(const Q3TileKey& k) const noexcept
        {
            return (std::size_t(k.a) << 3) ^ std::size_t(k.m);
        }
    };

    struct Q3TileKeyEq
    {
        bool operator()(const Q3TileKey& x, const Q3TileKey& y) const noexcept
        {
            return x.a == y.a && x.m == y.m;
        }
    };

    enum class Q3NodeAllowedState : std::uint8_t
    {
        NotComputed,
        AllWords,   // no constraint; use full word list
        Filtered,   // use per-node candidate list
        Impossible  // node cannot be extended under spacing+dict+preview
    };

    struct Q3NodeConstraints
    {
        Q3NodeAllowedState            state = Q3NodeAllowedState::NotComputed;
        std::vector<std::uint32_t>    candidates; // valid next-word ids
    };


    // Immutable inputs for the phrase search
    struct Q3PhraseSearchContext
    {
        const std::vector<std::string>& keys_slice;
        const std::vector<AlphabetCandidate>& alphabets; // (currently unused, kept for symmetry)
        const std::vector<DeviceAlphabetTile>& tiles;
        const GpuBatchResult& gres;
        std::size_t a0;
        std::size_t a1;
        std::size_t text_len;
        const std::vector<int>* spacing_pattern;
        const std::unordered_map<int, std::unordered_set<std::string>>* spacing_words_by_length;
        const WordlistParser::SpacingPrefixIndex* spacing_prefix_index_map;
        const WordlistParser::GlobalPrefixIndex* key_prefix_map;
        const WordlistParser::QuadgramTable* quadTable_ptr;
        float IOC_GATE;
        float CHI_GATE;
    };

    inline int q3_invert_decrypt_idx(int cipher_idx, int plain_idx, std::uint8_t mode)
    {
        switch (mode)
        {
        case 0: // Vigenere: p = (c - k) mod 26 -> k = (c - p) mod 26
            return (cipher_idx - plain_idx + 26) % 26;
        case 1: // Beaufort: p = (k - c) mod 26 -> k = (p + cipher_idx) % 26
            return (plain_idx + cipher_idx) % 26;
        case 2: // Variant-like: p = (c + k) mod 26 -> k = (p - cipher_idx + 26) % 26
            return (plain_idx - cipher_idx + 26) % 26;
        default:
            return -1;
        }
    }

    inline RootKeyPacked q3_pack_root_key(std::uint32_t key_id,
        std::uint32_t alphabet_id,
        std::uint8_t  mode)
    {
        // 32 bits key_id | 24 bits alphabet_id | 8 bits mode (high)
        return (RootKeyPacked(key_id)
            | (RootKeyPacked(alphabet_id) << 32)
            | (RootKeyPacked(mode) << 56));
    }

    inline std::uint32_t q3_encode_seq(const std::uint8_t* s, int len)
    {
        std::uint32_t code = 0;
        for (int i = 0; i < len; ++i)
            code |= (std::uint32_t(s[i] & 0x1F) << (5 * i));
        return code;
    }

    inline int q3_get_cipher_idx_for_plain_pos(const DeviceAlphabetTile& tile,
        int plain_pos)
    {
        int lp = 0;
        for (int i = 0; i < tile.text_len; ++i)
        {
            if (!tile.mask[(std::size_t)i]) continue;
            if (lp == plain_pos)
                return (int)tile.cipher_idx[(std::size_t)i];
            ++lp;
        }
        return -1;
    }

    inline int q3_tile_index_for(std::uint32_t alph_id,
        std::uint8_t mode,
        std::size_t a0,
        std::size_t a1,
        const std::vector<DeviceAlphabetTile>& tiles)
    {
        if (alph_id < a0 || alph_id >= a1)
            return -1;
        std::size_t local_a = (std::size_t)(alph_id - a0);
        std::size_t idx = local_a * 3 + mode;
        if (idx >= tiles.size())
            return -1;
        return (int)idx;
    }

    static std::vector<Q3RootState>q3_build_initial_root_states(const Q3PhraseSearchContext& ctx)
    {
        std::vector<Q3RootState> root_states;
        if (ctx.gres.front_hits.empty())
            return root_states;

        std::unordered_map<RootKeyPacked, std::uint32_t> root_map;
        root_map.reserve(ctx.gres.front_hits.size() * 2);

        // 1) Build unique roots: (key_id, alphabet_id, mode) -> first hit index
        for (std::uint32_t hi = 0; hi < (std::uint32_t)ctx.gres.front_hits.size(); ++hi)
        {
            const HitRecord& h = ctx.gres.front_hits[hi];
            if (h.key_id >= ctx.keys_slice.size())
                continue;

            RootKeyPacked rk = q3_pack_root_key(h.key_id, h.alphabet_id, (std::uint8_t)h.mode);

            // keep first hit for each (key,alphabet,mode)
            if (root_map.find(rk) != root_map.end())
                continue;

            root_map.emplace(rk, hi);
        }

        if (root_map.empty())
            return root_states;

        // 2) Build per-root state and initial frontier
        root_states.reserve(root_map.size());

        for (const auto& kv : root_map)
        {
            const std::uint32_t hi = kv.second;
            const HitRecord& h = ctx.gres.front_hits[hi];

            if (h.alphabet_id < ctx.a0 || h.alphabet_id >= ctx.a1)
                continue;

            const std::uint32_t key_id = h.key_id;
            if (key_id >= ctx.keys_slice.size())
                continue;

            const auto& root_word = ctx.keys_slice[key_id];
            const std::size_t root_len = root_word.size();
            if (root_len == 0 || root_len > ctx.text_len)
                continue;

            Q3RootState rs;
            rs.hit_index = hi;

            Q3PhraseNode node;
            node.word_count = 1;
            node.word_ids[0] = key_id;
            node.alphabet_id = h.alphabet_id;
            node.mode = (std::uint8_t)h.mode;
            node.key_len = (std::uint16_t)root_len;
            node.prefix_score = h.ioc;
            node.preview_len = h.preview_len;
            std::memcpy(node.preview.data(), h.preview, h.preview_len);



            rs.frontier.push_back(std::move(node));
            root_states.push_back(std::move(rs));
        }

        return root_states;
    }

    // Small helper: decrypt p_idx (KEYED) from cipher_idx (KEYED) and key_idx (KEYED)
// for the various modes (same convention as elsewhere).
    inline int q3_autokey_decrypt_plain_idx(int cipher_idx, int key_idx, std::uint8_t mode)
    {
        cipher_idx &= 0x1F;
        key_idx &= 0x1F;

        switch (mode)
        {
        case 0: // Vigenere: c = p + k (mod 26) -> p = c - k
            return (cipher_idx - key_idx + 26) % 26;
        case 1: // Beaufort: c = k - p       -> p = k - c
            return (key_idx - cipher_idx + 26) % 26;
        case 2: // Variant-like: p = c + k   -> p = c + k
            return (cipher_idx + key_idx) % 26;
        default:
            return -1;
        }
    }

    inline std::uint32_t q3_pack_quadgram(const std::uint8_t* t)
    {
        return (((std::uint32_t)t[0] * 26u + (std::uint32_t)t[1]) * 26u + (std::uint32_t)t[2]) * 26u
            + (std::uint32_t)t[3];
    }

    inline double q3_score_quadgram_english(
        const std::uint8_t* text,
        int len,
        const double* quadFlat)
    {
        if (len < 4)
            return -1e9;

        if (!quadFlat)
            return WordlistParser::Q3_QUADGRAM_FLOOR_LOGP;

        double score = 0.0;
        int    quads = 0;

        for (int i = 0; i + 3 < len; ++i)
        {
            const std::uint8_t* p = text + i;

            if (p[0] >= 26 || p[1] >= 26 || p[2] >= 26 || p[3] >= 26)
            {
                score += WordlistParser::Q3_QUADGRAM_FLOOR_LOGP;
            }
            else
            {
                std::uint32_t code = q3_pack_quadgram(p);
                score += quadFlat[code];
            }
            ++quads;
        }

        return (quads > 0) ? (score / quads) : -1e9;
    }


    // Autokey preview test, with safer gating.
    static bool q3_autokey_preview_passes(
        const Q3PhraseNode& node,
        const Q3PhraseSearchContext& ctx,
        const DeviceAlphabetTile& tile,
        std::uint8_t mode,
        const std::array<int8_t, 26>& keyedToCanonical,
        const std::vector<int>& cipher_keyed) // keyed ciphertext [0..25]
    {
        // Tunables
        constexpr int AUTOKEY_LOOKAHEAD = 16;
        constexpr int MIN_PRIMER_FOR_PREVIEW = 8;    // don't preview for very short keys
        constexpr int MIN_PLAIN_FOR_PREVIEW = 8;    // need enough known plaintext
        constexpr double AUTOKEY_THRESH = -5.0;

        const int cipher_len = (int)cipher_keyed.size();
        if (cipher_len <= 0)
            return true; // nothing to say -> don't prune

        // Don't preview aggressively for tiny key phrases.
        if (node.key_len < MIN_PRIMER_FOR_PREVIEW)
            return true;

        // 1) Build primer in KEYED indices from node's words.
        std::vector<int> primer_keyed;
        primer_keyed.reserve(node.key_len);

        for (std::uint8_t wi = 0; wi < node.word_count; ++wi)
        {
            std::uint32_t wid = node.word_ids[wi];
            if (wid >= ctx.keys_slice.size())
                return true; // be conservative: don't prune if something is weird

            const std::string& w = ctx.keys_slice[wid];
            for (char ch : w)
            {
                if (ch < 'A' || ch > 'Z')
                    continue;

                int canon = ch - 'A';
                int keyed = tile.index_map[canon]; // canonical -> keyed
                if (keyed < 0 || keyed >= 26)
                    return true; // mapping issue, don't prune

                primer_keyed.push_back(keyed);
            }
        }

        if (primer_keyed.empty())
            return true;

        const int primer_len = (int)primer_keyed.size();

        // 2) Decrypt the FIRST primer_len positions using the primer only.
        //    This is the plaintext we "know" from the static Vigenere/autokey prefix.
        std::vector<int>          base_plain_keyed;
        std::vector<std::uint8_t> base_plain_canon;
        base_plain_keyed.reserve(primer_len);
        base_plain_canon.reserve(primer_len);

        const int max_prefix = std::min(primer_len, cipher_len);
        for (int pos = 0; pos < max_prefix; ++pos)
        {
            int c_keyed = cipher_keyed[pos];
            if (c_keyed < 0 || c_keyed >= 26)
                break;

            // For the first primer_len chars, autokey uses the primer directly.
            int k_keyed = primer_keyed[pos];
            int p_keyed = q3_autokey_decrypt_plain_idx(c_keyed, k_keyed, mode);
            if (p_keyed < 0 || p_keyed >= 26)
                break;

            int p_canon = keyedToCanonical[p_keyed];
            if (p_canon < 0 || p_canon >= 26)
                break;

            base_plain_keyed.push_back(p_keyed);
            base_plain_canon.push_back((std::uint8_t)p_canon);
        }

        const int plain_len = (int)base_plain_canon.size();
        if (plain_len < MIN_PLAIN_FOR_PREVIEW)
            return true; // not enough known plaintext to make a confident call

        // This is the amount of plaintext we currently have and will use as the autokey stream.
        const int window_len = plain_len;

        // We must have space ahead of the key to place at least one window.
        if (cipher_len <= primer_len + window_len)
            return true;

        // 3) Slide the KNOWN PLAINTEXT forward as the autokey stream.
        //
        // For a candidate alignment at 'start', the autokey boundary is assumed to be 'start',
        // and we use base_plain_keyed[0..window_len) as the key stream for
        // cipher_keyed[start .. start+window_len).
        //
        // We don't test anything before primer_len (we already know how that portion behaves),
        // and we cap how far we look ahead.
        const int max_start = std::min(cipher_len - window_len,
            primer_len + AUTOKEY_LOOKAHEAD);

        std::vector<std::uint8_t> win_plain_canon(window_len);

        double best = -1e9;
        bool   have_window = false;

        for (int start = primer_len; start <= max_start; ++start)
        {
            bool ok = true;
            int  wlen = 0;

            for (int j = 0; j < window_len; ++j)
            {
                int c_idx = start + j;
                if (c_idx >= cipher_len)
                {
                    ok = false;
                    break;
                }

                int c_keyed = cipher_keyed[c_idx];
                if (c_keyed < 0 || c_keyed >= 26)
                {
                    ok = false;
                    break;
                }

                int k_keyed = base_plain_keyed[j]; // known plaintext as autokey
                int p_keyed = q3_autokey_decrypt_plain_idx(c_keyed, k_keyed, mode);
                if (p_keyed < 0 || p_keyed >= 26)
                {
                    ok = false;
                    break;
                }

                int p_canon = keyedToCanonical[p_keyed];
                if (p_canon < 0 || p_canon >= 26)
                {
                    ok = false;
                    break;
                }

                win_plain_canon[wlen++] = (std::uint8_t)p_canon;
            }

            if (!ok || wlen < window_len)
                continue;

            have_window = true;

            // Score the whole window_len chunk; you could also sub-window with a fixed
            // AUTOKEY_WINDOW if you wanted tighter calibration, but this keeps it
            // "decode window == key_len".
            double s = q3_score_quadgram_english(
                win_plain_canon.data(),
                wlen,
                ctx.quadTable_ptr->data());

            if (s > best)
                best = s;

            if (best >= AUTOKEY_THRESH)
                return true; // found at least one promising alignment
        }

        // If we couldn't form any reasonable window, don't prune.
        if (!have_window)
            return true;

        // Otherwise, only keep this node if the best alignment passes the threshold.
        return (best >= AUTOKEY_THRESH);
    }


    struct PruneChunkResult
    {
        std::vector<Q3PhraseNode>  nodes;
        std::vector<std::uint32_t> meta;
    };

    // Parallel pruning driver
    static void q3_parallel_autokey_prune(
        const Q3PhraseSearchContext& ctx,
        const DeviceAlphabetTile& tile,
        std::uint8_t mode,
        const std::array<int8_t, 26>& keyedToCanonical,
        const std::vector<int>& cipher_keyed,
        std::vector<Q3PhraseNode>& expanded_nodes,
        std::vector<std::uint32_t>& meta_root_idx)
    {
        const std::size_t N = expanded_nodes.size();
        if (N == 0)
            return;

        // If small, don't bother with threads
        constexpr std::size_t PARALLEL_THRESHOLD = 256;
        if (N < PARALLEL_THRESHOLD)
        {
            std::vector<Q3PhraseNode>  filtered_nodes;
            std::vector<std::uint32_t> filtered_meta;
            filtered_nodes.reserve(N);
            filtered_meta.reserve(N);

            for (std::size_t i = 0; i < N; ++i)
            {
                const Q3PhraseNode& pn = expanded_nodes[i];
                if (q3_autokey_preview_passes(pn, ctx, tile, mode, keyedToCanonical, cipher_keyed))
                {
                    filtered_nodes.push_back(pn);
                    filtered_meta.push_back(meta_root_idx[i]);
                }
            }

            expanded_nodes.swap(filtered_nodes);
            meta_root_idx.swap(filtered_meta);
            return;
        }

        // Decide how many threads
        const unsigned hw = std::thread::hardware_concurrency();
        const unsigned maxThreads = (hw == 0 ? 4u : hw);
        // Keep 1 core for "other" work / OS
        const unsigned numThreads = std::min<unsigned>(maxThreads - 1, (unsigned)(N / PARALLEL_THRESHOLD));
        if (numThreads <= 1)
        {
            // fallback to serial
            // (You could just tail-call the serial branch above to avoid duplication)
            std::vector<Q3PhraseNode>  filtered_nodes;
            std::vector<std::uint32_t> filtered_meta;
            filtered_nodes.reserve(N);
            filtered_meta.reserve(N);

            for (std::size_t i = 0; i < N; ++i)
            {
                const Q3PhraseNode& pn = expanded_nodes[i];
                if (q3_autokey_preview_passes(pn, ctx, tile, mode, keyedToCanonical, cipher_keyed))
                {
                    filtered_nodes.push_back(pn);
                    filtered_meta.push_back(meta_root_idx[i]);
                }
            }

            expanded_nodes.swap(filtered_nodes);
            meta_root_idx.swap(filtered_meta);
            return;
        }

        const std::size_t chunkSize = (N + numThreads - 1) / numThreads;

        std::vector<std::thread>       threads;
        std::vector<PruneChunkResult>  results(numThreads);

        threads.reserve(numThreads);

        for (unsigned t = 0; t < numThreads; ++t)
        {
            const std::size_t begin = t * chunkSize;
            const std::size_t end = std::min<std::size_t>(begin + chunkSize, N);
            if (begin >= end)
                break;

            threads.emplace_back(
                [&, t, begin, end]()
                {
                    PruneChunkResult local;
                    local.nodes.reserve(end - begin);
                    local.meta.reserve(end - begin);

                    for (std::size_t i = begin; i < end; ++i)
                    {
                        const Q3PhraseNode& pn = expanded_nodes[i];
                        if (q3_autokey_preview_passes(pn, ctx, tile, mode, keyedToCanonical, cipher_keyed))
                        {
                            local.nodes.push_back(pn);
                            local.meta.push_back(meta_root_idx[i]);
                        }
                    }

                    results[t] = std::move(local);
                });
        }

        for (auto& th : threads)
            th.join();

        // Merge results in thread index order (preserves relative order within chunks)
        std::vector<Q3PhraseNode>  filtered_nodes;
        std::vector<std::uint32_t> filtered_meta;

        filtered_nodes.reserve(N); // upper bound
        filtered_meta.reserve(N);

        for (unsigned t = 0; t < results.size(); ++t)
        {
            auto& r = results[t];
            filtered_nodes.insert(filtered_nodes.end(),
                std::make_move_iterator(r.nodes.begin()),
                std::make_move_iterator(r.nodes.end()));
            filtered_meta.insert(filtered_meta.end(),
                std::make_move_iterator(r.meta.begin()),
                std::make_move_iterator(r.meta.end()));
        }

        expanded_nodes.swap(filtered_nodes);
        meta_root_idx.swap(filtered_meta);
    }


    static std::vector<std::uint32_t> q3_build_all_word_ids(const Q3PhraseSearchContext& ctx)
    {
        std::vector<std::uint32_t> all_word_ids;
        all_word_ids.reserve(ctx.keys_slice.size());
        for (std::uint32_t i = 0; i < (std::uint32_t)ctx.keys_slice.size(); ++i)
            all_word_ids.push_back(i);
        return all_word_ids;
    }

    static const std::vector<std::uint32_t>* q3_compute_constraints_for_node(
        std::uint32_t root_idx,
        std::uint32_t frontier_idx,
        std::vector<Q3RootState>& root_states,
        const Q3PhraseSearchContext& ctx,
        const DeviceAlphabetTile& tile,
        std::uint8_t mode,
        const std::vector<int>& logicalToCipher,                        // KEYED cipher indices
        const std::array<int8_t, 26>& keyedToCanonical,                 // keyed -> canonical
        std::unordered_map<int, std::vector<KeyedWord>>& dict_keyed_by_len,
        std::vector<std::vector<Q3NodeConstraints>>& node_constraints,
        bool have_spacing)
    {
        auto& rs = root_states[root_idx];

        if (frontier_idx >= rs.frontier.size())
            return nullptr;

        Q3PhraseNode& node = rs.frontier[frontier_idx];

        // Ensure constraints slot
        auto& v = node_constraints[root_idx];
        if (v.size() < rs.frontier.size())
            v.resize(rs.frontier.size());

        Q3NodeConstraints& nc = v[frontier_idx];

        if (nc.state != Q3NodeAllowedState::NotComputed)
        {
            if (nc.state == Q3NodeAllowedState::Filtered)
                return &nc.candidates;
            // AllWords or Impossible
            return nullptr;
        }

        // --- Quick exits when spacing is not usable ---
        if (!have_spacing ||
            node.key_len == 0 ||
            !ctx.spacing_pattern || !ctx.spacing_words_by_length)
        {
            nc.state = Q3NodeAllowedState::AllWords;
            return nullptr;
        }

        const auto& sp = *ctx.spacing_pattern;
        const auto& dict_by_len = *ctx.spacing_words_by_length;
        const auto* prefix_idx = ctx.spacing_prefix_index_map;

        const int preview_len = (int)node.preview_len;
        const int key_len = (int)node.key_len;

        if (sp.empty() || preview_len <= 0)
        {
            nc.state = Q3NodeAllowedState::AllWords;
            return nullptr;
        }

        // --------------------------------------------------------
        // Find which spacing word is partially covered by key_len
        // --------------------------------------------------------
        int offset = 0;   // global plaintext letter index
        int slot = -1;    // index in spacing pattern
        int covered_in_slot = 0;  // letters of that word already covered

        for (int wi = 0; wi < (int)sp.size(); ++wi)
        {
            const int len = sp[wi];
            const int end = offset + len;

            // If key_len is within [offset, end), we are in THIS word.
            // This handles both:
            //   - interior:  offset < key_len < end  -> covered_in_slot > 0
            //   - boundary:  key_len == offset      -> covered_in_slot == 0 (next word case)
            if (key_len >= offset && key_len < end)
            {
                slot = wi;
                covered_in_slot = key_len - offset; // can be 0 or >0
                break;
            }

            offset = end; // fully covered this word, move to next

            if (key_len < offset)
                break; // key is before this word => no slot
        }

        if (slot < 0)
        {
            // Could not tie this node to any spacing slot at all
            nc.state = Q3NodeAllowedState::AllWords;
            return nullptr;
        }


        const int word_len = sp[slot];
        int       missing = word_len - covered_in_slot; // unknown letters remaining

        if (missing <= 0)
        {
            nc.state = Q3NodeAllowedState::AllWords;
            return nullptr;
        }

        // We only handle up to Q3_MAX_MISSING letters efficiently.
        if (missing > Q3_MAX_MISSING)
        {
            nc.state = Q3NodeAllowedState::AllWords;
            return nullptr;
        }

        // Only require preview coverage if we actually use preview
        // (i.e. we have a partially covered word).
        if (covered_in_slot > 0)
        {
            const int required_preview = offset + covered_in_slot;
            if (required_preview > preview_len)
            {
                nc.state = Q3NodeAllowedState::AllWords;
                return nullptr;
            }
        }


        auto dict_it = dict_by_len.find(word_len);
        if (dict_it == dict_by_len.end())
        {
            nc.state = Q3NodeAllowedState::AllWords;
            return nullptr;
        }

        const auto& wordset = dict_it->second;

        // --------------------------------------------------------
        // Choose dictionary subset:
        //   - If we have spacing_prefix_map, restrict to words
        //     of (word_len, prefix) from preview (1–2 letters).
        //   - Otherwise, use full "word_len" bucket.
        //   Dictionary words are stored in KEYED space.
        // --------------------------------------------------------
        std::vector<KeyedWord>        local_keyed_list; // used when prefix index is active
        const std::vector<KeyedWord>* dict_words = nullptr;

        auto& keyed_list_full = dict_keyed_by_len[word_len];

        bool used_prefix_index = false;

        if (prefix_idx && !prefix_idx->empty() && covered_in_slot > 0)
        {
            const int maxPrefixLen = 2;
            int prefix_len_for_index =
                std::min(std::min(covered_in_slot, word_len), maxPrefixLen);

            if (prefix_len_for_index > 0)
            {
                // node.preview is assumed CANONICAL plaintext indices (0..25)
                int p0 = node.preview[(std::size_t)offset];
                if (p0 < 0 || p0 >= 26)
                {
                    prefix_len_for_index = 0;
                }

                int p1 = 0;
                if (prefix_len_for_index >= 2)
                {
                    p1 = node.preview[(std::size_t)(offset + 1)];
                    if (p1 < 0 || p1 >= 26)
                    {
                        prefix_len_for_index = 1; // drop back to 1-letter index
                    }
                }

                if (prefix_len_for_index > 0)
                {
                    WordlistParser::SpacingPrefixKey k{};
                    k.wordLen = (std::uint16_t)word_len;
                    k.prefixLen = (std::uint8_t)prefix_len_for_index;
                    k.c0 = (std::uint8_t)p0;
                    k.c1 = (prefix_len_for_index >= 2
                        ? (std::uint8_t)p1
                        : (std::uint8_t)0xFF);

                    auto itPI = prefix_idx->find(k);
                    if (itPI == prefix_idx->end())
                    {
                        // No dictionary words of this length with this prefix -> impossible node
                        nc.state = Q3NodeAllowedState::Impossible;
                        return nullptr;
                    }

                    const auto& subset = itPI->second; // std::vector<const std::string*>

                    // Build KEYED dictionary for this subset only
                    local_keyed_list.reserve(subset.size());
                    for (const std::string* wPtr : subset)
                    {
                        const std::string& w = *wPtr;
                        if ((int)w.size() != word_len)
                            continue;

                        KeyedWord plain;
                        plain.resize(word_len);
                        bool ok2 = true;
                        for (int i = 0; i < word_len; ++i)
                        {
                            char ch = w[(std::size_t)i];
                            if (ch < 'A' || ch > 'Z') { ok2 = false; break; }

                            int pi = tile.index_map[ch - 'A']; // canonical->keyed
                            if (pi < 0) { ok2 = false; break; }
                            plain[i] = pi;
                        }
                        if (ok2)
                            local_keyed_list.push_back(std::move(plain));
                    }

                    dict_words = &local_keyed_list;
                    used_prefix_index = true;
                }
            }
        }

        if (!used_prefix_index)
        {
            // Build/reuse full KEYED list for this word length
            auto& full = keyed_list_full;
            if (full.empty())
            {
                full.reserve(wordset.size());
                for (const std::string& w : wordset)
                {
                    if ((int)w.size() != word_len)
                        continue;

                    KeyedWord plain;
                    plain.resize(word_len);
                    bool ok2 = true;
                    for (int i = 0; i < word_len; ++i)
                    {
                        char ch = w[(std::size_t)i];
                        if (ch < 'A' || ch > 'Z') { ok2 = false; break; }

                        int pi = tile.index_map[ch - 'A']; // canonical->keyed
                        if (pi < 0) { ok2 = false; break; }
                        plain[i] = pi;
                    }
                    if (ok2)
                        full.push_back(std::move(plain));
                }
            }
            dict_words = &full;
        }

        if (!dict_words || dict_words->empty())
        {
            // No usable dictionary words -> no constraint
            nc.state = Q3NodeAllowedState::AllWords;
            return nullptr;
        }

        // --------------------------------------------------------
        // Build set of allowed missing-letter key sequences
        // BUT store them in CANONICAL key space.
        //
        // We still:
        //   - use KEYED cipher indices (logicalToCipher)
        //   - use KEYED plaintext indices (plain[])
        //   - compute KEYED key index via q3_invert_decrypt_idx
        // Then map each key index -> canonical via keyedToCanonical
        // and pack that canonical sequence.
        // --------------------------------------------------------
        std::unordered_set<std::uint32_t> allowed_sequences;
        allowed_sequences.reserve(dict_words->size());

        for (const auto& plain : *dict_words)
        {
            bool ok = true;

            // Covered part of spacing word must match preview
            // preview is canonical; dict is keyed -> map preview through tile.index_map
            for (int k = 0; k < covered_in_slot; ++k)
            {
                int prev_bin = (int)node.preview[(std::size_t)(offset + k)]; // canonical 0..25
                if (prev_bin < 0 || prev_bin >= 26) { ok = false; break; }

                int prev_alpha = tile.index_map[prev_bin]; // canonical->keyed
                if (prev_alpha < 0) { ok = false; break; }

                if (plain[k] != prev_alpha)
                {
                    ok = false;
                    break;
                }
            }

            if (!ok)
                continue;

            std::array<std::uint8_t, Q3_MAX_MISSING> seq{};
            int seq_len = 0;

            for (int j = 0; j < missing; ++j)
            {
                const int global_plain_pos = offset + covered_in_slot + j;
                if (global_plain_pos < 0 ||
                    global_plain_pos >= (int)logicalToCipher.size())
                {
                    ok = false;
                    break;
                }

                const int c_idx_keyed = logicalToCipher[global_plain_pos]; // KEYED cipher index
                if (c_idx_keyed < 0)
                {
                    ok = false;
                    break;
                }

                const int p_idx_keyed = plain[covered_in_slot + j];        // KEYED plain index

                const int k_idx_keyed = q3_invert_decrypt_idx(c_idx_keyed, p_idx_keyed, mode);
                if (k_idx_keyed < 0 || k_idx_keyed >= 26)
                {
                    ok = false;
                    break;
                }

                // Convert KEYED key index to CANONICAL letter
                int canon_k = keyedToCanonical[k_idx_keyed];
                if (canon_k < 0 || canon_k >= 26)
                {
                    ok = false;
                    break;
                }

                seq[seq_len++] = (std::uint8_t)canon_k; // store canonical key letter
            }

            if (!ok || seq_len != missing)
                continue;

            std::uint32_t code = q3_encode_seq(seq.data(), seq_len); // CANONICAL-packed
            allowed_sequences.insert(code);
        }

        if (allowed_sequences.empty())
        {
            // Valid spacing slot but no dictionary word can cohere
            // with this node under this tile/mode => node is dead.
            nc.state = Q3NodeAllowedState::Impossible;
            return nullptr;
        }

        // --------------------------------------------------------
        // Filter keywords by whether they can realize any allowed sequence.
        // Now allowed_sequences is CANONICAL, same space as GlobalPrefixIndex.
        // No conversion needed at lookup time.
        // --------------------------------------------------------
        nc.candidates.clear();

        if (ctx.key_prefix_map &&
            missing > 0 &&
            missing <= WordlistParser::GlobalPrefixIndex::MAX_PREFIX_LEN)
        {
            const WordlistParser::GlobalPrefixIndex& gidx = *ctx.key_prefix_map;
            const auto& bucketMap = gidx.byLen[missing];

            for (std::uint32_t canon_code : allowed_sequences)
            {
                auto it = bucketMap.find(canon_code);
                if (it == bucketMap.end())
                    continue;

                const auto& wids = it->second; // std::vector<std::uint32_t>
                nc.candidates.insert(nc.candidates.end(), wids.begin(), wids.end());
            }

            if (nc.candidates.empty())
            {
                // No global-prefix matches at all for this node -> impossible
                nc.state = Q3NodeAllowedState::Impossible;
                return nullptr;
            }

            std::sort(nc.candidates.begin(), nc.candidates.end());
            nc.candidates.erase(
                std::unique(nc.candidates.begin(), nc.candidates.end()),
                nc.candidates.end());

            nc.state = Q3NodeAllowedState::Filtered;
            return &nc.candidates;
        }

        // If no key_prefix_map or missing is out of range, treat as no constraint.
        nc.state = Q3NodeAllowedState::AllWords;
        return nullptr;
    }










    template <typename ProcessHitsForKeysetFn>
    static bool q3_process_depth_level(
        int depth,
        std::vector<Q3RootState>& root_states,
        const Q3PhraseSearchContext& ctx,
        const std::vector<std::uint32_t>& all_word_ids,
        ProcessHitsForKeysetFn&& process_hits_for_keyset)
    {
        
        (void)depth; // currently unused, but nice to have for logging/profiling labels

        using GroupMap = std::unordered_map<
            Q3TileKey,
            std::vector<std::pair<std::uint32_t, std::uint32_t>>,
            Q3TileKeyHash,
            Q3TileKeyEq>;

        GroupMap groups;
        groups.reserve(root_states.size() * 2);

        // ----------------------------------------------------------------
        // Group active frontier nodes by (alphabet,mode) tile
        // ----------------------------------------------------------------
        for (std::uint32_t r = 0; r < (std::uint32_t)root_states.size(); ++r)
        {
            auto& rs = root_states[r];
            if (rs.frontier.empty())
                continue;

            const HitRecord& rh = ctx.gres.front_hits[rs.hit_index];

            if (rh.alphabet_id < ctx.a0 || rh.alphabet_id >= ctx.a1)
                continue;

            Q3TileKey tk{ rh.alphabet_id, (std::uint8_t)rh.mode };
            auto& vec = groups[tk];

            for (std::uint32_t fi = 0; fi < (std::uint32_t)rs.frontier.size(); ++fi)
                vec.emplace_back(r, fi);
        }

        if (groups.empty())
            return false;

        std::vector<std::vector<Q3PhraseNode>> new_frontiers(root_states.size());
        bool any_frontier = false;

        const bool have_spacing =
            (ctx.spacing_pattern && ctx.spacing_words_by_length &&
                !ctx.spacing_pattern->empty() && !ctx.spacing_words_by_length->empty());
        
        // ----------------------------------------------------------------
        // Process each tile group
        // ----------------------------------------------------------------
        for (const auto& entry : groups)
        {
            const Q3TileKey& tk = entry.first;
            const auto& rf_pairs = entry.second;
            if (rf_pairs.empty())
                continue;

            const std::uint32_t alph_id = tk.a;
            const std::uint8_t  mode = tk.m;

            int t_idx = q3_tile_index_for(alph_id, mode, ctx.a0, ctx.a1, ctx.tiles);
            if (t_idx < 0)
                continue;

            const DeviceAlphabetTile& tile = ctx.tiles[(std::size_t)t_idx];
            const AlphabetCandidate& alphabet = ctx.alphabets[alph_id];

            const std::size_t total_frontier_nodes = rf_pairs.size();
            const std::size_t theoretical_phrases =
                total_frontier_nodes * ctx.keys_slice.size();
            const std::size_t depth_cap =
                std::min<std::size_t>(theoretical_phrases, Q3_MAX_PHRASE_GLOBAL_TILE);
            if (depth_cap == 0)
                continue;

            std::vector<Q3PhraseNode>  expanded_nodes;
            std::vector<std::uint32_t> meta_root_idx;
            expanded_nodes.reserve(depth_cap);
            meta_root_idx.reserve(depth_cap);

            // --------------------------------------------------------------------
            // Precomputations for this tile group
            // --------------------------------------------------------------------

            // 1) logical plaintext index -> cipher_idx (KEYED alphabet)
            std::vector<int> logicalToCipher;
            logicalToCipher.reserve(tile.text_len);
            for (int i = 0; i < tile.text_len; ++i)
            {
                if (!tile.mask[i]) continue;
                logicalToCipher.push_back((int)tile.cipher_idx[i]); // keyed index 0..25
            }

            // 3) dictionary cache by word_len in KEYED indices
            std::unordered_map<int, std::vector<KeyedWord>> dict_keyed_by_len;

            // Per-node constraints cache (per root/frontier index)
            std::vector<std::vector<Q3NodeConstraints>> node_constraints(root_states.size());


            // 2) keyed -> canonical map (inverse of tile.index_map)
            std::array<int8_t, 26> keyedToCanonical;
            keyedToCanonical.fill(-1);
            for (int canon = 0; canon < 26; ++canon)
            {
                int keyed = tile.index_map[canon]; // canonical -> keyed
                if (keyed >= 0 && keyed < 26)
                    keyedToCanonical[keyed] = (int8_t)canon;
            }


            // helper: encode a short sequence of indices into a 32-bit code
            auto encode_seq = [](const std::uint8_t* s, int len) -> std::uint32_t
            {
                std::uint32_t code = 0;
                for (int i = 0; i < len; ++i)
                    code |= (std::uint32_t(s[i] & 0x1F) << (5 * i));
                return code;
            };

            // ------------------------------------------------------------
            // Expansion within this tile group
            // ------------------------------------------------------------
            auto add_expansion =
                [&](std::uint32_t root_idx,
                    const Q3PhraseNode& base,
                    std::uint32_t next_word_id)
            {
                if (expanded_nodes.size() >= depth_cap)
                    return;
                if (next_word_id >= ctx.keys_slice.size())
                    return;

                const auto& word = ctx.keys_slice[next_word_id];
                if (word.empty())
                    return;

                if (base.word_count >= Q3_MAX_PHRASE_WORDS)
                    return; // should not happen, but be safe

                const std::uint16_t new_len =
                    static_cast<std::uint16_t>(base.key_len + word.size());
                if (new_len > ctx.text_len)
                    return;

                Q3PhraseNode child;
                child.word_count = base.word_count + 1;
                // copy existing word_ids
                for (std::uint8_t i = 0; i < base.word_count; ++i)
                    child.word_ids[i] = base.word_ids[i];
                child.word_ids[base.word_count] = next_word_id;

                child.alphabet_id = base.alphabet_id;
                child.mode = base.mode;
                child.key_len = new_len;
                child.prefix_score = base.prefix_score;
                child.preview_len = base.preview_len;
                child.preview = base.preview; // std::array copy

                expanded_nodes.push_back(std::move(child));
                meta_root_idx.push_back(root_idx);
            };

            // Expand all frontier nodes in this tile group
            {
                DebugScopeTimer rf("RF PAIRS");
                for (const auto& rf : rf_pairs)
                {
                    if (expanded_nodes.size() >= depth_cap)
                        break;

                    const std::uint32_t root_idx = rf.first;
                    const std::uint32_t frontier_idx = rf.second;

                    auto& rs = root_states[root_idx];
                    if (frontier_idx >= rs.frontier.size())
                        continue;

                    const Q3PhraseNode& base = rs.frontier[frontier_idx];

                    const std::vector<std::uint32_t>* candidates = &all_word_ids;

                    if (have_spacing)
                    {
                        const std::vector<std::uint32_t>* filtered =
                            q3_compute_constraints_for_node(
                                root_idx,
                                frontier_idx,
                                root_states,
                                ctx,
                                tile,
                                mode,
                                logicalToCipher,
                                keyedToCanonical,
                                dict_keyed_by_len,
                                node_constraints,
                                have_spacing);

                        if (filtered)
                            candidates = filtered;

                        Q3NodeAllowedState st = node_constraints[root_idx][frontier_idx].state;

                        if (st == Q3NodeAllowedState::Impossible)
                            continue; // dead node
                    }


                    for (std::uint32_t next_id : *candidates)
                    {
                        add_expansion(root_idx, base, next_id);
                        if (expanded_nodes.size() >= depth_cap)
                            break;
                    }
                }
            }
            

            // ------------------------------------------------------------
            // Optional Autokey pruning BEFORE GPU:
            // For each expanded phrase, run a short autokey preview and
            // prune phrases whose preview never hits a "good" n-gram window.
            // ------------------------------------------------------------
            if (false)
            {
                std::vector<Q3PhraseNode>  filtered_nodes;
                std::vector<std::uint32_t> filtered_meta;
                filtered_nodes.reserve(expanded_nodes.size());
                filtered_meta.reserve(meta_root_idx.size());

                for (std::size_t i = 0; i < expanded_nodes.size(); ++i)
                {
                    const Q3PhraseNode& pn = expanded_nodes[i];

                    // If you only want this for autokey searches, you could guard here
                    // (e.g. if (!ctx.use_autokey_preview) { filtered_nodes.push_back(...); })
                    // For now we always apply.
                    if (q3_autokey_preview_passes(pn, ctx, tile, mode, keyedToCanonical, logicalToCipher))
                    {
                        filtered_nodes.push_back(pn);
                        filtered_meta.push_back(meta_root_idx[i]);
                    }
                }

                expanded_nodes.swap(filtered_nodes);
                meta_root_idx.swap(filtered_meta);
            }

            if (expanded_nodes.empty())
                continue;


            // ------------------------------------------------------------
            // Pack phrase keys (no intermediate strings)
            // ------------------------------------------------------------
            PackedKeysHostLetters phrase_packed;
            phrase_packed.key_offsets.resize(expanded_nodes.size());
            phrase_packed.key_lengths.resize(expanded_nodes.size());

            std::size_t total_len = 0;
            for (const auto& pn : expanded_nodes)
                total_len += pn.key_len;

            phrase_packed.key_chars_flat.resize(total_len);

            {
                std::size_t off = 0;
                for (std::size_t i = 0; i < expanded_nodes.size(); ++i)
                {
                    phrase_packed.key_offsets[i] = (uint32_t)off;
                    phrase_packed.key_lengths[i] = expanded_nodes[i].key_len;

                    const auto& pn = expanded_nodes[i];
                    for (std::uint8_t wi = 0; wi < pn.word_count; ++wi)
                    {
                        auto wid = pn.word_ids[wi];
                        const auto& w = ctx.keys_slice[wid];
                        std::memcpy(&phrase_packed.key_chars_flat[off],
                            w.data(), w.size());
                        off += w.size();
                    }
                }
            }

            // GPU caps for this group
            std::vector<DeviceAlphabetTile> group_tiles(1);
            group_tiles[0] = tile;

            const std::size_t num_keys = phrase_packed.num_keys();

            const std::size_t phrase_full_cap =
                std::max<std::size_t>(num_keys * group_tiles.size() / 1000,
                    (std::size_t)8192);
            const std::size_t phrase_front_cap =
                std::max<std::size_t>(num_keys * group_tiles.size() / 500,
                    (std::size_t)16384);

            // ------------------------------------------------------------
            // Decide CPU vs GPU for this phrase batch
            // ------------------------------------------------------------
            const bool use_gpu = true;

            GpuBatchResult phrase_gres;

            {
                if (use_gpu)
                {
                    phrase_gres =
                        launch_q3_gpu_megabatch(
                            group_tiles,
                            phrase_packed,
                            ctx.IOC_GATE,
                            ctx.CHI_GATE,
                            phrase_full_cap,
                            phrase_front_cap
                            /*ctx.quadTable_ptr,
                            true*/);
                }
                else
                {
                    // CPU fallback for small batches
                    phrase_gres =
                        CPUDecode::launch_q3_cpu_phrase_megabatch(
                            tile,
                            alphabet,
                            alph_id,
                            static_cast<Mode>(mode),
                            phrase_packed,
                            ctx.IOC_GATE,
                            ctx.CHI_GATE,
                            phrase_full_cap,
                            phrase_front_cap);
                }
            }

            // key_lookup that reconstructs the phrase string on demand
            auto phrase_key_lookup =
                [&](std::uint32_t id) -> const std::string*
            {
                if (id >= expanded_nodes.size())
                    return nullptr;

                static thread_local std::string tmp;
                tmp.clear();

                const auto& pn = expanded_nodes[id];
                tmp.reserve(pn.key_len);

                for (std::uint8_t wi = 0; wi < pn.word_count; ++wi)
                {
                    auto wid = pn.word_ids[wi];
                    tmp += ctx.keys_slice[wid];
                }

                return &tmp;
            };

            // Reuse "GPU static -> CPU static + autokey" pipeline
            process_hits_for_keyset(phrase_gres, phrase_key_lookup);

            // Build next frontier per root from phrase hits
            std::vector<char>  keep(expanded_nodes.size(), 0);
            std::vector<float> new_scores(expanded_nodes.size(), 0.0f);

            for (const auto& h : phrase_gres.front_hits)
            {
                if (h.key_id < keep.size())
                {
                    keep[h.key_id] = 1;
                    new_scores[h.key_id] = h.ioc;

                    // Store this phrase's preview into the node so it can be used
                    // for constraints at the next depth.
                    Q3PhraseNode& pn = expanded_nodes[h.key_id];
                    pn.preview_len = h.preview_len;
                    std::memcpy(pn.preview.data(), h.preview, h.preview_len);
                }
            }

            for (std::size_t i = 0; i < expanded_nodes.size(); ++i)
            {
                if (!keep[i])
                    continue;

                const auto root_idx = meta_root_idx[i];
                if (root_idx >= new_frontiers.size())
                    continue;

                expanded_nodes[i].prefix_score = new_scores[i];
                new_frontiers[root_idx].push_back(std::move(expanded_nodes[i]));
                any_frontier = true;
            }

        } // tile groups

        // Commit new frontiers
        for (std::size_t r = 0; r < root_states.size(); ++r)
        {
            auto& nf = new_frontiers[r];
            if (!nf.empty())
                root_states[r].frontier.swap(nf);
            else
                root_states[r].frontier.clear();
        }

        return any_frontier;
    }


};
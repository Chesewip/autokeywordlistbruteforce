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

    constexpr int Q3_PREVIEW_MAX = 80;
    constexpr int Q3_MAX_PHRASE_WORDS = 3;
    constexpr std::size_t Q3_MAX_PHRASE_GLOBAL_TILE = 500000;
    constexpr std::size_t MIN_GPU_KEYS_FOR_PHRASES = 16;   // e.g. don’t GPU < 256 keys
    constexpr std::size_t MIN_GPU_TOTAL_CHARS = 64;  // or skip if less than 4k chars

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

    static std::vector<std::uint32_t> q3_build_all_word_ids(const Q3PhraseSearchContext& ctx)
    {
        std::vector<std::uint32_t> all_word_ids;
        all_word_ids.reserve(ctx.keys_slice.size());
        for (std::uint32_t i = 0; i < (std::uint32_t)ctx.keys_slice.size(); ++i)
            all_word_ids.push_back(i);
        return all_word_ids;
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
        std::unique_ptr<DebugScopeTimer> t1 = std::make_unique<DebugScopeTimer>("GROUP Mapping");
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
        t1.reset();

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

            // 1) logical plaintext index -> cipher_idx (replaces q3_get_cipher_idx_for_plain_pos)
            std::vector<int> logicalToCipher;
            logicalToCipher.reserve(tile.text_len);
            for (int i = 0; i < tile.text_len; ++i)
            {
                if (!tile.mask[i]) continue;
                logicalToCipher.push_back((int)tile.cipher_idx[i]);
            }

            // 2) keyed dictionary cache by word_len (lazy-filled inside lambda)
            using KeyedWord = std::vector<int>;
            std::unordered_map<int, std::vector<KeyedWord>> dict_keyed_by_len;

            // 3) keyword prefix cache (first few letters, keyed space)
            constexpr int Q3_MAX_MISSING = 4; // safe upper bound for "missing" letters
            std::vector<std::array<std::uint8_t, Q3_MAX_MISSING>> key_prefix_keyed(ctx.keys_slice.size());
            std::vector<std::uint8_t> key_prefix_len(ctx.keys_slice.size(), 0);

            for (std::uint32_t wid = 0; wid < (std::uint32_t)ctx.keys_slice.size(); ++wid)
            {
                const auto& kw = ctx.keys_slice[wid];
                if (kw.empty())
                {
                    key_prefix_len[wid] = 0;
                    continue;
                }

                int len = (int)std::min<std::size_t>(kw.size(), (std::size_t)Q3_MAX_MISSING);
                std::uint8_t stored_len = 0;
                bool ok = true;

                for (int j = 0; j < len; ++j)
                {
                    char ch = kw[(std::size_t)j];
                    if (ch < 'A' || ch > 'Z') { ok = false; break; }

                    int idx = tile.index_map[ch - 'A']; // canonical -> keyed
                    if (idx < 0) { ok = false; break; }

                    key_prefix_keyed[wid][j] = (std::uint8_t)idx;
                    ++stored_len;
                }

                key_prefix_len[wid] = (ok ? stored_len : 0);
            }

            // helper: encode a short sequence of key indices into a 32-bit code
            auto encode_seq = [](const std::uint8_t* s, int len) -> std::uint32_t
            {
                std::uint32_t code = 0;
                for (int i = 0; i < len; ++i)
                    code |= (std::uint32_t(s[i] & 0x1F) << (5 * i));
                // length is implicitly "len" known by caller; if needed, you can
                // also tuck length into high bits.
                return code;
            };

            // Per-node constraints cache
            std::vector<std::vector<Q3NodeConstraints>> node_constraints(root_states.size());

            // ------------------------------------------------------------
            // Spacing/dictionary constraints for a given frontier node
            // ------------------------------------------------------------
            auto compute_constraints_for_node =
                [&](std::uint32_t root_idx,
                    std::uint32_t frontier_idx) -> const std::vector<std::uint32_t>*
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
                const int   preview_len = (int)node.preview_len;
                const int   key_len = (int)node.key_len;

                if (sp.empty() || preview_len <= 0)
                {
                    nc.state = Q3NodeAllowedState::AllWords;
                    return nullptr;
                }

                // --------------------------------------------------------
                // Find which spacing word is partially covered by key_len
                // --------------------------------------------------------
                int offset = 0;  // global plaintext letter index
                int slot = -1; // index in spacing pattern
                int covered_in_slot = 0;  // letters of that word already covered

                for (int wi = 0; wi < (int)sp.size(); ++wi)
                {
                    const int len = sp[wi];
                    const int end = offset + len;

                    if (key_len >= end)
                    {
                        offset = end; // fully covered this word
                        continue;
                    }

                    if (key_len > offset && key_len < end)
                    {
                        slot = wi;
                        covered_in_slot = key_len - offset;
                        break;
                    }

                    if (key_len <= offset)
                        break;
                }

                if (slot < 0 || covered_in_slot <= 0)
                {
                    // key ends on a word boundary or we're not inside a word
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
                // For longer gaps, fall back to "no constraint".
                if (missing > Q3_MAX_MISSING)
                {
                    nc.state = Q3NodeAllowedState::AllWords;
                    return nullptr;
                }

                const int required_preview = offset + covered_in_slot;
                if (required_preview > preview_len)
                {
                    // Not enough preview letters to reliably constrain
                    nc.state = Q3NodeAllowedState::AllWords;
                    return nullptr;
                }

                auto dict_it = dict_by_len.find(word_len);
                if (dict_it == dict_by_len.end())
                {
                    nc.state = Q3NodeAllowedState::AllWords;
                    return nullptr;
                }

                const auto& wordset = dict_it->second;

                // --------------------------------------------------------
                // Build keyed dictionary once per (tile, word_len)
                // --------------------------------------------------------
                auto& keyed_list = dict_keyed_by_len[word_len];
                if (keyed_list.empty())
                {
                    keyed_list.reserve(wordset.size());
                    for (const auto& w : wordset)
                    {
                        if ((int)w.size() != word_len)
                            continue;

                        KeyedWord plain;
                        plain.resize(word_len);
                        bool ok = true;
                        for (int i = 0; i < word_len; ++i)
                        {
                            char ch = w[(std::size_t)i];
                            if (ch < 'A' || ch > 'Z') { ok = false; break; }

                            int pi = tile.index_map[ch - 'A']; // canonical->keyed
                            if (pi < 0) { ok = false; break; }
                            plain[i] = pi;
                        }
                        if (ok)
                            keyed_list.push_back(std::move(plain));
                    }
                }

                // --------------------------------------------------------
                // Build set of allowed missing-letter key sequences
                // as encoded uint32_t
                // --------------------------------------------------------
                std::unordered_set<std::uint32_t> allowed_sequences;
                allowed_sequences.reserve(keyed_list.size());

                for (const auto& plain : keyed_list)
                {
                    // prefix of the spacing word must match node.preview
                    bool ok = true;

                    for (int k = 0; k < covered_in_slot; ++k)
                    {
                        int prev_bin = (int)node.preview[(std::size_t)(offset + k)];
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

                        const int c_idx = logicalToCipher[global_plain_pos];
                        if (c_idx < 0)
                        {
                            ok = false;
                            break;
                        }

                        const int p_idx = plain[covered_in_slot + j];

                        const int k_idx = q3_invert_decrypt_idx(c_idx, p_idx, mode);
                        if (k_idx < 0 || k_idx >= 26)
                        {
                            ok = false;
                            break;
                        }

                        seq[seq_len++] = (std::uint8_t)k_idx;
                    }

                    if (!ok || seq_len != missing)
                        continue;

                    std::uint32_t code = encode_seq(seq.data(), missing);
                    allowed_sequences.insert(code);
                }

                if (allowed_sequences.empty())
                {
                    // We had a valid spacing slot but no dictionary word can
                    // cohere with this node under this tile/mode => node is dead.
                    nc.state = Q3NodeAllowedState::Impossible;
                    return nullptr;
                }

                // --------------------------------------------------------
                // Filter keywords by whether they can realize any allowed sequence
                // --------------------------------------------------------
                nc.candidates.clear();

                for (std::uint32_t wid = 0; wid < (std::uint32_t)ctx.keys_slice.size(); ++wid)
                {
                    const auto& kw = ctx.keys_slice[wid];
                    if (kw.empty())
                        continue;

                    const int kw_len = (int)kw.size();

                    if (kw_len < missing)
                    {
                        // Be lenient with shorter words (same as before)
                        nc.candidates.push_back(wid);
                        continue;
                    }

                    const int prefix_len = key_prefix_len[wid];
                    if (prefix_len <= 0)
                        continue; // invalid prefix

                    if (prefix_len < missing)
                    {
                        // Valid prefix but shorter than missing; conservative choice:
                        // allow it (or you could skip it if you want stricter filtering).
                        nc.candidates.push_back(wid);
                        continue;
                    }

                    std::uint32_t cand_code =
                        encode_seq(key_prefix_keyed[wid].data(), missing);

                    if (allowed_sequences.find(cand_code) != allowed_sequences.end())
                    {
                        nc.candidates.push_back(wid);
                    }
                }

                if (nc.candidates.empty())
                {
                    nc.state = Q3NodeAllowedState::Impossible;
                    return nullptr;
                }

                std::sort(nc.candidates.begin(), nc.candidates.end());
                nc.candidates.erase(
                    std::unique(nc.candidates.begin(), nc.candidates.end()),
                    nc.candidates.end());

                nc.state = Q3NodeAllowedState::Filtered;
                return &nc.candidates;
            }; // compute_constraints_for_node

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
            //std::unique_ptr<DebugScopeTimer> t1 = std::make_unique<DebugScopeTimer>("rf loop");
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
                        compute_constraints_for_node(root_idx, frontier_idx);
                    
                    Q3NodeAllowedState st =
                        node_constraints[root_idx][frontier_idx].state;

                    if (st == Q3NodeAllowedState::Impossible)
                        continue; // dead node

                    if (st == Q3NodeAllowedState::Filtered && filtered)
                        candidates = filtered;
                }

                for (std::uint32_t next_id : *candidates)
                {
                    add_expansion(root_idx, base, next_id);
                    if (expanded_nodes.size() >= depth_cap)
                        break;
                }
            }
            //t1.reset();

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
                //(num_keys >= PhraseBuilder::MIN_GPU_KEYS_FOR_PHRASES) &&
                //(total_len >= PhraseBuilder::MIN_GPU_TOTAL_CHARS);

            GpuBatchResult phrase_gres;

            {
                //DebugScopeTimer t(use_gpu ? "phrase batch GPU" : "phrase batch CPU");

                if (use_gpu)
                {
                    phrase_gres =
                        launch_q3_gpu_megabatch(
                            group_tiles,
                            phrase_packed,
                            ctx.IOC_GATE,
                            ctx.CHI_GATE,
                            phrase_full_cap,
                            phrase_front_cap);
                }
                else
                {
                    const AlphabetCandidate& alphabet =
                        ctx.alphabets[alph_id]; // whatever your ctx holds

                    // CPU fallback for small batches
                    phrase_gres =
                        CPUDecode::launch_q3_cpu_phrase_megabatch(
                            tile,
                            alphabet,
                            alph_id,                         // <-- pass id here
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
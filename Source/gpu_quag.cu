#include "gpu_quag.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <cstring>

#define CUDA_OK(expr) do { cudaError_t _e = (expr); if (_e != cudaSuccess) \
  throw std::runtime_error(std::string("CUDA error: ")+cudaGetErrorString(_e)); } while(0)

namespace {
    constexpr int N = 96;           // padded length
    constexpr int PREVIEW_MAX = 80;
    constexpr int TPB = 256;

    constexpr int MAX_SPACING_WORDS = 16; // plenty for your sentence

    __constant__ int d_spacing_pattern[MAX_SPACING_WORDS];
    __constant__ int d_spacing_len; // number of valid entries in d_spacing_pattern


    __device__ double* g_q3_tri_table = nullptr;
    __device__ uint32_t g_q3_tri_table_size = 0;


    // 676 bits => 22 uint32_t; 26^4 = 456,976 bits => 14,281 uint32_t
    __constant__ uint32_t c_two_gram_bits[22];
    __constant__ uint32_t c_four_gram_bits[14281];

    __device__ inline bool test2(uint8_t a, uint8_t b) {
        const uint32_t idx = uint32_t(a) * 26u + uint32_t(b);
        return (c_two_gram_bits[idx >> 5] >> (idx & 31)) & 1u;
    }
    //__device__ inline bool test4(uint8_t a, uint8_t b, uint8_t c, uint8_t d) {
    //    uint32_t idx = (((uint32_t(a) * 26u + uint32_t(b)) * 26u + uint32_t(c)) * 26u + uint32_t(d));
    //    return (c_four_gram_bits[idx >> 5] >> (idx & 31)) & 1u;
    //}


    // ---- Dictionary of allowed words by length, encoded as base-26 codes ----
    //
    // We store ALL allowed words of lengths 2..6 in one flat array d_word_codes.
    // d_word_offsets[len] gives the starting index of words with that length,
    // and d_word_offsets[len+1] the end index.
    //
    // Example: length-3 words are in
    //   d_word_codes[d_word_offsets[3] .. d_word_offsets[4]-1]
    //
    // We use binary search on those segments for membership tests.
    //
    __device__ uint32_t* d_word_codes = nullptr;
    __constant__ int     d_word_offsets[8]; // indices for len=0..7 (we use 2..6)


    __device__ uint32_t* d_word_bits = nullptr;
    __constant__ uint32_t d_word_bit_offsets[8]; // bit offsets for len 0..7
    __constant__ int      d_word_bit_max_len;    // e.g. 5

    __device__ __forceinline__
        bool test_dictionary_bitset(int len, uint32_t code)
    {
        if (!d_word_bits) return false;
        if (len < 2 || len > d_word_bit_max_len) return false;

        uint32_t base_bit = d_word_bit_offsets[len];
        uint32_t bit_index = base_bit + code; // code in [0, 26^len)

        uint32_t word_idx = bit_index >> 5;   // /32
        uint32_t bit = bit_index & 31u;
        uint32_t mask = 1u << bit;

        return (d_word_bits[word_idx] & mask) != 0u;
    }

    // ---- Check spacing + words, mirroring validate_covered_words() ----
    //
    // preview[] holds plaintext alphabet indices (0..25) in order of decrypted letters.
    // preview_len = how many we actually filled
    // key_len     = key length in LETTERS (for the static key)
    //
    // Logic:
    //  - Only consider words in spacing_pattern that are FULLY covered by key_len
    //  - For each fully covered word:
    //      * if it's the FIRST word and length==2 -> check via test2() against the
    //        small starting-bigram dictionary
    //      * otherwise -> encode substring as base-26 and test in the large dictionary
    //  - Require at least ONE word to be fully covered and valid.
    //
    __device__ __forceinline__
        bool spacing_prefix_ok(const uint8_t* preview, int preview_len, int key_len)
    {
        if (d_spacing_len <= 0) {
            // No spacing info uploaded -> don't enforce word filter here
            return true;
        }

        const int letters = (key_len < preview_len ? key_len : preview_len);
        if (letters <= 0) return false;

        int  offset = 0;
        bool saw_any = false;

        for (int wi = 0; wi < d_spacing_len; ++wi)
        {
            const int word_len = d_spacing_pattern[wi];
            const int end = offset + word_len;

            // Not fully covered by key length or by decrypted prefix
            if (end > key_len || end > letters)
                break;

            // ---- SPECIAL CASE: first word, length == 2 -> use small starting dict ----
            if (wi == 0 && word_len == 2)
            {
                if (end > preview_len) return false; // should not happen, but be safe

                const uint8_t idx0 = preview[offset];
                const uint8_t idx1 = preview[offset + 1];

                if (idx0 >= 26 || idx1 >= 26) return false;

                // test2 uses the bigram bitset that was already uploaded from two_letter_set
                if (!test2(idx0, idx1))
                    return false;  // first 2 letters are NOT one of the allowed starters

                saw_any = true;
                offset = end;
                continue;  // go on to next word in spacing pattern
            }

            // ---- General dictionary path for all other words (including later 2-letter words) ----
            uint32_t code = 0;
            for (int i = offset; i < end; ++i)
            {
                const uint8_t idx = preview[i];
                if (idx >= 26) return false; // shouldn't happen
                code = code * 26u + uint32_t(idx);
            }

            if (!test_dictionary_bitset(word_len, code))
                return false;

            saw_any = true;
            offset = end;
        }

        return saw_any;
    }

    __constant__ float c_english_freq[26] = {
  0.08167f,0.01492f,0.02782f,0.04253f,0.12702f,0.02228f,0.02015f,
  0.06094f,0.06966f,0.00153f,0.00772f,0.04025f,0.02406f,0.06749f,
  0.07507f,0.01929f,0.00095f,0.05987f,0.06327f,0.09056f,0.02758f,
  0.00978f,0.02360f,0.00150f,0.01974f,0.00074f
    };

    __device__ inline uint8_t decrypt_idx(uint8_t c, uint8_t k, int mode) {
        int v;
        if (mode == 0) { v = c - k; if (v < 0) v += 26; }
        else if (mode == 1) { v = k - c; if (v < 0) v += 26; }
        else { v = c + k; if (v >= 26) v -= 26; }
        return (uint8_t)v;
    }

    struct DeviceKeyLetters {
        const uint8_t* key_chars_flat;  // letters 'A'..'Z'
        const uint32_t* key_offsets;
        const uint16_t* key_lengths;
        int num_keys;
    };

    struct DeviceOutputs {
        HitRecord* full_hits;
        uint32_t* full_count;
        HitRecord* front_hits;
        uint32_t* front_count;
        uint32_t   full_cap;   // NEW
        uint32_t   front_cap;  // NEW
    };

    __device__ __noinline__ double q3_score_trigram_english_device(
        const uint8_t* text,
        int len)
    {
        // With log(count) scoring and a dense table, we don’t really need a floor.
        if (len < 3)
            return 0.0;

        double* tri_table = g_q3_tri_table;
        uint32_t size = g_q3_tri_table_size;

        if (!tri_table || size == 0)
            return 0.0;

        double score = 0.0;
        int    trigrams = 0;

        // seed with first two letters
        uint32_t code = static_cast<uint32_t>(text[0]) * 26u
            + static_cast<uint32_t>(text[1]);

        constexpr uint32_t BASE_26_2 = 26u * 26u;

        for (int i = 2; i < len; ++i)
        {
            // slide in the next letter -> 3-letter code in [0, 26^3)
            code = code * 26u + static_cast<uint32_t>(text[i]);

            if (code < size)
                score += tri_table[code];

            ++trigrams;

            // keep only the last 2 letters for next step
            code %= BASE_26_2;
        }

        if (trigrams == 0)
            return 0.0;

        return score;
    }



    __device__ bool autokey_window_filter(
        const uint8_t* __restrict__ cipher_idx, // s_cipher (canonical 0..25 indices)
        const uint8_t* __restrict__ mask,       // s_mask
        int text_len,
        const uint8_t* __restrict__ primer,     // canonical 0..25
        int primer_len,
        const char* __restrict__ alph,          // s_alph
        const int8_t* __restrict__ index_map,   // s_index_map
        uint8_t mode)
    {
        // If we have no primer or no trigram table, don't filter at all
        if (primer_len <= 0)
            return true;

        if (!g_q3_tri_table || g_q3_tri_table_size == 0)
            return true;

        constexpr int AUTOKEY_BUFFER_MAX = 42;  // max plaintext we hold per attempt
        constexpr int AUTOKEY_MAX_OFFSETS = 16;  // how many starting positions to test
        constexpr int AUTOKEY_MIN_WINDOW = 6;   // minimum window length to bother scoring

        // Base threshold: tuned around some "typical" primer length
        constexpr int    AUTOKEY_TARGET_WIN = 16;   // window length where base threshold applies
        constexpr double AUTOKEY_TRI_BASE = 11.5; // your previous tuned trigram avg
        constexpr double AUTOKEY_TRI_RELAX = 0.0;  // how much more lenient for very short windows

        // Window length = primer length, clamped to buffer
        int windowLen = primer_len;
        if (windowLen > AUTOKEY_BUFFER_MAX)
            windowLen = AUTOKEY_BUFFER_MAX;

        if (windowLen < AUTOKEY_MIN_WINDOW)
            return true; // not enough letters to get a stable trigram score

        // Dynamic threshold based on windowLen (shorter windows -> more lenient)
        int effWin = windowLen;
        if (effWin < AUTOKEY_MIN_WINDOW) effWin = AUTOKEY_MIN_WINDOW;
        if (effWin > AUTOKEY_TARGET_WIN) effWin = AUTOKEY_TARGET_WIN;

        double t = double(effWin - AUTOKEY_MIN_WINDOW)
            / double(AUTOKEY_TARGET_WIN - AUTOKEY_MIN_WINDOW);
        if (t < 0.0) t = 0.0;
        if (t > 1.0) t = 1.0;

        const double dynamic_thresh =
            AUTOKEY_TRI_BASE - (1.0 - t) * AUTOKEY_TRI_RELAX;

        // Where autokey can start in the ciphertext:
        // conceptually "right after the primer" region of the static key.
        constexpr int AUTOKEY_SKIP = 0;
        const int baseStart = primer_len + AUTOKEY_SKIP;
        if (baseStart >= text_len)
            return true; // nowhere to test

        // Try several possible autokey starting offsets
        for (int offset = 0; offset <= AUTOKEY_MAX_OFFSETS; ++offset)
        {
            int start = baseStart + offset;
            if (start >= text_len)
                break;

            // Decode exactly windowLen plaintext letters (or bail if we can't)
            uint8_t plain[AUTOKEY_BUFFER_MAX];
            int     plain_len = 0;

            // We walk forward in the ciphertext from this start, skipping masked positions.
            // Keystream for this LOCAL attempt:
            //  - first primer_len positions: primer[0..primer_len-1]
            //  - further positions (if any): autokey from plain[]
            for (int i = start; i < text_len && plain_len < windowLen; ++i)
            {
                if (!mask[i])
                    continue; // masked positions produce no plaintext, no autokey advance

                int j = plain_len; // position in the LOCAL keystream for this attempt

                int key_canon;
                if (j < primer_len)
                {
                    // Seed region: primer supplies keystream
                    key_canon = primer[j];
                }
                else
                {
                    // Autokey region: use previously produced plaintext
                    int idx = j - primer_len;
                    if (idx < 0 || idx >= plain_len)
                    {
                        // Not enough plaintext to continue autokey
                        plain_len = 0; // clear so we fail this offset cleanly
                        break;
                    }
                    key_canon = plain[idx];
                }

                int kix = key_canon; // 0..25
                int km = ((unsigned)kix < 26u) ? int(index_map[kix]) : -1;
                if (km < 0) km = 0;

                uint8_t p_idx = decrypt_idx(cipher_idx[i], (uint8_t)km, (int)mode);
                char    ch = alph[p_idx];
                int     bin = int(ch) - 'A';
                if ((unsigned)bin >= 26u)
                    continue; // non A-Z, skip but keep trying to fill the window

                plain[plain_len++] = (uint8_t)bin;
            }

            // If we couldn't fill the full window, this offset is inconclusive: try next.
            if (plain_len < windowLen)
                continue;

            // Score this LOCAL window of length windowLen
            const double score = q3_score_trigram_english_device(plain, windowLen);
            const double avg = score / double(windowLen - 2); // avg per trigram

            if (avg >= dynamic_thresh)
            {
                // This offset produced a plausible autokey plaintext  keep this candidate
                return true;
            }

            // else: bad at this offset  try the next offset
        }

        // All tested offsets looked like garbage
        return false;
    }







    __global__ void q3_kernel(const DeviceAlphabetTile* __restrict__ tiles, int num_tiles,
        DeviceKeyLetters keys,
        float ioc_gate, float chi_gate,
        DeviceOutputs out)
    {
        const int tile_id = blockIdx.x;
        if (tile_id >= num_tiles) return;

        // ---- Stage tile into shared memory ----
        __shared__ uint8_t  s_cipher[N];
        __shared__ uint8_t  s_mask[N];
        __shared__ char     s_alph[26];
        __shared__ int8_t   s_index_map[26];
        __shared__ int      s_text_len;
        __shared__ uint32_t s_alphabet_id;
        __shared__ uint8_t  s_mode;

        // ---- Early first-6 filter state (from old kernel) ----
        uint8_t first6[6];
        int     produced = 0;


        if (threadIdx.x < N) {
            s_cipher[threadIdx.x] = tiles[tile_id].cipher_idx[threadIdx.x];
            s_mask[threadIdx.x] = tiles[tile_id].mask[threadIdx.x];
        }
        if (threadIdx.x < 26) {
            s_alph[threadIdx.x] = tiles[tile_id].alph[threadIdx.x];
            s_index_map[threadIdx.x] = tiles[tile_id].index_map[threadIdx.x];
        }
        if (threadIdx.x == 0) {
            s_text_len = tiles[tile_id].text_len;
            s_alphabet_id = tiles[tile_id].alphabet_id;
            s_mode = tiles[tile_id].mode;
        }
        __syncthreads();

        // ---- Key assignment ----
        const int key_global_idx = blockIdx.y * blockDim.x + threadIdx.x;
        if (key_global_idx >= keys.num_keys) return;

        const uint32_t off = keys.key_offsets[key_global_idx];
        int klen = int(keys.key_lengths[key_global_idx]);
        if (klen <= 0) return;

        // ---- Per-thread histogram in shared memory (26 bins per thread) ----
        __shared__ uint32_t s_hist[26 * TPB];
        uint32_t* my_hist = &s_hist[threadIdx.x * 26];

#pragma unroll
        for (int i = 0; i < 26; ++i) my_hist[i] = 0u;
        __syncthreads();

        // ---- Streaming decrypt ----
        uint8_t preview[PREVIEW_MAX];
        int     preview_fill = 0;

        int ki = 0; // rolling key index (no modulo in loop body)

#pragma unroll
#pragma unroll
        for (int i = 0; i < N; ++i)
        {
            if (i >= s_text_len) break;
            if (!s_mask[i])      continue;

            const uint8_t kch = __ldg(&keys.key_chars_flat[off + ki]);
            ++ki; if (ki == klen) ki = 0;

            const int kix = int(kch) - 'A';
            int km = ((unsigned)kix < 26u) ? int(s_index_map[kix]) : -1;
            if (km < 0) km = 0;

            const uint8_t p_idx = decrypt_idx(s_cipher[i], uint8_t(km), int(s_mode));
            const char    ch = s_alph[p_idx];
            const int     bin = int(ch) - 'A';

            // canonical 0..25 index for *plaintext letter* (as in old first6 logic)
            uint8_t canonical = (uint8_t)((bin < 0) ? 0 : (bin > 25 ? 25 : bin));

            // preview for host (now using canonical rather than keyed index)
            if (preview_fill < PREVIEW_MAX) {
                preview[preview_fill++] = canonical;
            }

            // ---- OLD early reject logic: first 2 and then quad in first 6 letters ----
            if (produced < 6) {
                first6[produced++] = canonical;

                if (produced == 2) {
                    if (!test2(first6[0], first6[1])) {
                        // first bigram not allowed => kill key early
                        return;
                    }
                }
                //else if (produced == 6) {
                //    if (!test4(first6[2], first6[3], first6[4], first6[5])) {
                //        // second word quadgram not allowed => kill key early
                //        return;
                //    }
                //}
            }

            // histogram
            if ((unsigned)bin < 26u) {
                my_hist[bin] += 1u;
            }
        }

        if (preview_fill < 2) return; // no real prefix

        // ---- NEW: spacing/dictionary word filter on decrypted prefix ----
        //
        // This enforces:
        //  - only words fully covered by key length (and decrypted prefix) are checked
        //  - all those covered words must be in the sentence dictionary
        //  - at least one word must be fully covered and valid
        //
        if (!spacing_prefix_ok(preview, preview_fill, klen))
            return;

        // ---- Compute IoC / Chi² from my_hist ----
        int   n = 0;
        int   num = 0;
#pragma unroll
        for (int i = 0; i < 26; ++i) {
            const int c = int(my_hist[i]);
            n += c;
            num += c * (c - 1);
        }
        if (n <= 1) return;

        float chi = 0.f;
#pragma unroll
        for (int i = 0; i < 26; ++i) {
            const float expc = c_english_freq[i] * float(n);
            if (expc > 0.f) {
                const float diff = float(my_hist[i]) - expc;
                chi += (diff * diff) / expc;
            }
        }
        const float ioc = float(num) / (float(n) * float(n - 1));
        const bool stats_ok = (ioc > ioc_gate && chi < chi_gate);

        // ---- Safe write with capacity guard; copy only preview_len bytes ----
        auto write_rec = [&](HitRecord& rec, uint8_t flags) {
            rec.key_id = (uint32_t)key_global_idx;
            rec.alphabet_id = s_alphabet_id;
            rec.mode = s_mode;
            rec.flags = flags;
            rec.ioc = stats_ok ? ioc : 0.f;
            rec.chi = stats_ok ? chi : 0.f;
            rec.preview_len = (uint8_t)preview_fill;
#pragma unroll
            for (int i = 0; i < PREVIEW_MAX; ++i) {
                if (i < preview_fill) rec.preview[i] = preview[i];
                else break;
            }
        };

        bool autokey_ok = true;
        if (!stats_ok) 
        {
            // primer is the first klen chars of preview, but preview might be shorter
            int primer_len = klen;
            if (primer_len > preview_fill) primer_len = preview_fill;

            autokey_ok = autokey_window_filter(
                s_cipher,
                s_mask,
                s_text_len,
                preview,          // canonical 0..25 plaintext from static pass
                primer_len,
                s_alph,
                s_index_map,
                s_mode);

            if (!autokey_ok)
                return; // reject this key entirely (never write to front_hits)
        }

        if (stats_ok) {
            uint32_t slot = atomicAdd(out.full_count, 1u);
            if (slot < out.full_cap) {
                write_rec(out.full_hits[slot], (HIT_FRONT_OK | HIT_STATS_OK));
            }
        }
        else {
            uint32_t slot = atomicAdd(out.front_count, 1u);
            if (slot < out.front_cap) {
                write_rec(out.front_hits[slot], HIT_FRONT_OK);
            }
        }
    }



} // anon

// ---------- helpers ----------
void build_bigram_bitset(const std::vector<uint16_t>& codes, std::vector<uint32_t>& out_bits) {
    out_bits.assign(22, 0);
    for (auto c : codes) {
        uint32_t idx = c;
        out_bits[idx >> 5] |= (1u << (idx & 31));
    }
}
void build_quadgram_bitset(const std::vector<uint32_t>& codes, std::vector<uint32_t>& out_bits) {
    out_bits.assign(14281, 0);
    for (auto c : codes) {
        uint32_t idx = c;
        out_bits[idx >> 5] |= (1u << (idx & 31));
    }
}

void gpu_upload_gram_bitsets(const std::vector<uint32_t>& two_bits,
    const std::vector<uint32_t>& four_bits)
{
    if (two_bits.size() != 22 || four_bits.size() != 14281)
        throw std::runtime_error("Bitset size mismatch");
    CUDA_OK(cudaMemcpyToSymbol(c_two_gram_bits, two_bits.data(), 22 * sizeof(uint32_t), 0, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpyToSymbol(c_four_gram_bits, four_bits.data(), 14281 * sizeof(uint32_t), 0, cudaMemcpyHostToDevice));
}

void gpu_upload_spacing_and_words(const std::vector<int>& spacing_pattern, const WordCodeTable& table)
{
    // ---- upload spacing pattern ----
    int h_pattern[MAX_SPACING_WORDS] = { 0 };
    int len = (int)std::min<std::size_t>(spacing_pattern.size(), MAX_SPACING_WORDS);
    for (int i = 0; i < len; ++i)
        h_pattern[i] = spacing_pattern[i];

    CUDA_OK(cudaMemcpyToSymbol(
        d_spacing_pattern,
        h_pattern,
        sizeof(int) * MAX_SPACING_WORDS,
        0,
        cudaMemcpyHostToDevice));

    CUDA_OK(cudaMemcpyToSymbol(
        d_spacing_len,
        &len,                       // <-- pointer
        sizeof(int),
        0,
        cudaMemcpyHostToDevice));

    // ---- upload word codes ----
    uint32_t* d_codes = nullptr;
    if (!table.codes.empty()) {
        const size_t bytes = table.codes.size() * sizeof(uint32_t);
        CUDA_OK(cudaMalloc(&d_codes, bytes));
        CUDA_OK(cudaMemcpy(d_codes,
            table.codes.data(),
            bytes,
            cudaMemcpyHostToDevice));
    }

    CUDA_OK(cudaMemcpyToSymbol(
        d_word_codes,
        &d_codes,                   // pointer to device pointer
        sizeof(uint32_t*),
        0,
        cudaMemcpyHostToDevice));

    CUDA_OK(cudaMemcpyToSymbol(
        d_word_offsets,
        table.offsets.data(),
        table.offsets.size() * sizeof(int),
        0,
        cudaMemcpyHostToDevice));
}

void gpu_upload_word_bitsets(const std::vector<int>& spacing_pattern,
    const WordBitsetTable& table,
    int max_len)
{
    // ---- upload spacing pattern ----
    int h_pattern[MAX_SPACING_WORDS] = { 0 };
    int len = (int)std::min<std::size_t>(spacing_pattern.size(), MAX_SPACING_WORDS);
    for (int i = 0; i < len; ++i)
        h_pattern[i] = spacing_pattern[i];

    CUDA_OK(cudaMemcpyToSymbol(
        d_spacing_pattern,
        h_pattern,
        sizeof(int) * MAX_SPACING_WORDS,
        0,
        cudaMemcpyHostToDevice));

    CUDA_OK(cudaMemcpyToSymbol(
        d_spacing_len,
        &len,
        sizeof(int),
        0,
        cudaMemcpyHostToDevice));

    // ---- upload bitset table ----
    uint32_t* d_bits = nullptr;
    if (!table.bits.empty())
    {
        const size_t bytes = table.bits.size() * sizeof(uint32_t);
        CUDA_OK(cudaMalloc(&d_bits, bytes));
        CUDA_OK(cudaMemcpy(d_bits,
            table.bits.data(),
            bytes,
            cudaMemcpyHostToDevice));
    }

    CUDA_OK(cudaMemcpyToSymbol(
        d_word_bits,
        &d_bits,
        sizeof(uint32_t*),
        0,
        cudaMemcpyHostToDevice));

    CUDA_OK(cudaMemcpyToSymbol(
        d_word_bit_offsets,
        table.offsets.data(),
        table.offsets.size() * sizeof(uint32_t),
        0,
        cudaMemcpyHostToDevice));

    CUDA_OK(cudaMemcpyToSymbol(
        d_word_bit_max_len,
        &max_len,
        sizeof(int),
        0,
        cudaMemcpyHostToDevice));
}

void gpu_upload_q3_trigram_table(const std::vector<double>& hostTable)
{
    double* d_ptr = nullptr;
    uint32_t size32 = static_cast<uint32_t>(hostTable.size());

    if (size32 > 0)
    {
        CUDA_OK(cudaMalloc(&d_ptr, size32 * sizeof(double)));
        CUDA_OK(cudaMemcpy(d_ptr,
            hostTable.data(),
            size32 * sizeof(double),
            cudaMemcpyHostToDevice));
    }

    // Publish pointer + size to device globals
    CUDA_OK(cudaMemcpyToSymbol(g_q3_tri_table, &d_ptr, sizeof(d_ptr)));
    CUDA_OK(cudaMemcpyToSymbol(g_q3_tri_table_size, &size32, sizeof(size32)));
}



GpuBatchResult launch_q3_gpu_megabatch(const std::vector<DeviceAlphabetTile>& tiles,
    const PackedKeysHostLetters& packed_letters,
    float ioc_gate, float chi_gate,
    std::size_t max_full_hits,
    std::size_t max_front_hits)
{
    if (tiles.empty() || packed_letters.num_keys() == 0) return {};

    // Device tiles
    DeviceAlphabetTile* d_tiles = nullptr;
    CUDA_OK(cudaMalloc(&d_tiles, tiles.size() * sizeof(DeviceAlphabetTile)));
    CUDA_OK(cudaMemcpy(d_tiles, tiles.data(), tiles.size() * sizeof(DeviceAlphabetTile),
        cudaMemcpyHostToDevice));

    // Device keys (letters)
    uint8_t* d_key_chars = nullptr;
    uint32_t* d_key_off = nullptr;
    uint16_t* d_key_len = nullptr;

    CUDA_OK(cudaMalloc(&d_key_chars, packed_letters.key_chars_flat.size() * sizeof(uint8_t)));
    CUDA_OK(cudaMemcpy(d_key_chars, packed_letters.key_chars_flat.data(),
        packed_letters.key_chars_flat.size() * sizeof(uint8_t), cudaMemcpyHostToDevice));

    CUDA_OK(cudaMalloc(&d_key_off, packed_letters.key_offsets.size() * sizeof(uint32_t)));
    CUDA_OK(cudaMemcpy(d_key_off, packed_letters.key_offsets.data(),
        packed_letters.key_offsets.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));

    CUDA_OK(cudaMalloc(&d_key_len, packed_letters.key_lengths.size() * sizeof(uint16_t)));
    CUDA_OK(cudaMemcpy(d_key_len, packed_letters.key_lengths.data(),
        packed_letters.key_lengths.size() * sizeof(uint16_t), cudaMemcpyHostToDevice));

    // Device outputs
    HitRecord* d_full = nullptr;  uint32_t* d_full_count = nullptr;
    HitRecord* d_front = nullptr;  uint32_t* d_front_count = nullptr;
    CUDA_OK(cudaMalloc(&d_full, max_full_hits * sizeof(HitRecord)));
    CUDA_OK(cudaMalloc(&d_front, max_front_hits * sizeof(HitRecord)));
    CUDA_OK(cudaMalloc(&d_full_count, sizeof(uint32_t)));
    CUDA_OK(cudaMalloc(&d_front_count, sizeof(uint32_t)));
    CUDA_OK(cudaMemset(d_full_count, 0, sizeof(uint32_t)));
    CUDA_OK(cudaMemset(d_front_count, 0, sizeof(uint32_t)));

    DeviceKeyLetters kb{ d_key_chars, d_key_off, d_key_len, (int)packed_letters.num_keys() };
    DeviceOutputs out{
        d_full, d_full_count,
        d_front, d_front_count,
        (uint32_t)max_full_hits,
        (uint32_t)max_front_hits
    };

    const int num_tiles = (int)tiles.size();
    dim3 grid(num_tiles, (packed_letters.num_keys() + TPB - 1) / TPB);
    dim3 block(TPB);

    q3_kernel << <grid, block >> > (d_tiles, num_tiles, kb, ioc_gate, chi_gate, out);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaDeviceSynchronize());

    // Fetch counts
    uint32_t h_full = 0, h_front = 0;
    CUDA_OK(cudaMemcpy(&h_full, d_full_count, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(&h_front, d_front_count, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    if (h_full > max_full_hits)  h_full = (uint32_t)max_full_hits;
    if (h_front > max_front_hits) h_front = (uint32_t)max_front_hits;

    // Fetch records
    std::vector<HitRecord> full(h_full), front(h_front);
    if (h_full)  CUDA_OK(cudaMemcpy(full.data(), d_full, h_full * sizeof(HitRecord), cudaMemcpyDeviceToHost));
    if (h_front) CUDA_OK(cudaMemcpy(front.data(), d_front, h_front * sizeof(HitRecord), cudaMemcpyDeviceToHost));

    // Cleanup
    cudaFree(d_tiles);
    cudaFree(d_key_chars);
    cudaFree(d_key_off);
    cudaFree(d_key_len);
    cudaFree(d_full);
    cudaFree(d_front);
    cudaFree(d_full_count);
    cudaFree(d_front_count);

    return { std::move(full), std::move(front) };
}

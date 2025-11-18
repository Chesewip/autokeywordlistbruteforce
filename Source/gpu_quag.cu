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


    // 676 bits => 22 uint32_t; 26^4 = 456,976 bits => 14,281 uint32_t
    __constant__ uint32_t c_two_gram_bits[22];
    __constant__ uint32_t c_four_gram_bits[14281];

    __device__ inline bool test2(uint8_t a, uint8_t b) {
        const uint32_t idx = uint32_t(a) * 26u + uint32_t(b);
        return (c_two_gram_bits[idx >> 5] >> (idx & 31)) & 1u;
    }
    __device__ inline bool test4(uint8_t a, uint8_t b, uint8_t c, uint8_t d) {
        uint32_t idx = (((uint32_t(a) * 26u + uint32_t(b)) * 26u + uint32_t(c)) * 26u + uint32_t(d));
        return (c_four_gram_bits[idx >> 5] >> (idx & 31)) & 1u;
    }


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

    // ---- Binary search helper ----
    __device__ __forceinline__
        bool binary_search_word(uint32_t code, int start, int end, const uint32_t* codes)
    {
        int lo = start;
        int hi = end - 1;
        while (lo <= hi)
        {
            int mid = (lo + hi) >> 1;
            uint32_t v = codes[mid];
            if (v == code) return true;
            if (v < code)  lo = mid + 1;
            else           hi = mid - 1;
        }
        return false;
    }

    // ---- Lookup "does this length-L code exist in the dictionary?" ----
    __device__ __forceinline__
        bool test_dictionary_word(int len, uint32_t code)
    {
        if (!d_word_codes) return false;      // no table loaded
        if (len < 2 || len > 6) return false; // we only support 2..6

        int start = d_word_offsets[len];
        int end = d_word_offsets[len + 1];
        if (start >= end) return false;       // no words of this length

        return binary_search_word(code, start, end, d_word_codes);
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

            if (!test_dictionary_word(word_len, code))
                return false; // a fully covered word is NOT in dictionary

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

            if (preview_fill < PREVIEW_MAX) {
                uint8_t canonical = (uint8_t)((bin < 0) ? 0 : (bin > 25 ? 25 : bin));
                preview[preview_fill++] = canonical;
            }

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

void gpu_upload_spacing_and_words(const std::vector<int>& spacing_pattern,
    const WordCodeTable& table)
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

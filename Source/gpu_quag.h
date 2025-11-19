#pragma once
#include <cstdint>
#include <vector>
#include <string>
#include <array>

namespace {
    // Must match __constant__ N in gpu_quag.cu
    static constexpr int GPU_PLAINTEXT_STRIDE = 96;
}

// One (alphabet, mode) tile worth of data (padded to 96)
struct DeviceAlphabetTile {
    // per-alphabet, per-mode small tile (96 padded)
    std::uint8_t cipher_idx[96];
    std::uint8_t mask[96];
    char         alph[26];      // index -> ASCII letter (keyed alphabet)
    std::int8_t  index_map[26]; // ASCII 'A'..'Z' -> index (0..25), -1 if absent
    int          text_len;
    std::uint32_t alphabet_id;  // absolute index in alphabets vector
    std::uint8_t  mode;         // 0=Vig,1=Beaufort,2=VariantBeaufort
};

// flags returned by kernel
enum : std::uint8_t {
    HIT_FRONT_OK = 1u << 0,    // passed 2-gram/4-gram
    HIT_STATS_OK = 1u << 1     // passed IoC/chi gates
};

struct HitRecord {
    std::uint32_t key_id;
    std::uint32_t alphabet_id;
    std::uint8_t  mode;
    std::uint8_t  flags;       // HIT_FRONT_OK | HIT_STATS_OK
    float         ioc;         // valid iff HIT_STATS_OK
    float         chi;         // valid iff HIT_STATS_OK
    std::uint8_t  preview_len; // optional; can be 0
    std::uint8_t  preview[80]; // plaintext indices in keyed space (optional)
};

struct GpuBatchResult {
    std::vector<HitRecord> full_hits;   // front+stats
    std::vector<HitRecord> front_hits;  // front only (stats failed)
};

struct PackedKeysHostLetters {
    std::vector<std::uint8_t>  key_chars_flat; // raw letters 'A'..'Z'
    std::vector<std::uint32_t> key_offsets;
    std::vector<std::uint16_t> key_lengths;
    std::size_t num_keys() const { return key_lengths.size(); }
};

struct WordCodeTable
{
    std::vector<uint32_t> codes;         // concatenated; sorted within each length
    std::array<int, 8>    offsets{};     // offsets[len]..offsets[len+1]-1 = that length's words
};

struct WordBitsetTable
{
    // offsets[len] gives the starting bit index for words of that length (in bits).
    // offsets[len+1] is the end.
    std::array<uint32_t, 8> offsets{}; // we use lengths 2..5 or 6

    // Flat bitvector over all lengths (2..MAX_LEN). Bit i == 1 if code i is present.
    std::vector<uint32_t> bits; // 32 bits per entry
};

// Upload prebuilt bigram/quadgram bitsets to constant/device memory (your existing impl)
void build_bigram_bitset(const std::vector<std::uint16_t>& codes,
    std::vector<std::uint32_t>& bitset_out);
void build_quadgram_bitset(const std::vector<std::uint32_t>& codes,
    std::vector<std::uint32_t>& bitset_out);
void gpu_upload_gram_bitsets(const std::vector<std::uint32_t>& bigram_bits,
    const std::vector<std::uint32_t>& quad_bits);

void gpu_upload_spacing_and_words(const std::vector<int>& spacing_pattern, const WordCodeTable& table);

void gpu_upload_word_bitsets(const std::vector<int>& spacing_pattern, const WordBitsetTable& table, int max_len);

void gpu_upload_q3_trigram_table(const std::vector<double>& hostTable);

// New mega-batch launcher (one launch for many alphabets × all 3 modes)
GpuBatchResult launch_q3_gpu_megabatch(const std::vector<DeviceAlphabetTile>& tiles,
    const PackedKeysHostLetters& packed_letters,
    float ioc_gate, float chi_gate,
    std::size_t max_full_hits,
    std::size_t max_front_hits);

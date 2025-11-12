#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// Shared cipher mode enumeration used by both the CPU and GPU paths.
enum class Mode : int {
  kVigenere = 0,
  kBeaufort = 1,
  kVariantBeaufort = 2,
};

// Representation of a keyed alphabet candidate.
struct AlphabetCandidate {
  std::string alphabet;
  std::array<int, 26> index_map{};
  std::string base_word;
  bool keyword_reversed = false;
  bool alphabet_reversed = false;
  bool keyword_front = true;
};

// Resulting statistics for a plaintext computed on the GPU.
struct GpuPlainStats {
  double ioc = 0.0;
  double chi = 0.0;
  double score = 0.0;
  int letters = 0;
  std::uint8_t valid = 0;
  std::uint8_t reserved[3] = {0, 0, 0};
};

struct GpuBatchResult {
  bool success = false;
  std::string error_message;
  std::size_t batch_size = 0;
  std::vector<Mode> modes;
  std::vector<GpuPlainStats> stats;
};

// Returns true when at least one CUDA device is available.
bool gpu_is_available();

// Launches the static-key GPU kernel for a batch of repeating-key candidates.
GpuBatchResult launch_static_gpu(
    const AlphabetCandidate &alphabet,
    const std::vector<Mode> &modes,
    const std::vector<std::uint8_t> &cipher_indices,
    const std::vector<std::uint8_t> &letter_mask,
    const std::uint8_t *key_schedules,
    const std::uint8_t *key_schedule_valid,
    const std::uint8_t *key_invalid_flags,
    std::size_t text_len,
    std::size_t padded_len,
    std::size_t batch_size);

// Convenience overload for single-mode launches.
GpuBatchResult launch_static_gpu(
    const AlphabetCandidate &alphabet,
    Mode mode,
    const std::vector<std::uint8_t> &cipher_indices,
    const std::vector<std::uint8_t> &letter_mask,
    const std::uint8_t *key_schedules,
    const std::uint8_t *key_schedule_valid,
    const std::uint8_t *key_invalid_flags,
    std::size_t text_len,
    std::size_t padded_len,
    std::size_t batch_size);


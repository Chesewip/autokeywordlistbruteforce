#include "gpu_quag.h"

#include <cuda_runtime.h>
#include <math_constants.h>

#include <cmath>
#include <string>
#include <vector>

namespace {

constexpr double kIocTarget = 0.066;

__device__ __forceinline__ std::uint8_t decrypt_symbol_device(
    std::uint8_t cipher_idx,
    std::uint8_t key_idx,
    Mode mode) {
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

__device__ __forceinline__ double compute_ioc_device(
    const int counts[26], int total_letters) {
  if (total_letters <= 1) {
    return 0.0;
  }
  double numerator = 0.0;
  for (int i = 0; i < 26; ++i) {
    const double c = static_cast<double>(counts[i]);
    numerator += c * (c - 1.0);
  }
  const double denom = static_cast<double>(total_letters) *
                       static_cast<double>(total_letters - 1);
  return (denom == 0.0) ? 0.0 : numerator / denom;
}

__constant__ double kEnglishFreqDevice[26] = {
    0.08167, 0.01492, 0.02782, 0.04253, 0.12702, 0.02228, 0.02015,
    0.06094, 0.06966, 0.00153, 0.00772, 0.04025, 0.02406, 0.06749,
    0.07507, 0.01929, 0.00095, 0.05987, 0.06327, 0.09056, 0.02758,
    0.00978, 0.02360, 0.00150, 0.01974, 0.00074};

__device__ __forceinline__ double compute_chi_device(
    const int counts[26], int total_letters) {
  if (total_letters <= 0) {
    return CUDART_INF;
  }
  double chi = 0.0;
  const double n = static_cast<double>(total_letters);
  for (int i = 0; i < 26; ++i) {
    const double observed = static_cast<double>(counts[i]);
    const double expected = n * kEnglishFreqDevice[i];
    if (expected > 0.0) {
      const double diff = observed - expected;
      chi += (diff * diff) / expected;
    }
  }
  return chi;
}

__device__ __forceinline__ double compute_stats_score(double ioc, double chi) {
  const double ioc_delta = fabs(ioc - kIocTarget);
  const double ioc_score = fmax(0.0, 1.0 - ioc_delta / 0.02);
  const double chi_clamped = fmin(400.0, chi);
  const double chi_score = fmax(0.0, 1.0 - chi_clamped / 400.0);
  const double quality_factor = 0.1 + 0.9 * chi_score;
  return ioc_score * quality_factor;
}

__global__ void quag_static_kernel(
    const std::uint8_t *cipher_indices,
    const std::uint8_t *letter_mask,
    std::size_t text_len,
    std::size_t padded_len,
    const std::uint8_t *key_schedules,
    const std::uint8_t *key_schedule_valid,
    std::size_t schedule_stride,
    const std::uint8_t *key_invalid_flags,
    const char *alphabet,
    std::size_t alphabet_len,
    const Mode *modes,
    std::size_t mode_count,
    std::size_t batch_size,
    GpuPlainStats *out_stats) {
  const std::size_t key_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (key_idx >= batch_size) {
    return;
  }

  (void)padded_len;
  const std::uint8_t *schedule = key_schedules + key_idx * schedule_stride;
  const std::uint8_t *schedule_valid =
      key_schedule_valid + key_idx * schedule_stride;
  const std::uint8_t invalid_flag = key_invalid_flags[key_idx];

  for (std::size_t mode_index = 0; mode_index < mode_count; ++mode_index) {
    const Mode mode = modes[mode_index];
    int counts[26];
    for (int i = 0; i < 26; ++i) {
      counts[i] = 0;
    }
    int total_letters = 0;

    for (std::size_t pos = 0; pos < text_len; ++pos) {
      if (!letter_mask[pos]) {
        continue;
      }
      if (!schedule_valid[pos]) {
        continue;
      }
      const std::uint8_t cipher_idx = cipher_indices[pos];
      const std::uint8_t key_idx_local = schedule[pos];
      const std::uint8_t plain_idx =
          decrypt_symbol_device(cipher_idx, key_idx_local, mode);
      if (plain_idx >= alphabet_len) {
        continue;
      }
      const char plain_char = alphabet[plain_idx];
      if (plain_char >= 'A' && plain_char <= 'Z') {
        counts[plain_char - 'A'] += 1;
        total_letters += 1;
      }
    }

    GpuPlainStats stat{};
    stat.letters = total_letters;
    const bool has_letters = (total_letters > 1) && (invalid_flag == 0);
    if (has_letters) {
      stat.ioc = compute_ioc_device(counts, total_letters);
      stat.chi = compute_chi_device(counts, total_letters);
      stat.score = compute_stats_score(stat.ioc, stat.chi);
      stat.valid = 1;
    } else {
      stat.ioc = 0.0;
      stat.chi = CUDART_INF;
      stat.score = 0.0;
      stat.valid = 0;
    }

    const std::size_t out_index = key_idx * mode_count + mode_index;
    out_stats[out_index] = stat;
  }
}

} // namespace

bool gpu_is_available() {
  int count = 0;
  cudaError_t err = cudaGetDeviceCount(&count);
  if (err != cudaSuccess) {
    return false;
  }
  return count > 0;
}

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
    std::size_t batch_size) {
  GpuBatchResult result;
  result.batch_size = batch_size;
  result.modes = modes;
  if (batch_size == 0 || modes.empty()) {
    result.success = true;
    return result;
  }
  if (cipher_indices.size() < padded_len || letter_mask.size() < text_len) {
    result.error_message = "cipher or letter mask size mismatch";
    return result;
  }
  if (!gpu_is_available()) {
    result.error_message = "no CUDA device available";
    return result;
  }

  const std::size_t schedule_stride = padded_len;
  const std::size_t schedule_bytes = batch_size * schedule_stride;

  std::uint8_t *d_cipher = nullptr;
  std::uint8_t *d_letter_mask = nullptr;
  std::uint8_t *d_schedules = nullptr;
  std::uint8_t *d_schedule_valid = nullptr;
  std::uint8_t *d_invalid_flags = nullptr;
  Mode *d_modes = nullptr;
  char *d_alphabet = nullptr;
  GpuPlainStats *d_stats = nullptr;

  auto cleanup = [&]() {
    cudaFree(d_cipher);
    cudaFree(d_letter_mask);
    cudaFree(d_schedules);
    cudaFree(d_schedule_valid);
    cudaFree(d_invalid_flags);
    cudaFree(d_modes);
    cudaFree(d_alphabet);
    cudaFree(d_stats);
  };

  cudaError_t err = cudaMalloc(&d_cipher, padded_len * sizeof(std::uint8_t));
  if (err != cudaSuccess) {
    result.error_message = cudaGetErrorString(err);
    cleanup();
    return result;
  }
  err = cudaMalloc(&d_letter_mask, text_len * sizeof(std::uint8_t));
  if (err != cudaSuccess) {
    result.error_message = cudaGetErrorString(err);
    cleanup();
    return result;
  }
  err = cudaMalloc(&d_schedules, schedule_bytes * sizeof(std::uint8_t));
  if (err != cudaSuccess) {
    result.error_message = cudaGetErrorString(err);
    cleanup();
    return result;
  }
  err = cudaMalloc(&d_schedule_valid, schedule_bytes * sizeof(std::uint8_t));
  if (err != cudaSuccess) {
    result.error_message = cudaGetErrorString(err);
    cleanup();
    return result;
  }
  err = cudaMalloc(&d_invalid_flags, batch_size * sizeof(std::uint8_t));
  if (err != cudaSuccess) {
    result.error_message = cudaGetErrorString(err);
    cleanup();
    return result;
  }
  err = cudaMalloc(&d_modes, modes.size() * sizeof(Mode));
  if (err != cudaSuccess) {
    result.error_message = cudaGetErrorString(err);
    cleanup();
    return result;
  }
  const std::size_t alphabet_len = alphabet.alphabet.size();
  err = cudaMalloc(&d_alphabet, alphabet_len * sizeof(char));
  if (err != cudaSuccess) {
    result.error_message = cudaGetErrorString(err);
    cleanup();
    return result;
  }
  err = cudaMalloc(&d_stats,
                   batch_size * modes.size() * sizeof(GpuPlainStats));
  if (err != cudaSuccess) {
    result.error_message = cudaGetErrorString(err);
    cleanup();
    return result;
  }

  err = cudaMemcpy(d_cipher, cipher_indices.data(),
                   padded_len * sizeof(std::uint8_t), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    result.error_message = cudaGetErrorString(err);
    cleanup();
    return result;
  }
  err = cudaMemcpy(d_letter_mask, letter_mask.data(),
                   text_len * sizeof(std::uint8_t), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    result.error_message = cudaGetErrorString(err);
    cleanup();
    return result;
  }
  err = cudaMemcpy(d_schedules, key_schedules,
                   schedule_bytes * sizeof(std::uint8_t), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    result.error_message = cudaGetErrorString(err);
    cleanup();
    return result;
  }
  err = cudaMemcpy(d_schedule_valid, key_schedule_valid,
                   schedule_bytes * sizeof(std::uint8_t), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    result.error_message = cudaGetErrorString(err);
    cleanup();
    return result;
  }
  err = cudaMemcpy(d_invalid_flags, key_invalid_flags,
                   batch_size * sizeof(std::uint8_t), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    result.error_message = cudaGetErrorString(err);
    cleanup();
    return result;
  }
  err = cudaMemcpy(d_modes, modes.data(),
                   modes.size() * sizeof(Mode), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    result.error_message = cudaGetErrorString(err);
    cleanup();
    return result;
  }
  err = cudaMemcpy(d_alphabet, alphabet.alphabet.data(),
                   alphabet_len * sizeof(char), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    result.error_message = cudaGetErrorString(err);
    cleanup();
    return result;
  }

  const int threads = 128;
  const int blocks = (static_cast<int>(batch_size) + threads - 1) / threads;
  quag_static_kernel<<<blocks, threads>>>(
      d_cipher,
      d_letter_mask,
      text_len,
      padded_len,
      d_schedules,
      d_schedule_valid,
      schedule_stride,
      d_invalid_flags,
      d_alphabet,
      alphabet_len,
      d_modes,
      modes.size(),
      batch_size,
      d_stats);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    result.error_message = cudaGetErrorString(err);
    cleanup();
    return result;
  }
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    result.error_message = cudaGetErrorString(err);
    cleanup();
    return result;
  }

  result.stats.resize(batch_size * modes.size());
  err = cudaMemcpy(result.stats.data(), d_stats,
                   result.stats.size() * sizeof(GpuPlainStats),
                   cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    result.error_message = cudaGetErrorString(err);
    cleanup();
    return result;
  }

  cleanup();
  result.success = true;
  return result;
}

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
    std::size_t batch_size) {
  std::vector<Mode> modes = {mode};
  return launch_static_gpu(alphabet, modes, cipher_indices, letter_mask,
                           key_schedules, key_schedule_valid,
                           key_invalid_flags, text_len, padded_len, batch_size);
}


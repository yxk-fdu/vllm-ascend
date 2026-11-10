// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

// Cold-registered physical byte ranges. No Tensor or Python object is accessed
// by the DMA consumer, including during request destruction.
struct NativeDmaField {
  std::vector<uintptr_t> sources, destinations;
  size_t bytes;
};
struct NativeDmaLayer {
  std::vector<NativeDmaField> fields;
  size_t experts, slots;
  NativeDmaLayer(std::vector<std::vector<uintptr_t>> sources,
                 std::vector<std::vector<uintptr_t>> destinations,
                 std::vector<size_t> sizes, size_t expert_count, size_t slot_count)
      : experts(expert_count), slots(slot_count) {
    if (sources.size() != destinations.size() || sources.size() != sizes.size() ||
        (slots && sources.empty()) || !experts)
      throw std::invalid_argument("Invalid native DMA field table dimensions");
    for (size_t f = 0; f < sizes.size(); ++f) {
      if (!sizes[f] || sources[f].size() != experts || destinations[f].size() != slots)
        throw std::invalid_argument("Invalid native DMA expert/slot table");
      for (const auto* addresses : {&sources[f], &destinations[f]})
        for (auto address : *addresses)
          if (!address || address > std::numeric_limits<uintptr_t>::max() - sizes[f])
            throw std::invalid_argument("Invalid native DMA byte range");
      fields.push_back({std::move(sources[f]), std::move(destinations[f]), sizes[f]});
    }
  }
};

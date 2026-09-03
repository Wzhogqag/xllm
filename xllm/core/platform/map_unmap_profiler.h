/* Copyright 2025 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>

namespace xllm {

// Lightweight aggregate profiler for VMM map/unmap driver calls
// (aclrtMapMem / aclrtUnmapMem and their CUDA/MLU equivalents).
//
// Both map and unmap funnel through vmm::map / vmm::unmap in vmm_api.cpp, so
// recording there gives full coverage of every category (KV / activation /
// weight / migration / sync / async) with a single instrumentation point.
//
// We only keep aggregates: total count, total/min/max latency, and a
// logarithmic histogram of per-call latency. No per-event log, no timestamps,
// no model/reason tagging. A single record() is one relaxed atomic increment
// per field, so the hot path stays well under "does not affect end-to-end
// latency". A background thread periodically overwrites a summary JSON so the
// last snapshot survives even when the process is killed with SIGKILL.
//
// Note: per-step attribution is intentionally not provided. On single card the
// real vmm::map runs on the XTensorManager RPC service thread (KV allocation is
// dispatched via allocate_async), decoupled from the scheduler step loop, so
// "map cost per scheduler step" has no clean definition. Aggregate stats are
// the honest granularity here.
class MapUnmapProfiler {
 public:
  enum class Op : int { kMap = 0, kUnmap = 1 };

  // Number of logarithmic latency buckets. Bucket i covers
  // [kBucketBaseNs << i, kBucketBaseNs << (i+1)) nanoseconds; the last bucket
  // is an open-ended overflow. Base 100ns, 24 buckets -> up to ~1.7s.
  static constexpr int kNumBuckets = 24;
  static constexpr uint64_t kBucketBaseNs = 100;

  static MapUnmapProfiler& instance();

  // Record one driver call of `op` that took `duration_ns`. No-op when
  // profiling is disabled. Safe to call from any thread.
  void record(Op op, uint64_t duration_ns);

  bool enabled() const { return enabled_; }

 private:
  MapUnmapProfiler();
  ~MapUnmapProfiler();
  MapUnmapProfiler(const MapUnmapProfiler&) = delete;
  MapUnmapProfiler& operator=(const MapUnmapProfiler&) = delete;

  struct Bucket {
    std::atomic<uint64_t> count{0};
    std::atomic<uint64_t> total_ns{0};
    std::atomic<uint64_t> min_ns{UINT64_MAX};
    std::atomic<uint64_t> max_ns{0};
    std::array<std::atomic<uint64_t>, kNumBuckets> hist{};
  };

  void worker_loop();
  void write_summary() const;
  static int bucket_index(uint64_t duration_ns);

  bool enabled_ = false;
  int32_t flush_ms_ = 500;
  std::string output_path_;

  std::array<Bucket, 2> ops_{};  // [kMap], [kUnmap]

  std::thread worker_;
  std::mutex mu_;
  std::condition_variable cv_;
  bool stop_ = false;
};

}  // namespace xllm

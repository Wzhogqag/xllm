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

#include "map_unmap_profiler.h"

#include <glog/logging.h>

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

#include "common/global_flags.h"

namespace xllm {

MapUnmapProfiler& MapUnmapProfiler::instance() {
  static MapUnmapProfiler profiler;
  return profiler;
}

MapUnmapProfiler::MapUnmapProfiler() {
  enabled_ = FLAGS_enable_map_unmap_profiling;
  if (!enabled_) {
    return;
  }
  flush_ms_ = FLAGS_map_unmap_profiling_flush_ms > 0
                  ? FLAGS_map_unmap_profiling_flush_ms
                  : 500;

  std::string dir = FLAGS_map_unmap_profiling_dir;
  if (dir.empty()) {
    dir = ".";
  }
  if (dir.back() == '/') {
    dir.pop_back();
  }
  output_path_ = dir + "/map_unmap_summary.json";

  worker_ = std::thread(&MapUnmapProfiler::worker_loop, this);
  LOG(INFO) << "[map_unmap_profiler] enabled, flush_ms=" << flush_ms_
            << ", output=" << output_path_;
}

MapUnmapProfiler::~MapUnmapProfiler() {
  if (!enabled_) {
    return;
  }
  {
    std::lock_guard<std::mutex> lock(mu_);
    stop_ = true;
  }
  cv_.notify_all();
  if (worker_.joinable()) {
    worker_.join();
  }
  write_summary();
}

int MapUnmapProfiler::bucket_index(uint64_t duration_ns) {
  // Bucket 0 covers [0, kBucketBaseNs). Bucket i (>=1) covers
  // [kBucketBaseNs << (i-1), kBucketBaseNs << i). Saturate at the last bucket.
  if (duration_ns < kBucketBaseNs) {
    return 0;
  }
  int idx = 1;
  uint64_t upper = kBucketBaseNs;
  while (idx < kNumBuckets - 1 && duration_ns >= upper * 2) {
    upper *= 2;
    ++idx;
  }
  return idx;
}

void MapUnmapProfiler::record(Op op, uint64_t duration_ns) {
  if (!enabled_) {
    return;
  }
  Bucket& b = ops_[static_cast<int>(op)];
  b.count.fetch_add(1, std::memory_order_relaxed);
  b.total_ns.fetch_add(duration_ns, std::memory_order_relaxed);

  uint64_t prev_min = b.min_ns.load(std::memory_order_relaxed);
  while (duration_ns < prev_min &&
         !b.min_ns.compare_exchange_weak(
             prev_min, duration_ns, std::memory_order_relaxed)) {
  }
  uint64_t prev_max = b.max_ns.load(std::memory_order_relaxed);
  while (duration_ns > prev_max &&
         !b.max_ns.compare_exchange_weak(
             prev_max, duration_ns, std::memory_order_relaxed)) {
  }

  b.hist[bucket_index(duration_ns)].fetch_add(1, std::memory_order_relaxed);
}

namespace {

// Approximate a percentile from the log histogram, returning the lower edge of
// the bucket in which the percentile falls (nanoseconds).
uint64_t percentile_from_hist(
    const std::array<uint64_t, MapUnmapProfiler::kNumBuckets>& hist,
    uint64_t total,
    double p) {
  if (total == 0) {
    return 0;
  }
  const uint64_t target = static_cast<uint64_t>(p * total);
  uint64_t cumulative = 0;
  for (int i = 0; i < MapUnmapProfiler::kNumBuckets; ++i) {
    cumulative += hist[i];
    if (cumulative >= target) {
      if (i == 0) {
        return 0;
      }
      return MapUnmapProfiler::kBucketBaseNs << (i - 1);
    }
  }
  return MapUnmapProfiler::kBucketBaseNs << (MapUnmapProfiler::kNumBuckets - 1);
}

void append_op_json(
    std::ostringstream& os,
    const char* name,
    uint64_t count,
    uint64_t total_ns,
    uint64_t min_ns,
    uint64_t max_ns,
    const std::array<uint64_t, MapUnmapProfiler::kNumBuckets>& hist) {
  const double total_ms = total_ns / 1e6;
  const double mean_us = count > 0 ? (total_ns / 1e3) / count : 0.0;
  const uint64_t p50 = percentile_from_hist(hist, count, 0.50);
  const uint64_t p90 = percentile_from_hist(hist, count, 0.90);
  const uint64_t p99 = percentile_from_hist(hist, count, 0.99);

  os << "  \"" << name << "\": {\n";
  os << "    \"count\": " << count << ",\n";
  os << "    \"total_ns\": " << total_ns << ",\n";
  os << "    \"total_ms\": " << total_ms << ",\n";
  os << "    \"mean_us\": " << mean_us << ",\n";
  os << "    \"min_ns\": " << (count > 0 ? min_ns : 0) << ",\n";
  os << "    \"max_ns\": " << max_ns << ",\n";
  os << "    \"p50_ge_ns\": " << p50 << ",\n";
  os << "    \"p90_ge_ns\": " << p90 << ",\n";
  os << "    \"p99_ge_ns\": " << p99 << ",\n";
  os << "    \"hist\": [";
  for (int i = 0; i < MapUnmapProfiler::kNumBuckets; ++i) {
    if (i != 0) {
      os << ", ";
    }
    os << hist[i];
  }
  os << "]\n";
  os << "  }";
}

}  // namespace

void MapUnmapProfiler::write_summary() const {
  std::ostringstream os;
  os << "{\n";
  os << "  \"bucket_base_ns\": " << kBucketBaseNs << ",\n";
  os << "  \"num_buckets\": " << kNumBuckets << ",\n";
  os << "  \"hist_note\": \"bucket 0 = [0,base); bucket i = [base<<(i-1), "
        "base<<i)\",\n";

  for (int op = 0; op < 2; ++op) {
    const Bucket& b = ops_[op];
    std::array<uint64_t, kNumBuckets> hist{};
    for (int i = 0; i < kNumBuckets; ++i) {
      hist[i] = b.hist[i].load(std::memory_order_relaxed);
    }
    append_op_json(os,
                   op == 0 ? "map" : "unmap",
                   b.count.load(std::memory_order_relaxed),
                   b.total_ns.load(std::memory_order_relaxed),
                   b.min_ns.load(std::memory_order_relaxed),
                   b.max_ns.load(std::memory_order_relaxed),
                   hist);
    os << (op == 0 ? ",\n" : "\n");
  }
  os << "}\n";

  // Write atomically: tmp file then rename, so a SIGKILL mid-write never
  // leaves a truncated summary.
  const std::string tmp_path = output_path_ + ".tmp";
  {
    std::ofstream ofs(tmp_path, std::ios::trunc);
    if (!ofs) {
      LOG_EVERY_N(WARNING, 100)
          << "[map_unmap_profiler] cannot open " << tmp_path;
      return;
    }
    ofs << os.str();
  }
  std::rename(tmp_path.c_str(), output_path_.c_str());
}

void MapUnmapProfiler::worker_loop() {
  while (true) {
    std::unique_lock<std::mutex> lock(mu_);
    cv_.wait_for(
        lock, std::chrono::milliseconds(flush_ms_), [this] { return stop_; });
    const bool stopping = stop_;
    lock.unlock();

    write_summary();
    if (stopping) {
      break;
    }
  }
}

}  // namespace xllm

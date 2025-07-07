// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifdef PADDLE_WITH_XPU

#include <thread>
#include <unordered_set>

#include "paddle/common/errors.h"
#include "paddle/common/macros.h"
#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/utils/optional.h"

namespace phi {
namespace backends {
namespace xpu {

class CUDAGraphContextManager {
  // TODO(lihaoran06): implement this class
};

class gpuKernelParams {
 public:
  explicit gpuKernelParams(void **params) : kernelParams(params) {}

  template <typename T>
  T &As(size_t idx) const {
    return *reinterpret_cast<T *>(kernelParams[idx]);
  }

  void **getParams() const {
    return kernelParams;
  }

 private:
  void **kernelParams;
};

using cudaGraphExecuterSetter_t = std::function<void(cudaGraphExec_t)>;

class CUDAGraphNodeLauncher {
 public:
  using parameterSetter_t = std::function<void(gpuKernelParams&)>;
  using cudaGraphExecuterSetter_t = std::function<void(cudaGraphExec_t)>;
 private:
  DISABLE_COPY_AND_ASSIGN(CUDAGraphNodeLauncher);
};

#if CUDA_VERSION >= 10010
static void ThrowErrorIfNotSupportCUDAGraph() {}
#else
enum xpuStreamCaptureMode {
  cudaStreamCaptureModeGlobal = 0,
  cudaStreamCaptureModeThreadLocal = 1,
  cudaStreamCaptureModeRelaxed = 2
};
static void ThrowErrorIfNotSupportCUDAGraph() {
  PADDLE_THROW(common::errors::Unimplemented(
      "CUDA Graph is only supported when CUDA version >= 10.1"));
}
#endif

using CUDAGraphID = unsigned long long;

class CUDAGraph {
  DISABLE_COPY_AND_ASSIGN(CUDAGraph);
  CUDAGraph();
 public:
  using CUDAPostResetCallback =
      std::function<void(paddle::optional<const CUDAGraph&>)>;
  using CUDAPreCaptureCallback = std::function<void()>;
  using CUDAPostCaptureCallback = std::function<void()>;
  using SetSeedFunc = std::function<bool(gpuKernelParams *, bool)>;

  static constexpr int64_t kDefaultPoolID = 0;
  static constexpr int64_t kInvalidPoolID = -1;

  static int64_t SetMemoryPoolID(int64_t pool_id);
  static int64_t CapturingPoolID();
  static void BeginCapture(phi::XPUPlace place,
                           cudaStream_t stream,
                           cudaStreamCaptureMode mode);
  static std::unique_ptr<CUDAGraph> EndCapture();
  static void BeginSegmentCapture();
  static void EndSegmentCapture();
  static void AddJoiningStreamDuringCapturing(cudaStream_t stream);
  static void AddPostResetCallbackDuringCapturing(
      CUDAPostResetCallback callback);
  static void AddPostCaptureCallbackDuringCapturing(
      CUDAPostCaptureCallback callback);
  static bool IsCapturing();
  static CUDAGraphID CapturingID();
  static phi::XPUPlace CapturingPlace();
  static bool IsValidCapturing();
  static bool IsThreadLocalCapturing();
  static bool IsThisThreadCapturing();
  static void RecordRandomKernelInfo(SetSeedFunc set_seed_func);
  static int64_t UniqueMemoryPoolID();

  ~CUDAGraph();
  CUDAGraphID ID() const;
  int64_t PoolID() const;
  void Replay();
  void Reset();
  void AddPostResetCallback(CUDAPostResetCallback callback);
  void AddPreCaptureCallback(CUDAPreCaptureCallback callback);
  void AddPostCaptureCallback(CUDAPostCaptureCallback callback);
  void AddJoiningStream(cudaStream_t stream);
  void PrintToDotFiles(const std::string &dirname, unsigned int flags);
  bool IsReplayed() const;

 private:
  static CUDAGraphID UniqueID();

#if CUDA_VERSION >= 10010
  std::vector<cudaGraph_t> graphs_;
  std::vector<cudaGraphExec_t> exec_graphs_;
  cudaStreamCaptureMode capture_mode_;
#endif
  cudaStream_t stream_{nullptr};
  phi::XPUPlace place_;
  CUDAGraphID id_;
  int64_t pool_id_{kInvalidPoolID};
  bool is_reset_{false};
  bool is_replayed_{false};
  std::mutex mtx_;

  std::vector<SetSeedFunc> set_seed_funcs_;

  std::unordered_set<cudaStream_t> streams_to_join_;

  // Holds callbacks that are triggered after the CUDA graph is reset. These
  // callbacks are used for operations that need to be performed following the
  // reset of a CUDA graph.
  std::vector<std::function<void(paddle::optional<const CUDAGraph &>)>>
      cudagraph_post_reset_callbacks_;

  static std::vector<std::function<void()>> cudagraph_pre_capture_callbacks_;

  // Contains callbacks that are invoked after the CUDA graph has been captured.
  // These callbacks are crucial for managing memory allocations related to the
  // CUDA graph. They ensure that memory blocks not associated with a graph (as
  // detailed in cuda_malloc_async_allocator) are not erroneously released
  // during the graph's lifecycle.
  std::vector<std::function<void()>> cudagraph_post_capture_callbacks_;

  // Maintains a collection of 'pre-hooks' - functions that are executed before
  // the CUDA graph is replayed. These pre-hooks are essential for setting up
  // the necessary conditions or states required for the correct execution of
  // the CUDA graph.
  std::vector<std::vector<cudaGraphExecuterSetter_t>>
      cudagraph_pre_replay_callbacks_;

  std::mutex func_mtx_;

  bool is_first_run_{true};

  static paddle::optional<std::thread::id> capturing_thread_id_;
  static std::unique_ptr<CUDAGraph> capturing_graph_;
};

} // namespace xpu
} // namespace backends
} // namespace phi

#endif
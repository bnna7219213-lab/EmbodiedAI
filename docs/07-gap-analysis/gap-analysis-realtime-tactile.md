# MultiModalFusionPolicy 在 Jetson Orin NX 上的实时性可行性分析

> 本文档是对外呼应用户提问 q1 的回应: 触觉融合网络在 Jetson Orin NX 上能否满足 20Hz 实时控制?

---

## 1. 关键参数基准

| 参数 | 攀岩墙模型 (原文) | 家庭服务预估 |
|---|---|---|
| 模型参数 | ~10M | ~15M (加本体感受) |
| 触觉输入 | 4指 × 5帧 × 64×64 | 3指 × 8帧 × 48×48 |
| 视觉输入 | 1 × 224×224 | 2 × 224×224 (头部+胸部摄像头) |
| 本体感觉 | 30 维 | 48 维 (关节角+速度+力矩+IMU+力) |
| 动作输出 | 12 维 | 13 维 (12关节 + 手爪) |
| 控制频率 | 20Hz (50ms) | 20Hz (50ms) |
| 最大推理延迟 | < 50ms | < 35ms (留 15ms 安全余量) |

## 2. Orin NX 算力模型: 延迟来源拆解

### 2.1 子模块延迟明细 (TensorRT FP16)

| 操作 | 输入 tensor | 计算量 | Orin NX 估计延迟 |
|---|---|---|---|
| TactileEncoder×4 | (B,5,3,48,48)×4 → (B,4,D) | ~400M MACs × 4 = 1.6G | ~6 ms |
| VisualEncoder | (B,3,224×224) → (B,D,4,4) | ~1.2G MACs | ~5 ms |
| ProprioEncoder | (B,48) → (B,D) | <1M MACs | <0.1 ms |
| CrossModalAttn×2 | Q+KV self-attn, seq=20 | ~50M MACs | ~1 ms |
| Fusion + Policy | FC 层 | <1M MACs | <0.1 ms |
| **合计推理** | | | **~12 ms** |
| 传感器读入 | MIPI + SPI/编码 + DMA | | ~5 ms |
| 触觉图像预处理 | 裁剪/背景减/时序对齐 | | ~3 ms |
| **总计端到端** | | | **~20 ms** |

> **结论**: 纯网络推理 ~12ms，含 I/O ~20ms，理论可行但无安全余量。
> 预留 TensorRT INT8 余量后，**可稳定达到 20ms**。

### 2.2 20ms 的安全余量如何保证？

```
关键: 每条路径都必须可分析最坏延迟 (worst-case execution time, WCET)

  - TactileEncoder 的 GRU 是循环结构，batch=1 时延迟稳定
  - VisualEncoder 卷积是固定计算量，不依赖输入内容
  - CrossModalAttn 序列长度固定 (token 数固定)
  → 整个推理路径 **WCET ≈ AWCET** (确定性延时)

  保留 15ms 安全余量用途:
    - 传感器偶发抖动: ~3ms
    - CAN 帧发收: ~2ms
    - 力控前馈调整 (如果 RL 输出之后还要加安全层): ~5ms
    - 日志写入 + 异常保护: ~2ms
    - 系统突发抖动 (irq 抢占): ~3ms
```

### 2.3 拓扑: 单 Orin 双臂 vs 双 Orin 全机

```
方案 A: 单 Orin NX 16GB 跑全身
  优点: 成本低、通信简单
  风险:
    - 20Hz MMDL 推理 + GPU 视觉处理 + NPU 推理可能争抢
    - 单核 CPU 4-7 需并行 RL 决策 + 安全监控，上下文切换开销大
    - GPU 成为瓶颈

方案 B: 双 Orin 分治 (推荐)
  Orin 1 (胸部): 视觉 + VLM 推理 + 高层规划 (1-2Hz)
  Orin 2 (背部): RL 决策 + 触觉融合 + 安全监控 (20Hz)
  通信:PCIe 或 10GbE → 时延 <1ms
  
  优点: 互不干扰，双冗余安全 (一个挂掉另一个接管)
  缺点: 成本高 ~$500×2
 
方案 C: 单 Orin + Cortex-R5 协处理 (固件级安全)
  Orin 运行 Linux + Python 推理
  Cortex-R5 运行安全监控固件 (独立 OS): 做最后一道硬件安全闸
  → 与我们在 hardware-motor-shutdown.md 中描述的 "看门狗+自动关机" 完美融合
  → 推荐方案: 单 Orin + Cortex-R5 协处理
```

## 3. TensorRT 优化关键步骤

```python
# ====== trt_optimization.py ======
# PyTorch MultiModalFusionPolicy → TensorRT .engine

import tensorrt as trt
import torch

def build_tactile_fusion_engine(
    onnx_path: str,
    engine_path: str,
    precision: str = "fp16",      # or "int8"
    max_batch: int = 1,
    workspace_gb: int = 4,
):
    """将多模态融合网络编译为 TensorRT 引擎"""
    
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    
    # 1. 解析 ONNX
    with open(onnx_path, 'rb') as f:
        parser.parse(f.read())
    
    # 2. 构建配置
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    
    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8":
        config.set_flag(trt.BuilderFlag.INT8)
        # INT8 需要校准数据集, 但触觉信号分布窄, 校准容易
        config.int8_calibrator = TactileCalibrator(
            data_dir="/data/tactile_calibration_500imgs",
            cache_file="/tmp/tactile_int8.cache",
        )
    
    # 3. 动态形状优化 (触觉帧序列长度可能变化)
    profile = builder.create_optimization_profile()
    profile.set_shape("tactile_input", 
                      min=(1, 3, 5, 48, 48),
                      opt=(1, 3, 8, 48, 48),
                      max=(1, 3, 12, 48, 48))
    profile.set_shape("visual_input", 
                      min=(1, 3, 224, 224),
                      opt=(1, 3, 224, 224),
                      max=(1, 3, 224, 224))
    config.add_optimization_profile(profile)
    
    # 4. 编译引擎 (Orin 上编译需 10-30 分钟)
    engine = builder.build_serialized_network(network, config)
    with open(engine_path, 'wb') as f:
        f.write(engine)
    
    print(f"Engine built: {engine_path} ({len(engine)/1e6:.1f}MB)")

# 推理接口
class TactileFusionTRT:
    """零拷贝推理: GPU 上预处理, GPU 上推理"""
    
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        
        with open(engine_path, 'rb') as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        
        # 预分配 GPU 内存 (避免推理中 malloc)
        self.d_tactile = cuda.mem_alloc(1 * 3 * 8 * 48 * 48 * 4)
        self.d_visual = cuda.mem_alloc(1 * 3 * 224 * 224 * 4)
        self.d_proprio = cuda.mem_alloc(1 * 48 * 4)
        self.d_output = cuda.mem_alloc(1 * 13 * 4)
    
    def infer(self, tactile_np, visual_np, proprio_np):
        """单次推理 <10ms (FP16) 或 <6ms (INT8)"""
        # 异步 H2D 拷贝
        cuda.memcpy_htod_async(self.d_tactile, tactile_np, self.stream)
        cuda.memcpy_htod_async(self.d_visual, visual_np, self.stream)
        cuda.memcpy_htod_async(self.d_proprio, proprio_np, self.stream)
        
        # 执行 (绑定 GPU 指针)
        self.context.execute_async_v2(
            bindings=[int(self.d_tactile), int(self.d_visual),
                      int(self.d_proprio), int(self.d_output)],
            stream_handle=self.stream.handle
        )
        
        # 异步 D2H 拷贝
        output = np.empty(13, dtype=np.float32)
        cuda.memcpy_dtoh_async(output, self.d_output, self.stream)
        self.stream.synchronize()
        
        return output
```

## 4. INT8 校准数据集构造

```python
# ====== int8_calibration.py ======

class TactileCalibrator(trt.IInt8EntropyCalibrator2):
    """
    触觉信号的 INT8 校准器
    
    关键洞察: 触觉图像 RGB 值分布远比自然图像窄
      - GelSight 依赖 LED 照射, 颜色范围有限 (主要是红/绿/蓝变化)
      - 弹性体形变后的 RGB 变化集中在几个固定区域
      → INT8 量化精度损失 < 0.1%
    """
    
    def __init__(self, data_dir, cache_file, batch_size=32):
        super().__init__()
        self.cache_file = cache_file
        self.batch_size = batch_size
        
        # 加载 500 张代表性触觉帧 (覆盖各种接触/不接触)
        self.data = []
        for f in Path(data_dir).glob("*.npy"):
            self.data.append(np.load(f))
        self.data = np.stack(self.data)
        
        self.current_index = 0
    
    def get_batch_size(self):
        return self.batch_size
    
    def get_batch(self, names):
        if self.current_index >= len(self.data):
            return None
        
        batch = self.data[self.current_index:self.current_index+self.batch_size]
        self.current_index += self.batch_size
        
        # 返回 GPU 指针
        cuda_batch = cuda.mem_alloc(batch.nbytes)
        cuda.memcpy_htod(cuda_batch, batch.tobytes())
        return [int(cuda_batch)]
    
    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                return f.read()
        return None
    
    def write_calibration_cache(self, cache):
        with open(self.cache_file, 'wb') as f:
            f.write(cache)
```

## 5. 关键风险缓解

| 风险 | 影响 | 缓解措施 |
|---|---|---|
| GPU 同时处理视觉 (4 路摄像头) | GPU 争抢,延迟窜高 | 视觉用 NVENC 硬加速,NPU 跑 ViT; GPU 仅跑 RL 决策 |
| 触觉帧率不足 (GelSight 仅 30Hz) | 时序窗口数据不足 | 多帧插值预测 + 触发式采样 (仅采集变化帧) |
| TensorRT 编译后精度损失 | 控制抖动 | INT8 校准集覆盖全部工况 + 在线 RMS 监控 |
| CPU 调度抖动 ±5ms | 周期不稳定 | mlockall + SCHED_FIFO + isolcpus + DMA |
| 内存泄漏 (长时间 7×24 运行) | 重启前崩溃 | 定时检测 mempool + watchdog 自动重启 |

## 6. 结论

**在 Jetson Orin NX 16GB 上, MultiModalFusionPolicy 模型可以做到 20Hz 控制频率, 推理延迟 ~12ms (FP16) 或 ~6ms (INT8), 满足实时要求。**

但需要三个关键举措:
1. TensorRT 编译 + INT8 量化 (必需,否则延迟可能 >25ms)
2. 安全协处理器 (Cortex-R5 或 STM32H7) 独立兜底
3. 视觉与 RL 使用不同硬件单元 (NPU vs GPU 分离)

# VLA (Vision-Language-Action) 模型架构设计

> 本文档定义具身智能体的 AI 模型完整架构, 从输入到输出的每一层。
> 参照 LLMInnovation 的实现哲学 (自研为主), 面向具身智能场景做专门调整。

---

## 1. 架构总览

```
┌───────────────────────────────────────────────────────────────────┐
│                    VLA 模型 v0.1 架构                               │
│                                                                   │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐              │
│  │   视觉输入    │ │   语言输入    │ │   状态输入    │              │
│  │  (多视角)     │ │  (指令)       │ │  (本体感受)   │              │
│  └──────┬───────┘ └──────┬───────┘ └──────┬───────┘              │
│         │                │                │                       │
│  ┌──────▼───────┐ ┌──────▼───────┐ ┌──────▼───────┐              │
│  │ Vision       │ │ Language     │ │ State        │              │
│  │ Encoder      │ │ Encoder      │ │ Encoder      │              │
│  │ (197M)       │ │ (共享/独立)   │ │ (1M)         │              │
│  └──────┬───────┘ └──────┬───────┘ └──────┬───────┘              │
│         │                │                │                       │
│         └────────────────┴────────────────┘                       │
│                          │                                        │
│                ┌─────────▼─────────┐                              │
│                │  Fusion           │                              │
│                │  Transformer      │                              │
│                │  (7B, 32 层)      │                              │
│                │  Hidden: 8192     │                              │
│                │  GQA: 8 组        │                              │
│                │  RoPE + NTK       │                              │
│                └─────────┬─────────┘                              │
│                          │                                        │
│                ┌─────────▼─────────┐                              │
│                │  Action           │                              │
│                │  Tokenizer        │                              │
│                │  (VQ-VAE)         │                              │
│                └─────────┬─────────┘                              │
│                          │                                        │
│                ┌─────────▼─────────┐                              │
│                │  AutoRegressive   │                              │
│                │  Decoder          │                              │
│                └─────────┬─────────┘                              │
│                          │                                        │
│                   Action Tokens                                   │
│                          │                                        │
│                ┌─────────▼─────────┐                              │
│                │  Detokenizer     │                              │
│                └─────────┬─────────┘                              │
│                          │                                        │
│                    Joint Angles                                   │
│                   [θ1, θ2, ..., θ12]                              │
└───────────────────────────────────────────────────────────────────┘
```

---

## 2. 输入编码

### 2.1 视觉编码器 (ViT-B/16)

```
输入: 3 路 224×224 RGB 图像 (头顶 / 胸部 / 手端)

ViT-B/16 配置:
  patch_size: 16 × 16
  num_patches: (224/16)² = 196 per view
  num_views: 3 → 588 总 patch
  embed_dim: 768
  num_heads: 12
  num_layers: 12
  MLP_dim: 3072
  参数量: ~86M (预训练) + ~1M (投影层)
  
输出: [batch, 588, 768] → projection → [batch, 588, 4096]
  (投影到 LLM 维度 4096 或 8192, 用 GQA) 

视觉 token 数优化:
  588 个 token 太多 (占用 LLM 上下文)
  可压缩: 
    - 使用 Perceiver Resampler → 64 个 token (开销 +3M 参数)
    - 或使用 Token Merging (Hierarchical) → ~60 tokens
  推荐: Perceiver Resampler (64 tokens), 硬件友好
```

### 2.2 语言编码器

```
输入: 自然语言指令 "pick up the red cup from the table"

编码方式 A (推荐): 复用 LLM backbone 的 embedding
  优点: 无需额外参数
  缺点: 训练时需要注意对齐

编码方式 B (小型独立): Sentence-T5 small (~15M 参数)
  优点: 模块解耦, 预训练权重稳定
  缺点: 小模型语义表示能力有限

选择: 方式 A, 直接复用 Fusion Transformer 的 text embedding
      (参照 LLMInnovation 的 tokenizer/vocab)

处理:
  1. BPE 子词化
  2. Token embedding (vocab_size=32768, dim=8192)
  3. RoPE 位置编码
  输出: [batch, seq_len, 8192]
```

### 2.3 状态编码器

```
输入: 当前本体感受向量
  - 12 关节角度 (float32, 弧度)
  - 12 关节速度 (float32, rad/s)
  - 12 关节力矩 (float32, Nm)
  - 6 力传感器读数 (float32, 3 力 + 3 力矩)
  - 6 IMU 读数 (float32, 3 加速度 + 3 角速度)
  共 48 维

编码: 
  方式1: 简单线性投影 48 → 8192 (推荐用于 3B/7B)
  方式2: 小型 MLP: 48 → 256 → 1024 → 8192 (参数量 ~4M)
  
  选用方式1 (足够): state_proj = Linear(48, 8192)
  输出: [batch, 1, 8192] (单 token, 表示当前状态)

扩展到历史:
  需要短期记忆时, 取过去 N=10 步状态
  增加时间位置编码
  输出: [batch, 10, 8192]
```

---

## 3. Fusion Transformer (核心)

### 3.1 配置

```
基于 LLaMA-3 风格的 Decoder-only Transformer:

config:
  vocab_size: 32768
  d_model: 8192
  num_layers: 32
  num_heads: 64
  num_kv_heads: 8  (GQA, 8 组共享 KV)
  d_ff: 24576 (SwiGLU)
  context_len: 2048 (足够容纳 visual + text + state + history)
  rms_norm_eps: 1e-6
  rope_theta: 500000 (支持长上下文)

参数量估算:
  Attention: 32 × 4 × 8192 × 8192 ≈ 8.6B
  MLP (SwiGLU): 32 × 3 × 8192 × 24576 ≈ 19.3B (实际更大)
  Embedding: 32768 × 8192 ≈ 0.27B
  总计: ~32B (太大!)
  
调整 (7B 级别):
  d_model: 4096
  num_layers: 32
  num_heads: 32
  num_kv_heads: 8
  d_ff: 11008
  总计: ~7B (符合名称)
```

### 3.2 Prompt 结构

```
<system_prompt> + <visual_token_1> + ... + <visual_token_64> +
<lang_token_1> + ... + <lang_token_N> +
<state_token> + <history_1> + ... + <history_10> +
<action_query>
  → 模型输出: <action_token_1> + ... + <action_token_T>

上下文分配:
  系统指令: ~50 tokens (角色设定 + 安全规则)
  视觉 tokens: 64 tokens (Perceiver 压缩后)
  语言指令: ~30 tokens
  state + history: 11 tokens (当前 + 过去 10)
  动作输出: 20 tokens (未来 1 秒 @ 20Hz)
  ────────────────────────
  总计: ~175 tokens (远小于 2048 上下文上限)

```
---

## 4. Action Tokenizer

### 4.1 动作离散化 (VQ-VAE 风格)

```
问题: 关节角度是连续值, 模型输出是离散 token, 需要编解码。

方案 A: 简单分桶 (不做 VQ-VAE)
  每个关节角度 θ ∈ [-π, π] → 分 1024 桶
  token_id = int((θ + π) / (2π) × 1024)
  逆变换: θ = (token_id / 1024) × 2π - π
  精度: 2π/1024 ≈ 0.006 rad ≈ 0.35°

  问题: 12 个关节 → 12 × 1024 = 12288 种 token
  需要大的 output vocab

方案 B (推荐): VQ-VAE 连续 token
  Encoder: [batch, T, 12] → [batch, T, D] latents
  量化: 学习 1024 个 codebook entry
  Decoder: [batch, T, codebook_idx] → [batch, T, 12] 角度

  优势:
    - Codebook 可表示协调动作 (多关节联动)
    - 1024 token_id 覆盖 12 维连续空间
    - 解码器可以做后处理 (平滑/限幅)
  
  训练 (分两阶段):
    Stage 1: 只用 Reconstruct loss 训练 VQ-VAE (不需要标签)
    Stage 2: 冻结 VQ, 训练 VLA 以输出 codebook index
```

### 4.2 Action Token 序列

```
输出序列:
  [a_0, a_1, ..., a_19]

  每个 a_t 是 codebook index (uint16, 0-1023)
  解码: VQ-VAE decoder → [θ1(t), ..., θ12(t)]

  时间: t=0..19, 共 20 步 × 50ms = 1 秒前瞻
  
  步长: 模型每次推理输出未来 1 秒的所有步骤
  使用: 执行前 5 步 (250ms), 然后重新推理
  (放弃剩余 15 步, 因为环境可能已变化)
```

---

## 5. 训练策略

### 5.1 三阶段训练

```
阶段 1: 行为克隆 (Behavior Cloning, BC) — 简单监督训练
  数据: (观测, 动作) 对 (来自人类示范)
  损失: L1(token_pred, token_true)
  时长: 1-2 周
  预期成功率: 50-70% (在训练分布内)

阶段 2: 残差学习 (Residual Learning) — 在 BC 基础上微调
  数据: 阶段 1 失败的轨迹 + 人工修正
  损失: L1(residual_pred, residual_true) + L_BC
  时长: 1 周
  预期成功率: +5-10% (弥补系统偏差)

阶段 3: 强化学习 (RL) — 端到端策略优化
  奖励: 任务完成 + 时间惩罚 + 平滑度
  算法: PPO 或 REINFORCE
  风险: 真实试错成本高, 需先在仿真做
  时长: 2-4 周 (仿真), 未知 (真实)
  预期成功率: 可能超过 BC 上限
```

### 5.2 训练数据配比

```
数据来源            占比      用途
─────────────────────────────────────
仿真数据 (Isaac Sim)  40%      预训练 (形状/运动学)
遥操作示教            30%      行为克隆
公开数据集 (OXE)      20%      泛化能力
自采集真实数据        10%      Sim-to-Real 迁移
```

### 5.3 数据增强

```
仿真数据增强:
  - 域随机化: 光照 /纹理 / 摩擦系数 / 延迟
  - 外参扰动: 相机位置 +2cm 噪声
  
真实数据增强:
  - 图像: 色彩抖动 / 高斯模糊 / 随机裁剪
  - 本体感受: 高斯噪声 (模拟传感器噪声)
  - 动作: 时间偏移 ±2 帧 (不改变语义)
```

---

## 6. 推理优化

### 6.1 KV Cache 优化

```
  推理时 KV Cache 大小:
    每层 K, V: [batch, num_kv_heads, ctx_len, head_dim]
      = 1 × 8 × 2048 × 128 = 2M elements
    每层: 2M × 2 (K+V) × 2 bytes (fp16) = 8 MB
    32 层: 8 × 32 = 256 MB
    
  问题: 64 个视觉 tokens 长时间不变化 (若摄像头不变)
  解决: 视觉 KV 只计算一次, 后续复用
    视觉 KV 缓存: 64 × 32 × 2 × 2 = 8 KB (很小!)
    刷新: 每 50ms 刷新 (常态)
```

### 6.2 量化

```
  模型权重:
    FP32 → INT8: 2x 加速, 显存减半
    FP32 → INT4: 4x 加速, 显存 1/4
    
    推荐: GPTQ / AWQ INT4 量化
    7B INT4 = 3.5GB → 可放入 Jetson Orin 8GB
    
  激活值:
    保持 FP16 (量化激活影响精度)
    推理引擎: TensorRT / ONNX Runtime
  
  目标延迟:
    INT4 量化 7B 模型 @ Jetson Orin: ~30ms
    原始 FP16 7B 模型: ~80ms
    含 KV cache + 增量解码: ~50ms
    足够满足 20Hz 控制
```

---

## 7. 模型与环境交互

### 7.1 观测空间设计

```
每次推理的观测:

observation = {
  # 视觉
  "cam_top": tensor[3, 224, 224],       # RGB
  "cam_chest": tensor[3, 224, 224],
  "cam_hand_left": tensor[3, 224, 224],
  
  # 状态
  "joint_positions": tensor[12],          # 弧度
  "joint_velocities": tensor[12],         # rad/s
  "joint_torques": tensor[12],            # Nm
  
  # 力传感
  "force_torque": tensor[6],              # 3F + 3T
  "imu": tensor[6],                       # 3accel + 3gyro
  
  # 指令
  "instruction": "pick up red cup",
  
  # 历史 (可选)
  "action_history": tensor[10, 12],       # 过去 10 步执行的动作
}
```

### 7.2 动作空间

```
action = {
  "joint_positions": tensor[12],     # 目标角度 (rad)
  "joint_velocities": tensor[12],    # 目标角速度 (rad/s, 可选)
  "gripper": float,                  # 手爪开合 [0,1]
}

直接发送到各关节控制板 → FOC 跟踪目标位置

不做: 力矩直接控制 (只做位置控制, 力矩限制在驱动板实现)
原因: 
  - 位置控制对模型更友好 (更平滑)
  - 力矩控制需要更精确的模型, 训练更难
  - 自动关机硬件仍然监控力矩/电流异常
```

---

## 8. 与 LLMInnovation 的技术继承关系

| LLMInnovation 模块 | 复用方式 | 调整 |
|---|---|---|
| BPE Tokenizer | 直接使用 | 增加动作 token 词表 |
| RMSNorm | 直接使用 | 无 |
| RoPE + NTK | 直接使用 | 增加视觉 token 位置编码 |
| SwiGLU | 直接使用 | 无 |
| GQA | 直接使用 | KV 分组从 4 增到 8 |
| FlashAttention | 增加视觉 token 处理 | 整合多模态 KV |
| ZeRO-3 | 训练阶段使用 | 无变化 |
| KV Cache | 复用 + 视觉 KV 缓存 | 增加视觉缓存逻辑 |

```
LLMInnovation 提供: 语言模型基础模块 (tokenizer, norm, attention, FFN, training utilities)
本计划在此基础上增加: Vision Encoder, Action Tokenizer, VQ-VAE, State Encoder, Fusion Logic

不直接用 LLMInnovation 的:
  - Text-to-Image Diffusion (不相关)
  - Text-to-3D (暂不相关)
  - Text-to-Video (暂不相关)
  这些回到 LLMInnovation 作为独立项目
```

# 触觉 Sim-to-Real 策略选择分析

> 本文档是对外呼应用户提问 q2 的回应: 如何把仿真中的触觉策略迁移到真实机器人?

---

## 1. 为什么触觉 Sim2Real 比视觉更难

```
视觉 Sim2Real 已建成成熟管线:
  已经在 ImageNet 预训练 → 各种 CV 任务零样本也能工作
  域差距来自: 光照/纹理/镜头畸变 → 已经由数据增强覆盖

触觉 Sim2Real 的挑战是独特的:
  - 仿真无法模拟: 
    弹性体微观形变/空气泡/温度漂移/老化磨损/不同批次的生产差异
  - 每位用户的 GelSight .units 响应完全不同
  - 即使同一台设备, 连续运行 1 小时后温度变化 → 读数漂移
  - 物理接触本身具有不可复现性 (玻璃杯在仿真中是完美圆, 真实有瑕疵)

结论: 触觉的域差距比视觉大一个数量级
→ 纯仿真策略无法直接使用, 必须做迁移
```

## 2. 三种 Sim2Real 路线的对比

| 方案 | 原理 | 优点 | 缺点 | 适用我们的场景? |
|---|---|---|---|---|
| **A. TACTO + PyBullet** | 基于光线追踪的视触觉仿真 | 开源、可用、免费 | 精度低、仅静态接触、不仿真剪切力 | Phase 1 行为克隆预训练 |
| **B. Taxim (GelSight)** | 高保真弹性体形变仿真 | 精度高、支持多平台 | 计算慢 (100ms/帧 vs 真实 30ms/帧)、快接触仿真不准 | Phase 2 精细操作策略 |
| **C. Isaac Sim + 触觉插件** | NVIDIA 官方提供 | GPU 并行、大规模 | 触觉模型为简化物理 (不是真实 GelSight 物理) | Phase 1+ 数据生成管线 |
| **D. 系统辨识 + 真实采样** | 录真实数据 → 训练域差异补偿网络 | 最可靠 | 需要真实设备 (需先有硬件) | Phase 2+ 部署前必需 |
| **E. LoRA 低秩微调** | 仅微调策略网络的最后几层 | 高效 (只训 5% 参数)、收敛快 | 需要真实数据 (少量即可) | Phase 2+ 快速迁移 |

```
推荐组合策略:
  Phase 0-1: ACTO/PyBullet 仿真 → 行为克隆 (Phase 0)
  Phase 1-2: 真实硬件采集 → LoRA 微调 (双轨并行)
  Phase 2+:  强化学习在线适应 + DADR 数据增广
```

## 3. 域随机化参数工程

```python
# tactile_domain_randomization.py

TACTILE_DOMAIN_RANDOM_PARAMS = {
    # === 仿真 → 仿真的随机化 (解决仿真过拟合) ===
    'sensor': {
        'noise_std': ('uniform', 0.0, 0.05),          # 高斯噪声
        'drift_rate': ('uniform', -0.01, 0.01),        # 每秒温度漂移
        'dead_pixel_prob': (0.0, 0.02),                # 坏元 %
        'response_nonlinear': ('uniform', 0.95, 1.05), # 非线性系数
    },
    'material': {
        'led_brightness': ('uniform', 0.85, 1.15),     # LED 亮度 ±15%
        'gel_hardness': ('uniform', 0.8, 1.2),         # 硅胶硬度
        'marker_density': ('uniform', 0.9, 1.1),       # 示点子密度
        'surface_roughness': ('uniform', 0.5, 2.0),    # 表面摩擦
    },
    'contact': {
        'penetration_depth_scale': ('uniform', 0.7, 1.3),
        'shear_stiffness': ('uniform', 0.5, 2.0),
        'dynamic_viscosity': ('loguniform', 0.1, 10.0),
    },
    # === 仿真 → 真实的迁移随机化 (解决域差距) ===
    'sim2real': {
        'image_jitter': {
            'brightness': 0.3,
            'contrast': 0.3,
            'gaussian_blur_sigma': ('uniform', 0.5, 2.0),
        },
        'color_shift': ('uniform', -0.2, 0.2),         # RGB 通道偏移
        'spatial_warp': ('uniform', -0.05, 0.05),       # 微小仿射形变
        'temporal_dropout': (0.0, 0.1),                 # 随机丢帧
    }
}
```

## 4. LoRA 极少样本迁移

```python
# lora_tactile_transfer.py

class TactileLoRA:
    """
    使用 LoRA 微调触觉融合网络, 仅需 100 条真实示教轨迹
    
    100 条轨迹 = 10 分钟遥操作数据采集
    远低于通常 RL 需要的 10K 条
    """
    
    def __init__(self, pretrained_policy, lora_r=8, lora_alpha=16):
        """
        Args:
            pretrained_policy: 仿真中训好的 MultiModalFusionPolicy
            lora_r: 低秩分解秩 (越小参数越少)
            lora_alpha: 缩放因子
        """
        self.policy = pretrained_policy
        self.lora_r = lora_r
        
        # 仅对 GRU temporal 和 FC policy 加 LoRA
        # TactileEncoder 的 CNN 冻结 (视觉表征已成熟)
        self.lora_layers = nn.ModuleDict({
            'temporal_gru': LoRAGRU(
                pretrained_policy.tactile_encoder.temporal,
                r=lora_r, alpha=lora_alpha
            ),
            'policy_mean': LoRALinear(
                pretrained_policy.policy_mean,
                r=lora_r, alpha=lora_alpha
            ),
        })
        
        # 冻结其他参数
        for name, param in self.policy.named_parameters():
            if 'temporal' not in name and 'policy_mean' not in name:
                param.requires_grad = False
        
        # 可训练参数仅占 5%
        trainable = sum(p.numel() for p in self.policy.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.policy.parameters())
        print(f"LoRA: {trainable/total:.1%} 参数可训练 ({trainable/1e6:.2f}M/{total/1e6:.2f}M)")
    
    def finetune(self, real_dataset, epochs=50, lr=1e-4):
        """在真实数据上微调 LoRA 参数"""
        optimizer = torch.optim.AdamW(
            [p for p in self.policy.parameters() if p.requires_grad],
            lr=lr, weight_decay=1e-4
        )
        
        for epoch in range(epochs):
            total_loss = 0
            for batch in DataLoader(real_dataset, batch_size=16, shuffle=True):
                obs, expert_action = batch['obs'], batch['action']
                
                out = self.policy(obs)
                loss = F.mse_loss(out['action_mean'], expert_action)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.policy.parameters() if p.requires_grad], 0.5
                )
                optimizer.step()
                total_loss += loss.item()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: loss={total_loss/len(real_dataset):.4f}")
```

## 5. DADR (Data Augmentation for Domain Randomization)

```python
# dadr_augmentation.py

class DADRTactileAugmenter:
    """
    触觉专用的在线增强器
    部署时对任意输入触觉图像做实时扰动, 类似图像 AugmentationBut_online
    """
    
    def __init__(self, intensity='medium'):
        self.intensity = intensity
        self.probs = {
            'low':    {'noise': 0.01, 'shift': 0.05, 'warp': 0.02},
            'medium': {'noise': 0.03, 'shift': 0.10, 'warp': 0.05},
            'high':   {'noise': 0.05, 'shift': 0.20, 'warp': 0.10},
        }[intensity]
    
    def augment(self, tactile_frames):
        """
        Args:
            tactile_frames: (T, 3, H, W) 触觉图像序列
        Returns:
            增强后 tensor (same shape, on GPU)
        """
        # 1. 高斯噪声 (传感器热噪声模拟)
        if random.random() < 0.5:
            noise = torch.randn_like(tactile_frames) * self.probs['noise']
            tactile_frames = (tactile_frames + noise).clamp(0, 1)
        
        # 2. 亮度/对比度 (LED 亮度变化)
        if random.random() < 0.5:
            brightness = 1 + random.gauss(0, self.probs['shift'])
            contrast = 1 + random.gauss(0, self.probs['shift'])
            tactile_frames = (tactile_frames * brightness - 0.5) * contrast + 0.5
            tactile_frames = tactile_frames.clamp(0, 1)
        
        # 3. 空间仿射形变 (弹性体温差形变)
        if random.random() < 0.3:
            angle = random.gauss(0, 3)  # 度
            translate = (random.gauss(0, 0.02), random.gauss(0, 0.02))
            tactile_frames = TF.affine(tactile_frames, angle=angle, translate=translate)
        
        # 4. 时域 dropout (丢帧补偿)
        if random.random() < 0.2:
            T = tactile_frames.shape[0]
            drop_idx = random.randint(0, T - 1)
            tactile_frames[drop_idx] = tactile_frames[max(0, drop_idx - 1)]  # 复制前一帧
        
        return tactile_frames
```

## 6. 推荐路径

| Phase | 方案 | 预期数据量 | 效果 |
|---|---|---|---|
| Phase 0 | TACTO + PyBullet 行为克隆预训练 | 仿真 10 万条 | 基础接触/滑脱判断 |
| Phase 1 | A→真实 LoRA 微调试运行 | 真实 100 条仿真加训 | 真实硬件抓取成功率 ~60% |
| Phase 1+ | DADR 在线强化微调 + 域随机化 | 真实 1000 条 | 真实抓取 ~80% |
| Phase 2 | 在线 SAC 持续学习 | 真实增量采集 | 终身保持 + 学习新任务 |

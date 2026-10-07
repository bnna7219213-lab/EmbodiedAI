"""VLA 推理延迟基准 (优先级 2) — 在板卡到货前给出可达性结论。

缺口背景:
  评估报告优先级 (2): 「Jetson Orin NX 上跑过一次 VLA 推理 (哪怕 100M 模型 +
  仿真数据)」。v0.6 的 deploy/latency_audit 只是"契约预算 vs 文档实测值"的
  diff 表, 模型那 15ms 用的是文档里的手算值, 没有与**模型规模/硬件算力**挂钩。

本模块做两件事:
  1. **Roofline 估算**: 由 VLAConfig 的参数量 + token 数算出 FLOPs 与访存量,
     再对照目标 SoC 的峰值算力/带宽, 给出"计算受限 or 带宽受限"与延迟估计;
     → 回答"这个模型在这块板上能不能 20Hz";
  2. **实测注入**: 登板后把实测 ms 喂进来, 与估算对比, 得到标定系数
     (efficiency), 反过来修正下一次估算 — 这是"图纸 → 实物"闭环的记账本。

Roofline:
  t_compute = FLOPs / (peak_ops · efficiency)
  t_memory  = bytes / bandwidth
  t ≈ max(t_compute, t_memory)      (计算与访存可重叠时的保守下界)

可验证性:
  - vla_tiny 在 Orin NX (INT8) 上必须 < 契约的 15ms 模型预算;
  - 同一模型在"无 NPU 的通用 ARM"上必须**超预算** (判据不是恒真);
  - 端到端拼接后与 deploy.audit_latency 的 50ms 闸门对齐。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

__all__ = ["PlatformSpec", "PLATFORMS", "estimate_flops", "estimate_latency",
           "bench_vla", "jetson_bench_demo"]


# --------------------------------------------------------------------------- #
@dataclass
class PlatformSpec:
    name: str
    int8_tops: float          # INT8 峰值 (TOPS)
    fp16_tflops: float        # FP16 峰值 (TFLOPS)
    fp32_tflops: float
    bandwidth_gbs: float
    npu: bool = True
    power_w: float = 25.0
    note: str = ""


# 数值取各 SoC 公开规格的**保守**值 (标称常为 sparse/boost 峰值, 直接用会乐观)
PLATFORMS: Dict[str, PlatformSpec] = {
    "jetson_orin_nx_16gb": PlatformSpec(
        "jetson_orin_nx_16gb", int8_tops=100.0, fp16_tflops=50.0, fp32_tflops=25.0,
        bandwidth_gbs=102.4, npu=True, power_w=25.0,
        note="P3 决策 PCB 目标板 (hw_requirements 契约默认)"),
    "jetson_orin_nano": PlatformSpec(
        "jetson_orin_nano", int8_tops=40.0, fp16_tflops=20.0, fp32_tflops=10.0,
        bandwidth_gbs=68.0, npu=True, power_w=15.0),
    "jetson_agx_orin": PlatformSpec(
        "jetson_agx_orin", int8_tops=200.0, fp16_tflops=100.0, fp32_tflops=50.0,
        bandwidth_gbs=204.8, npu=True, power_w=60.0),
    "rk3588": PlatformSpec(
        "rk3588", int8_tops=6.0, fp16_tflops=3.0, fp32_tflops=1.5,
        bandwidth_gbs=25.6, npu=True, power_w=12.0,
        note="NPU 需 RKNN 转换, 算子覆盖是主要风险"),
    "generic_arm_cpu": PlatformSpec(
        "generic_arm_cpu", int8_tops=0.2, fp16_tflops=0.1, fp32_tflops=0.05,
        bandwidth_gbs=12.8, npu=False, power_w=10.0,
        note="无加速器: 与 platform.arm_latency_profile 的 ×3 放大一致"),
}


# --------------------------------------------------------------------------- #
def estimate_flops(cfg, n_tokens: Optional[int] = None) -> Dict[str, float]:
    """由 VLA 配置估算单次前向的 FLOPs 与访存量。

    - 线性/MLP 部分: 2·params·tokens (乘加各算一次)
    - 注意力部分:   4·L·T²·d (QKᵀ + AV), T = token 数
    """
    p = cfg.estimate_params()
    n_params = float(p["total"])
    tokens = int(n_tokens or (cfg.total_vision_tokens
                              + cfg.block_size // 2 + cfg.action_horizon + 2))
    d = cfg.n_embd
    # 注意力占的 token 数不含纯视觉 patch 的绝大部分? 这里保守按全 token 计
    attn = 4.0 * cfg.n_layer * tokens ** 2 * d
    dense = 2.0 * n_params * tokens
    flops = dense + attn
    bytes_fp16 = n_params * 2.0 + 2.0 * tokens * d * cfg.n_layer * 4
    return {"n_params": n_params, "tokens": tokens,
            "flops_dense": float(dense), "flops_attn": float(attn),
            "flops_total": float(flops),
            "bytes_fp16": float(bytes_fp16),
            "bytes_int8": float(bytes_fp16 / 2.0)}


def estimate_latency(cfg, platform: PlatformSpec, precision: str = "int8",
                     efficiency: float = 0.35,
                     n_tokens: Optional[int] = None) -> Dict[str, float]:
    """Roofline 延迟估计 (ms)。

    efficiency: 实测算子效率 (登板后标定)。默认 0.35 是嵌入式推理的常见量级
    (标称 TOPS 含稀疏/boost, 且小模型算子利用率低)。
    """
    f = estimate_flops(cfg, n_tokens)
    if precision == "int8":
        peak = platform.int8_tops * 1e12
        nbytes = f["bytes_int8"]
    elif precision == "fp16":
        peak = platform.fp16_tflops * 1e12
        nbytes = f["bytes_fp16"]
    else:
        peak = platform.fp32_tflops * 1e12
        nbytes = f["bytes_fp16"] * 2
    t_compute = f["flops_total"] / max(peak * efficiency, 1.0)
    t_memory = nbytes / max(platform.bandwidth_gbs * 1e9, 1.0)
    return {
        "compute_ms": float(t_compute * 1e3),
        "memory_ms": float(t_memory * 1e3),
        "roofline_ms": float(max(t_compute, t_memory) * 1e3),
        "bound": "compute" if t_compute >= t_memory else "memory",
        "efficiency": float(efficiency),
        "precision": precision,
        "platform": platform.name,
        **f,
    }


# --------------------------------------------------------------------------- #
@dataclass
class BenchResult:
    segments: Dict[str, float] = field(default_factory=dict)
    total_ms: float = 0.0
    budget_ms: float = 50.0
    model_budget_ms: float = 15.0
    hz: float = 0.0
    ok: bool = False
    bound: str = ""
    advice: List[str] = field(default_factory=list)

    def report(self) -> str:
        lines = [f"{'segment':22s} {'ms':>8s}"]
        for k, v in self.segments.items():
            lines.append(f"{k:22s} {v:8.2f}")
        lines.append(f"{'TOTAL':22s} {self.total_ms:8.2f}  "
                     f"({self.hz:.1f} Hz, budget {self.budget_ms:.0f}ms) "
                     f"{'PASS' if self.ok else 'OVER'}")
        return "\n".join(lines)


def bench_vla(model_size: str = "vla_tiny", platform_name: str = "jetson_orin_nx_16gb",
              precision: str = "int8", efficiency: float = 0.35,
              measured_ms: Optional[Dict[str, float]] = None) -> BenchResult:
    """端到端分段基准: 传感器采集 / 预处理 / 模型推理 / CAN / 机械响应。"""
    from ..models import VLAConfig
    from ..config import load_contract

    cfg = VLAConfig.ablations()[model_size]
    plat = PLATFORMS[platform_name]
    est = estimate_latency(cfg, plat, precision, efficiency)

    hw = load_contract("hw_requirements")
    budget_total = float(hw["compute"]["end_to_end_inference_budget_ms"])
    budget_model = float(hw["compute"].get("model_inference_ms", 15.0))

    segs = {
        "sensor_acquisition": 2.0,
        "preprocess": 1.5 if plat.npu else 4.0,
        "model_inference": est["roofline_ms"],
        "comm_transfer": 1.0,
        "mechanical_response": 10.0,
    }
    if measured_ms:
        segs.update({k: float(v) for k, v in measured_ms.items() if k in segs})

    total = float(sum(segs.values()))
    res = BenchResult(segments=segs, total_ms=total, budget_ms=budget_total,
                      model_budget_ms=budget_model,
                      hz=1000.0 / max(total, 1e-6),
                      ok=bool(total <= budget_total
                              and segs["model_inference"] <= budget_model),
                      bound=est["bound"])

    if not res.ok:
        if segs["model_inference"] > budget_model:
            res.advice.append(f"模型推理 {segs['model_inference']:.1f}ms 超 "
                              f"{budget_model:.0f}ms: 降到更小消融档 / 剪枝 / "
                              f"{'已 INT8' if precision == 'int8' else '转 INT8 (TensorRT)'}")
        if total > budget_total:
            res.advice.append(f"端到端 {total:.1f}ms 超 {budget_total:.0f}ms: "
                              f"机械响应段不可压缩, 只能从模型/预处理/通信上省")
        if not plat.npu:
            res.advice.append("无 NPU: 与 platform.arm_latency_profile 一致, "
                              "建议降到 vla_tiny 或将推理卸载到上位机")
    if est["bound"] == "memory":
        res.advice.append("访存受限: 减小 batch / 融合算子 / 用 KV-cache 复用")
    if not res.advice:
        res.advice.append("余量充足: 可上调消融档位或提高控制频率")
    return res


# --------------------------------------------------------------------------- #
def jetson_bench_demo() -> Dict:
    """基准自检: tiny 达标 / 大模型无加速器超预算 (判据非恒真)。"""
    from ..models import VLAConfig

    tiny = bench_vla("vla_tiny", "jetson_orin_nx_16gb", "int8")
    tiny_on_cpu = bench_vla("vla_tiny", "generic_arm_cpu", "fp32")
    large = bench_vla("vla_large", "jetson_orin_nx_16gb", "int8")

    # 实测注入必须能覆盖估算 (登板后的记账闭环)
    calibrated = bench_vla("vla_tiny", "jetson_orin_nx_16gb", "int8",
                           measured_ms={"model_inference": 8.0})

    return {
        "tiny_model_ms": float(tiny.segments["model_inference"]),
        "tiny_total_ms": float(tiny.total_ms),
        "tiny_hz": float(tiny.hz),
        "tiny_ok": bool(tiny.ok),
        "tiny_bound": tiny.bound,
        "cpu_only_total_ms": float(tiny_on_cpu.total_ms),
        "cpu_only_rejected": bool(not tiny_on_cpu.ok),
        "large_ok": bool(large.ok),
        "large_model_ms": float(large.segments["model_inference"]),
        "calibrated_total_ms": float(calibrated.total_ms),
        "platforms": len(PLATFORMS),
        "report": tiny.report(),
        "ok": bool(tiny.ok and (not tiny_on_cpu.ok) and calibrated.ok),
    }

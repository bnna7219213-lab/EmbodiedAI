"""延迟预算审计 (deploy) — hw_requirements 声明预算 vs 分段实测的自动 diff。

吸收 ECAD-MCAD NG12 (MLOps/量化/延迟核验): 部署前的最后一道闸——
nn-to-hardware-pipeline.md 的人算预算 (2+15+1+1+10 ≈ 29ms) 改为契约声明 → 自动核对。

数据源:
  - contracts/hw_requirements.yaml: compute 段 (50ms 端到端预算 / 模型 15ms)
  - 分段实测 (可注入): 传感器采集 / 模型推理 / CAN 传输 / FOC 响应 / 机械响应
越界项分级: INFO (<80% 预算) / WARNING (80-100%) / BLOCKING (>100%)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# nn-to-hardware-pipeline.md §2 的时间线 (文档实测基准, 用于默认分段)
DOC_BASELINE_MS = {
    "sensor_acquisition": 2.0,
    "model_inference": 15.0,
    "comm_transfer": 1.0,
    "foc_response": 1.0,
    "mechanical_response": 10.0,
}


@dataclass
class LatencySegment:
    name: str
    measured_ms: float
    budget_ms: float

    @property
    def utilization(self) -> float:
        return self.measured_ms / max(self.budget_ms, 1e-9)

    @property
    def verdict(self) -> str:
        u = self.utilization
        if u > 1.0:
            return "BLOCKING"
        if u > 0.8:
            return "WARNING"
        return "INFO"


@dataclass
class LatencyAuditResult:
    total_budget_ms: float
    total_measured_ms: float
    segments: List[LatencySegment] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(s.verdict != "BLOCKING" for s in self.segments)

    def report_lines(self) -> List[str]:
        lines = [f"{'segment':24s} {'measured':>10s} {'budget':>8s} {'util':>7s}  verdict"]
        for s in self.segments:
            lines.append(f"{s.name:24s} {s.measured_ms:9.2f}ms {s.budget_ms:7.1f}ms "
                         f"{s.utilization * 100:6.1f}%  {s.verdict}")
        lines.append(f"{'TOTAL':24s} {self.total_measured_ms:9.2f}ms {self.total_budget_ms:7.1f}ms "
                     f"{self.total_measured_ms / self.total_budget_ms * 100:6.1f}%  "
                     f"{'PASS ✅' if self.ok else 'BLOCKING ❌'}")
        return lines


def audit_latency(measured: Optional[Dict[str, float]] = None,
                  end_to_end_budget_from_contract: bool = True,
                  budget_overrides: Optional[Dict[str, float]] = None) -> LatencyAuditResult:
    """自动延迟审计。

    measured: 分段实测 (缺省用文档基准值占位, 真机部署后替换)
    end_to_end_budget_from_contract: True 时总预算取 hw_requirements.compute.
        end_to_end_inference_budget_ms (50ms) — 契约唯一真源。
    budget_overrides: 分段预算覆盖 (v0.6: Linux on ARM 无 NPU/GPU 时由
        platform.arm_budget_overrides() 给出保守放大值)。只覆盖给定键。
    """
    measured = measured or {}
    if end_to_end_budget_from_contract:
        from ..config import load_contract
        hw = load_contract("hw_requirements")
        total_budget = hw["compute"]["end_to_end_inference_budget_ms"]
        model_budget = hw["compute"].get("model_inference_ms", 15.0)
    else:
        total_budget = 50.0
        model_budget = 15.0

    # 分段预算分配 (端到端 50ms 内): 其余分段沿用文档基准
    budgets = {
        "sensor_acquisition": DOC_BASELINE_MS["sensor_acquisition"],
        "model_inference": model_budget,
        "comm_transfer": DOC_BASELINE_MS["comm_transfer"],
        "foc_response": DOC_BASELINE_MS["foc_response"],
        "mechanical_response": DOC_BASELINE_MS["mechanical_response"],
    }
    if budget_overrides:
        budgets.update({k: float(v) for k, v in budget_overrides.items() if k in budgets})
    segs = []
    for name, budget in budgets.items():
        m = measured.get(name, DOC_BASELINE_MS[name])
        segs.append(LatencySegment(name, m, budget))
    total_measured = sum(s.measured_ms for s in segs)
    return LatencyAuditResult(total_budget, total_measured, segs)

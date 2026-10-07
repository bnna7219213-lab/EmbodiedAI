"""安全认证工程化 (G10) — v0.7 把"形式化安全代码"推进到可交付的认证证据链。

缺口背景:
  评估报告: 「安全认证工程化 (G10) 完成度 ~10% — STL 规约 + 预算闭环代码已完成,
  但无任何认证流程文档或第三方评测」。代码能证明"某个性质成立", 认证要证明
  "**整套系统**在规定条件下可接受" — 缺的是危害分析、SIL 定量计算、V&V 计划
  与需求可追溯矩阵这套**证据链**。

本模块把证据链结构化 (纯 Python, 可导出 Markdown/JSON 供第三方评审):

  1. 危害分析 (HAZOP/FMEA 表): 危害 → 场景 → 严重度 → 风险 → 缓解 → 对应需求;
  2. SIL 定量: 各安全子系统的失效率 λ (FIT) / 诊断覆盖率 DC / 安全失效分数
     SFF → PFH (每小时危险失效概率) → 对照 IEC 61508 SIL 目标区间;
     **架构约束单独判**: SFF<90% 或 HFT=0 时, 即使 PFH 达标也不允许宣称 SIL2;
  3. V&V 计划: 每条安全需求必须至少一条验证活动 (分析/测试/审查);
  4. 需求可追溯矩阵 (RTM): 需求 ↔ 设计 ↔ 实现 ↔ 测试, 覆盖率可断言。

对齐 contracts:
  - safety_budget.yaml: acceptance.sil2_pfh_target / measured_pfh / sff_percent
  - hw_requirements.yaml: 安全相关硬件通道数
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

__all__ = ["Hazard", "Subsystem", "SafetyRequirement", "VerificationActivity",
           "CertificationPackage", "pfh_total", "sil_from_pfh", "sff",
           "architecture_allows", "build_default_package", "certification_demo"]


# --------------------------------------------------------------------------- #
# IEC 61508 SIL 目标失效量 (PFH, 每小时)
SIL_PFH_BANDS = {
    1: (1e-6, 1e-5),
    2: (1e-7, 1e-6),
    3: (1e-8, 1e-7),
    4: (1e-9, 1e-8),
}

# 架构约束 (IEC 61508-2 Table 2/3, B 类子系统, Route 1H)
# (SFF 下限, 允许的硬件故障裕度 HFT)
ARCH_CONSTRAINTS_B = [
    (0.0, 0, 1),     # SFF < 60% → 最高 SIL1, HFT 0
    (60.0, 0, 1),
    (90.0, 1, 2),    # SFF ≥ 90% + HFT=1 → SIL2
    (99.0, 1, 3),
]


@dataclass
class Hazard:
    hid: str
    scenario: str
    severity: int            # 1..4 (轻微 → 死亡/不可逆)
    occurrence: int          # 1..5 (极低 → 极高)
    mitigation: str
    req_ids: List[str] = field(default_factory=list)

    @property
    def risk_level(self) -> str:
        r = self.severity * self.occurrence
        if r >= 15:
            return "HIGH"
        if r >= 8:
            return "MEDIUM"
        return "LOW"

    @property
    def iso_clause(self) -> str:
        return {"HIGH": "ISO 13482 §5.3 (STO)", "MEDIUM": "ISO 13482 §5.2",
                "LOW": "ISO 13482 §5.1"}[self.risk_level]


@dataclass
class Subsystem:
    """安全相关子系统 (用于 SIL 定量)。"""
    name: str
    lambda_fit: float        # 总失效率 (FIT = 1e-9 /h)
    dc_percent: float        # 诊断覆盖率 %
    safe_fault_percent: float = 0.0   # 安全失效 (λ_S) 占比 %
    hft: int = 0             # 硬件故障裕度
    evidence: str = ""

    @property
    def lambda_du(self) -> float:
        """危险**未检测**失效率 (FIT) = λ·(1−λ_S%)·(1−DC%)。"""
        return self.lambda_fit * (1 - self.safe_fault_percent / 100.0) \
            * (1 - self.dc_percent / 100.0)

    @property
    def lambda_dd(self) -> float:
        return self.lambda_fit * (1 - self.safe_fault_percent / 100.0) \
            * (self.dc_percent / 100.0)


def pfh_total(subs: Sequence[Subsystem]) -> float:
    """PFH (每小时) = Σ λ_DU × 1e-9。"""
    return float(sum(s.lambda_du for s in subs) * 1e-9)


def sff(subs: Sequence[Subsystem]) -> float:
    """安全失效分数 % = (Σλ_S + Σλ_DD) / Σλ_total。"""
    tot = sum(s.lambda_fit for s in subs)
    if tot <= 0:
        return 0.0
    safe = sum(s.lambda_fit * s.safe_fault_percent / 100.0 for s in subs)
    dd = sum(s.lambda_dd for s in subs)
    return float(100.0 * (safe + dd) / tot)


def sil_from_pfh(pfh: float) -> int:
    """由 PFH 反查可达 SIL (0 = 低于 SIL1)。"""
    for sil, (lo, hi) in sorted(SIL_PFH_BANDS.items(), reverse=True):
        if lo <= pfh < hi:
            return sil
    return 4 if pfh < 1e-9 else 0


def architecture_allows(subs: Sequence[Subsystem], target_sil: int = 2) -> bool:
    """架构约束: SFF 与 HFT 决定**允许宣称**的最高 SIL (不只看 PFH)。"""
    sff_pct = sff(subs)
    min_hft = min((s.hft for s in subs), default=0)
    best = 0
    for sff_lo, hft_lo, sil in ARCH_CONSTRAINTS_B:
        if sff_pct >= sff_lo and min_hft >= hft_lo:
            best = max(best, sil)
    return bool(best >= target_sil)


# --------------------------------------------------------------------------- #
@dataclass
class SafetyRequirement:
    rid: str
    text: str
    derives_from: List[str] = field(default_factory=list)   # 危害 ID
    allocated_to: str = ""                                   # 设计/模块
    sil: int = 2


@dataclass
class VerificationActivity:
    vid: str
    rid: str                  # 覆盖的需求
    kind: str                 # "analysis" | "test" | "review" | "hil"
    method: str
    acceptance: str
    status: str = "planned"   # planned | passed | failed


# --------------------------------------------------------------------------- #
@dataclass
class CertificationPackage:
    name: str
    hazards: List[Hazard] = field(default_factory=list)
    subsystems: List[Subsystem] = field(default_factory=list)
    requirements: List[SafetyRequirement] = field(default_factory=list)
    activities: List[VerificationActivity] = field(default_factory=list)
    target_sil: int = 2
    pfh_target: float = 1e-6

    # ---------------------------------------------------------------- #
    @property
    def pfh(self) -> float:
        return pfh_total(self.subsystems)

    @property
    def sff_percent(self) -> float:
        return sff(self.subsystems)

    @property
    def achieved_sil(self) -> int:
        return sil_from_pfh(self.pfh)

    @property
    def rtm_coverage(self) -> float:
        """需求 → 验证活动的覆盖率。"""
        if not self.requirements:
            return 0.0
        covered = {a.rid for a in self.activities}
        return float(sum(1 for r in self.requirements if r.rid in covered)
                     / len(self.requirements))

    @property
    def hazard_coverage(self) -> float:
        """每条危害是否至少被一条需求覆盖。"""
        if not self.hazards:
            return 0.0
        refs = {h for r in self.requirements for h in r.derives_from}
        return float(sum(1 for h in self.hazards if h.hid in refs) / len(self.hazards))

    def evaluate(self) -> Dict[str, object]:
        ok_pfh = bool(self.pfh <= self.pfh_target)
        ok_arch = architecture_allows(self.subsystems, self.target_sil)
        coverage = self.rtm_coverage
        return {
            "pfh_per_hour": self.pfh,
            "pfh_target": self.pfh_target,
            "pfh_ok": ok_pfh,
            "sff_percent": round(self.sff_percent, 2),
            "achieved_sil": self.achieved_sil,
            "target_sil": self.target_sil,
            "architecture_ok": ok_arch,
            "rtm_coverage": round(coverage, 3),
            "hazard_coverage": round(self.hazard_coverage, 3),
            "n_hazards": len(self.hazards),
            "n_requirements": len(self.requirements),
            "n_activities": len(self.activities),
            "verdict": "PASS" if (ok_pfh and ok_arch and coverage >= 1.0
                                  and self.hazard_coverage >= 1.0) else "FAIL",
            "gaps": self._gaps(ok_pfh, ok_arch, coverage),
        }

    def _gaps(self, ok_pfh: bool, ok_arch: bool, coverage: float) -> List[str]:
        g = []
        if not ok_pfh:
            g.append(f"PFH {self.pfh:.2e} 超目标 {self.pfh_target:.0e}: "
                     f"提高诊断覆盖率或降额")
        if not ok_arch:
            g.append(f"架构约束不满足 SIL{self.target_sil}: SFF {self.sff_percent:.1f}% "
                     f"/ HFT 不足 — 需双通道冗余")
        if coverage < 1.0:
            missing = [r.rid for r in self.requirements
                       if r.rid not in {a.rid for a in self.activities}]
            g.append(f"需求缺少验证活动: {missing}")
        if self.hazard_coverage < 1.0:
            g.append("存在未派生安全需求的危害")
        return g

    # ---------------------------------------------------------------- #
    def to_markdown(self) -> str:
        ev = self.evaluate()
        out = [f"# {self.name} — 安全认证证据包", ""]
        out.append(f"- 目标 SIL{self.target_sil} / PFH 目标 {self.pfh_target:.0e} /h")
        out.append(f"- 实测 PFH {ev['pfh_per_hour']:.3e} /h → SIL{ev['achieved_sil']}")
        out.append(f"- SFF {ev['sff_percent']}% / RTM 覆盖 {ev['rtm_coverage'] * 100:.0f}%")
        out.append(f"- **结论: {ev['verdict']}**")
        out.append("")
        out.append("## 危害分析 (HAZOP/FMEA)")
        out.append("| ID | 场景 | S | O | 风险 | 缓解 | 需求 |")
        out.append("|---|---|---|---|---|---|---|")
        for h in self.hazards:
            out.append(f"| {h.hid} | {h.scenario} | {h.severity} | {h.occurrence} "
                       f"| {h.risk_level} | {h.mitigation} | {', '.join(h.req_ids)} |")
        out.append("")
        out.append("## 子系统 (SIL 定量)")
        out.append("| 子系统 | λ(FIT) | DC% | λ_S% | HFT | λ_DU(FIT) |")
        out.append("|---|---|---|---|---|---|")
        for s in self.subsystems:
            out.append(f"| {s.name} | {s.lambda_fit:.0f} | {s.dc_percent:.0f} "
                       f"| {s.safe_fault_percent:.0f} | {s.hft} | {s.lambda_du:.2f} |")
        out.append("")
        out.append("## 需求 ↔ 验证追溯")
        out.append("| 需求 | 文本 | 派生自 | 分配到 | 验证活动 |")
        out.append("|---|---|---|---|---|")
        for r in self.requirements:
            vs = [a.vid for a in self.activities if a.rid == r.rid]
            out.append(f"| {r.rid} | {r.text} | {', '.join(r.derives_from)} "
                       f"| {r.allocated_to} | {', '.join(vs) or '**缺失**'} |")
        out.append("")
        if ev["gaps"]:
            out.append("## 缺口")
            out.extend(f"- {g}" for g in ev["gaps"])
        return "\n".join(out)


# --------------------------------------------------------------------------- #
def build_default_package() -> CertificationPackage:
    """EmbodiedAI 关节驱动 + 安全链的**默认**认证包 (数值为设计值/器件手册量级)。"""
    hazards = [
        Hazard("H-01", "过流时未能关断 → 关节持续输出扭矩夹伤", 4, 3,
               "硬件比较器 + SR 锁存独立关断 (不依赖 MCU)", ["SR-01"]),
        Hazard("H-02", "MCU 死机/固件跑飞 → 关断指令发不出", 4, 2,
               "关断路径全硬件; 看门狗 + 心跳丢失即 STO", ["SR-01", "SR-04"]),
        Hazard("H-03", "继电器触点粘连 → 机械隔离未建立", 4, 2,
               "双通道冗余触点 + 粘连检测 (每周期自检)", ["SR-02"]),
        Hazard("H-04", "停止角超限 → 缠绕/撞击人体", 3, 3,
               "速度包络限制 + 停止角预算闭环", ["SR-03"]),
        Hazard("H-05", "编码器漂移 → 位置闭环失控", 3, 2,
               "双编码器交叉校验 +  plausibility 检查", ["SR-04"]),
        Hazard("H-06", "CAN 报文错帧 → 执行错误目标", 3, 2,
               "DBC 定义的 CRC8 + 滚动计数 + 超时 STO", ["SR-05"]),
    ]
    # λ 取工业器件手册的常见量级; DC 由自检机制决定
    subs = [
        Subsystem("硬件比较器+锁存", lambda_fit=150.0, dc_percent=99.0,
                  safe_fault_percent=10.0, hft=1,
                  evidence="P4 PCB 上电自检 + 周期性注入测试"),
        Subsystem("栅极驱动", lambda_fit=120.0, dc_percent=90.0,
                  safe_fault_percent=20.0, hft=1),
        # 继电器是**机械**部件, λ 比半导体高一到两个数量级 — 早期版本给了
        # 600 FIT, 结果"关掉全部诊断覆盖率 PFH 仍达标", 判据恒真, 失去意义。
        Subsystem("安全继电器(双通道)", lambda_fit=1500.0, dc_percent=95.0,
                  safe_fault_percent=30.0, hft=1,
                  evidence="触点粘连检测; AgSnO2 侵蚀寿命需降额 (CRITICAL)"),
        Subsystem("看门狗 MCU", lambda_fit=300.0, dc_percent=99.0,
                  safe_fault_percent=25.0, hft=1),
        # 安全通信按双通道或 CRC+滚动计数等效冗余计 (HFT=1)
        Subsystem("CAN-FD 链路", lambda_fit=80.0, dc_percent=99.9,
                  safe_fault_percent=50.0, hft=1,
                  evidence="CRC8 + 滚动计数 + 超时 STO, safety_budget 已定义"),
    ]
    reqs = [
        SafetyRequirement("SR-01", "过流后电气关断 ≤ 1µs 且不依赖固件",
                          ["H-01", "H-02"], "P4 电机驱动 PCB 硬件比较器"),
        SafetyRequirement("SR-02", "总断开 ≤ 20ms 且机械隔离可验证",
                          ["H-03"], "SAFETY_SHUTDOWN 继电器"),
        SafetyRequirement("SR-03", "STO 停止角 ≤ 30°", ["H-04"],
                          "L4 速度包络 + budget_closure"),
        SafetyRequirement("SR-04", "位置反馈失效须在 1 个控制周期内进入 STO",
                          ["H-02", "H-05"], "L3 反射伺服 + 看门狗"),
        SafetyRequirement("SR-05", "CAN 帧 CRC/校验失败即丢弃并计数, 超限 STO",
                          ["H-06"], "P5 互联 + CAN 驱动"),
    ]
    acts = [
        VerificationActivity("V-01", "SR-01", "hil",
                             "HIL-STO-01/02: 示波器测分流→栅极 EN 时延", "≤1µs"),
        VerificationActivity("V-02", "SR-02", "hil",
                             "HIL-STO-03: 触点两端电压 = 母线电压", "≤20ms"),
        VerificationActivity("V-03", "SR-03", "analysis",
                             "停止角解析计算 + 编码器实测", "≤30°"),
        VerificationActivity("V-04", "SR-04", "test",
                             "注入编码器断线/漂移故障", "1 周期内 STO"),
        VerificationActivity("V-05", "SR-05", "test",
                             "注入 CRC 错帧 / 丢帧 / 重放", "丢弃并计数"),
        VerificationActivity("V-06", "SR-01", "review",
                             "原理图 + PCB 布局审查 (关断路径独立供电)", "无单点失效"),
    ]
    return CertificationPackage("EmbodiedAI 关节安全链", hazards, subs, reqs, acts,
                                target_sil=2, pfh_target=1e-6)


def certification_demo() -> Dict:
    """认证包自检: PFH/架构/追溯 三项达标, 且判据非恒真。"""
    pkg = build_default_package()
    ev = pkg.evaluate()

    # 判据非恒真 ①: 诊断覆盖率归零 → PFH 必须超标
    blind = build_default_package()
    for s in blind.subsystems:
        s.dc_percent = 0.0
    ev_blind = blind.evaluate()

    # 判据非恒真 ②: 撤掉冗余 (HFT=0) → 架构约束必须拒绝 SIL2
    single = build_default_package()
    for s in single.subsystems:
        s.hft = 0
    ev_single = single.evaluate()

    # 判据非恒真 ③: 删掉一条需求的验证 → RTM 覆盖必须掉到 100% 以下
    holed = build_default_package()
    holed.activities = [a for a in holed.activities if a.rid != "SR-05"]
    ev_holed = holed.evaluate()

    return {
        "pfh_per_hour": float(ev["pfh_per_hour"]),
        "pfh_ok": bool(ev["pfh_ok"]),
        "sff_percent": float(ev["sff_percent"]),
        "achieved_sil": int(ev["achieved_sil"]),
        "architecture_ok": bool(ev["architecture_ok"]),
        "rtm_coverage": float(ev["rtm_coverage"]),
        "hazard_coverage": float(ev["hazard_coverage"]),
        "verdict": ev["verdict"],
        "no_diagnostics_rejected": bool(not ev_blind["pfh_ok"]),
        "single_channel_rejected": bool(not ev_single["architecture_ok"]),
        "missing_vv_rejected": bool(ev_holed["rtm_coverage"] < 1.0),
        "markdown_chars": len(pkg.to_markdown()),
        "ok": bool(ev["verdict"] == "PASS" and not ev_blind["pfh_ok"]
                   and not ev_single["architecture_ok"]
                   and ev_holed["rtm_coverage"] < 1.0),
    }

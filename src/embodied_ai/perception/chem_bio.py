"""化学生化信号仿真传感 + 触觉融合 (plan_v6 Sprint N: N4)。

**缺口背景（自我进化的感知缺口）**
    用户口径: "触觉占具身比例很大，只有视觉无法产生与人类相同级别的智能，
    想要完成自我进化，必要要触觉与基于触觉的化学生化信号仿真传感融合"。
    项目此前只有视觉/触觉/力传感 —— 缺"化学生化"维度（气味/气体、pH、湿度
    等表面/环境化学状态），材质判断只能靠视觉+力, 距人类级多模态感知有缺口。

**设计（最小可用，不过度设计）**
    - `ChemBioSensor`: 化学生化传感阵列仿真 ——
        · e-nose 气体阵列 (MOX 阵列响应向量, 8 通道)
        · pH / 湿度 / 温度 标量
      输出 `ChemVector`（带 provenance 标注）
    - `MaterialEstimate`: 触觉+化学生化融合的材质/表面状态估计
    - `ChemTactileFusion.fuse(chem_vec, tactile_feats)`: 规则式融合
      （湿度→湿滑、pH→腐蚀性、气体→材质类别先验 + 触觉刚度/摩擦佐证）
      → MaterialEstimate, 供 D4 数据回流（自我进化）与决策输入

**provenance（诚实边界）**
    全部传感响应为 **synthetic**（合成仿真）; 真实 e-nose/pH/湿度标定属
    **待领域验证**。本模块不宣称任何真实化学传感精度。

**不做的（诚实边界）**
    - 不做真实化学传感器驱动/标定（Phase P 硬件到货后）
    - 不做化学反应动力学仿真（过度设计）
    - 不做学习式材质分类（规则式最小实现, 学习属 D4 训练后）

**反证要求（测试守护）**
    ChemVector 形状固定; 融合输出确定可复现; 输入变则估计变（防恒定）;
    坏形状报错; 高湿度/低 pH 场景触发对应表面状态标注。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


#: e-nose 气体阵列通道数 (MOX 阵列, synthetic)
ENOSE_CHANNELS = 8


@dataclass
class ChemVector:
    """化学生化传感读数 — e-nose 阵列 + pH/湿度/温度。"""
    enose: np.ndarray            # (ENOSE_CHANNELS,) 气体阵列响应
    ph: float                    # pH 值 0-14
    humidity: float              # 相对湿度 0-1
    temperature_c: float         # 温度 ℃
    provenance: str = "synthetic"

    def __post_init__(self):
        e = np.asarray(self.enose, dtype=np.float64).ravel()
        if e.size != ENOSE_CHANNELS:
            raise ValueError(f"enose 应为 {ENOSE_CHANNELS} 维, 得 {e.size}")
        self.enose = e
        if not (0.0 <= self.humidity <= 1.0):
            raise ValueError(f"humidity 应在 [0,1], 得 {self.humidity}")

    def to_dict(self) -> dict:
        return {"enose": self.enose.tolist(), "ph": self.ph,
                "humidity": self.humidity,
                "temperature_c": self.temperature_c,
                "provenance": self.provenance}


@dataclass
class MaterialEstimate:
    """触觉+化学生化融合的材质/表面状态估计。"""
    material: str                # 材质类别 (规则式先验)
    surface: str                 # 表面状态: dry / wet / slippery / corrosive
    confidence: float            # 0-1 (规则强度, 非学习概率)
    evidence: list = field(default_factory=list)   # 证据链 (可解释)

    def to_dict(self) -> dict:
        return {"material": self.material, "surface": self.surface,
                "confidence": self.confidence, "evidence": list(self.evidence)}


class ChemBioSensor:
    """化学生化传感阵列仿真 (provenance=synthetic)。

    响应模型: 基线 + 场景偏置 + 高斯噪声 —— 合成可复现, 供融合链路测试。
    """

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def read(self, material_hint: str = "unknown",
             wet: bool = False, ph: float = 7.0) -> ChemVector:
        """一次采样。material_hint 影响 e-nose 响应模式 (合成)。"""
        base = np.full(ENOSE_CHANNELS, 0.2)
        if material_hint == "metal":
            base[[0, 3]] += 0.6
        elif material_hint == "plastic":
            base[[1, 5]] += 0.5
        elif material_hint == "organic":
            base[[2, 4, 6]] += 0.7
        if wet:
            base[7] += 0.8
        enose = base + self.rng.normal(0.0, 0.05, size=ENOSE_CHANNELS)
        humidity = 0.9 if wet else 0.3 + float(self.rng.uniform(0, 0.2))
        return ChemVector(
            enose=enose, ph=float(ph), humidity=humidity,
            temperature_c=22.0 + float(self.rng.uniform(-1, 1)),
            provenance="synthetic",
        )


class ChemTactileFusion:
    """化学生化 × 触觉 融合 — 材质/表面状态估计 (规则式)。

    证据链可解释: 每条判定都写明依据 (如 "humidity=0.92>0.8 → wet")。
    """

    #: 判定阈值 (assumed — 规则参数, 待领域验证)
    WET_THRESHOLD = 0.8
    CORROSIVE_PH = 4.0
    SLIP_TACTILE_STD = 0.3     # 触觉特征 std 低于此视为低摩擦 (滑)

    def fuse(self, chem: ChemVector,
             tactile_feats: Optional[np.ndarray] = None) -> MaterialEstimate:
        """融合化学生化读数 + 可选触觉特征 → 材质估计。"""
        evidence = []
        # 表面状态判定
        if chem.humidity > self.WET_THRESHOLD:
            surface = "wet"
            evidence.append(f"humidity={chem.humidity:.2f}>{self.WET_THRESHOLD} → wet")
        elif chem.ph < self.CORROSIVE_PH:
            surface = "corrosive"
            evidence.append(f"ph={chem.ph:.1f}<{self.CORROSIVE_PH} → corrosive")
        else:
            surface = "dry"
            evidence.append(f"humidity={chem.humidity:.2f}≤{self.WET_THRESHOLD} → dry")
        # 触觉佐证: 低波动 = 低摩擦 → 滑
        if tactile_feats is not None:
            t = np.asarray(tactile_feats, dtype=np.float64).ravel()
            if t.size and float(t.std()) < self.SLIP_TACTILE_STD:
                surface = "slippery"
                evidence.append(f"tactile_std={t.std():.2f}<"
                                f"{self.SLIP_TACTILE_STD} → slippery")
        # 材质类别: e-nose 主导通道
        dominant = int(np.argmax(chem.enose))
        material = ("metal" if dominant in (0, 3) else
                    "plastic" if dominant in (1, 5) else
                    "organic" if dominant in (2, 4, 6) else "unknown")
        evidence.append(f"enose 主导通道={dominant} → {material}")
        conf = float(min(1.0, 0.5 + 0.1 * len(evidence)))
        return MaterialEstimate(material=material, surface=surface,
                                confidence=conf, evidence=evidence)


def chem_demo() -> dict:
    """化学生化 × 触觉融合演示: 三种场景对照。"""
    sensor = ChemBioSensor(seed=5)
    fusion = ChemTactileFusion()
    scenarios = {
        "dry_metal": dict(material_hint="metal", wet=False, ph=7.0),
        "wet_organic": dict(material_hint="organic", wet=True, ph=7.0),
        "corrosive": dict(material_hint="plastic", wet=False, ph=2.5),
    }
    out = {}
    for name, kw in scenarios.items():
        vec = sensor.read(**kw)
        est = fusion.fuse(vec)
        out[name] = {"surface": est.surface, "material": est.material,
                     "confidence": round(est.confidence, 2),
                     "humidity": round(vec.humidity, 2)}
    assert out["dry_metal"]["surface"] == "dry"
    assert out["wet_organic"]["surface"] == "wet"
    assert out["corrosive"]["surface"] == "corrosive"
    assert out["dry_metal"]["material"] == "metal"
    return out

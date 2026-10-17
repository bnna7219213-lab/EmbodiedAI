"""N1 层注册表 测试 (plan_v6 Sprint N)。

反证: C1-C5 齐全; L1-L7 编号无增删 (docs/09 硬约束); module_refs 可 import;
GUI _layer_stack 与注册表同源; validate_registry 能抓住重复 id / 缺层。
"""
import pytest

from embodied_ai.center.layer_registry import (
    LAYER_REGISTRY,
    REQUIRED_CONTROL_IDS,
    as_dict,
    layers_by_stack,
    validate_registry,
)


class TestRegistryCompleteness:
    def test_c_stack_present(self):
        for cid in ("C1", "C2", "C3", "C4", "C5"):
            assert cid in LAYER_REGISTRY, f"缺少中枢决策层 {cid}"

    def test_c_stack_covers_user_requirements(self):
        """用户口径: 中枢决策层包含 nanoGPT、VLA、记忆、感知、信号处理。"""
        joined = " ".join(
            " ".join(s.composition) + " " + s.description
            for s in LAYER_REGISTRY.values() if s.id.startswith("C"))
        for kw in ("nanoGPT", "VLA", "记忆", "感知", "分频"):
            assert kw.lower() in joined.lower(), f"C 栈缺少『{kw}』"

    def test_l1_l7_numbering_untouched(self):
        assert tuple(REQUIRED_CONTROL_IDS) == ("L7", "L6", "L5", "L4", "L3", "L2", "L1")
        for lid in REQUIRED_CONTROL_IDS:
            assert lid in LAYER_REGISTRY
        # 不允许出现 L8+ (控制栈编号不得扩张)
        extras = [k for k in LAYER_REGISTRY
                  if k.startswith("L") and k not in REQUIRED_CONTROL_IDS]
        assert extras == [], f"控制栈编号被扩张: {extras}"

    def test_physical_safety_data_stacks(self):
        for pid in ("P1", "P2", "P3", "P4", "P5", "S",
                    "D1", "D2", "D3", "D4", "D5"):
            assert pid in LAYER_REGISTRY, f"缺少层 {pid}"

    def test_ids_unique(self):
        assert len(LAYER_REGISTRY) == len(set(LAYER_REGISTRY))


class TestModuleRefs:
    def test_module_refs_importable(self):
        import importlib
        for spec in LAYER_REGISTRY.values():
            for mod in spec.module_refs:
                importlib.import_module(mod)

    def test_validate_registry_passes(self):
        assert validate_registry() == 0


class TestConsumers:
    def test_layers_by_stack(self):
        by = layers_by_stack()
        assert set(by) == {"控制栈", "物理栈", "安全", "数据栈", "中枢决策栈"}
        assert len(by["中枢决策栈"]) == 5

    def test_gui_layer_stack_same_source(self):
        """GUI _layer_stack() 必须与注册表同源 (防再次漂移)。"""
        from embodied_ai.gui.server import _layer_stack
        assert _layer_stack() == as_dict()

    def test_as_dict_shape(self):
        d = as_dict()
        assert set(d) == set(LAYER_REGISTRY)
        assert all(isinstance(v, str) and v for v in d.values())


class TestNegative:
    def test_validate_catches_missing_c(self, monkeypatch):
        """反证: 缺 C3 时 validate_registry 必须失败 (不是恒真自检)。"""
        import embodied_ai.center.layer_registry as lr
        broken = {k: v for k, v in lr.LAYER_REGISTRY.items() if k != "C3"}
        monkeypatch.setattr(lr, "LAYER_REGISTRY", broken)
        with pytest.raises(AssertionError):
            lr.validate_registry()

"""N4 化学生化传感融合 测试 (plan_v6 Sprint N)。

反证: ChemVector 形状/范围校验; 融合确定可复现; 输入变则估计变;
高湿度→wet / 低 pH→corrosive / 低触觉波动→slippery; GUI routes 键齐全。
"""
import numpy as np
import pytest

from embodied_ai.perception.chem_bio import (
    ENOSE_CHANNELS,
    ChemBioSensor,
    ChemTactileFusion,
    ChemVector,
    MaterialEstimate,
    chem_demo,
)


class TestChemVector:
    def test_shape(self):
        v = ChemVector(enose=np.zeros(ENOSE_CHANNELS), ph=7.0,
                       humidity=0.5, temperature_c=22.0)
        assert v.enose.shape == (ENOSE_CHANNELS,)

    def test_bad_enose_shape_raises(self):
        with pytest.raises(ValueError):
            ChemVector(enose=np.zeros(3), ph=7.0, humidity=0.5,
                       temperature_c=22.0)

    def test_bad_humidity_raises(self):
        with pytest.raises(ValueError):
            ChemVector(enose=np.zeros(ENOSE_CHANNELS), ph=7.0,
                       humidity=1.5, temperature_c=22.0)

    def test_to_dict(self):
        v = ChemVector(enose=np.zeros(ENOSE_CHANNELS), ph=7.0,
                       humidity=0.5, temperature_c=22.0)
        d = v.to_dict()
        assert set(d) == {"enose", "ph", "humidity", "temperature_c",
                          "provenance"}
        assert d["provenance"] == "synthetic"


class TestChemBioSensor:
    def test_deterministic(self):
        a = ChemBioSensor(seed=3).read(material_hint="metal")
        b = ChemBioSensor(seed=3).read(material_hint="metal")
        assert np.allclose(a.enose, b.enose)

    def test_wet_raises_humidity(self):
        s = ChemBioSensor(seed=1)
        dry = s.read(wet=False)
        wet = s.read(wet=True)
        assert wet.humidity > dry.humidity

    def test_material_hint_changes_enose(self):
        s = ChemBioSensor(seed=2)
        metal = s.read(material_hint="metal")
        plastic = s.read(material_hint="plastic")
        assert not np.allclose(metal.enose, plastic.enose)


class TestChemTactileFusion:
    def test_dry_scenario(self):
        fusion = ChemTactileFusion()
        v = ChemVector(enose=np.array([0.8] + [0.1] * (ENOSE_CHANNELS - 1)),
                       ph=7.0, humidity=0.3, temperature_c=22.0)
        est = fusion.fuse(v)
        assert est.surface == "dry"
        assert est.material == "metal"

    def test_wet_scenario(self):
        fusion = ChemTactileFusion()
        v = ChemVector(enose=np.zeros(ENOSE_CHANNELS), ph=7.0,
                       humidity=0.95, temperature_c=22.0)
        est = fusion.fuse(v)
        assert est.surface == "wet"
        assert any("wet" in e for e in est.evidence)

    def test_corrosive_scenario(self):
        fusion = ChemTactileFusion()
        v = ChemVector(enose=np.zeros(ENOSE_CHANNELS), ph=2.0,
                       humidity=0.3, temperature_c=22.0)
        est = fusion.fuse(v)
        assert est.surface == "corrosive"

    def test_slippery_from_tactile(self):
        fusion = ChemTactileFusion()
        v = ChemVector(enose=np.zeros(ENOSE_CHANNELS), ph=7.0,
                       humidity=0.3, temperature_c=22.0)
        est = fusion.fuse(v, tactile_feats=np.full(4, 0.01))
        assert est.surface == "slippery"

    def test_input_change_changes_estimate(self):
        """反证: 输入变则估计必变 (防恒定输出)。"""
        fusion = ChemTactileFusion()
        v1 = ChemVector(enose=np.zeros(ENOSE_CHANNELS), ph=7.0,
                        humidity=0.3, temperature_c=22.0)
        v2 = ChemVector(enose=np.zeros(ENOSE_CHANNELS), ph=7.0,
                        humidity=0.95, temperature_c=22.0)
        assert fusion.fuse(v1).surface != fusion.fuse(v2).surface

    def test_material_est_to_dict(self):
        est = MaterialEstimate(material="metal", surface="dry",
                               confidence=0.6, evidence=["x"])
        assert set(est.to_dict()) == {"material", "surface", "confidence",
                                      "evidence"}

    def test_demo(self):
        out = chem_demo()
        assert out["dry_metal"]["surface"] == "dry"
        assert out["wet_organic"]["surface"] == "wet"
        assert out["corrosive"]["surface"] == "corrosive"


class TestGuiRoutes:
    def test_chem_route_registered(self):
        from embodied_ai.gui.server import DashboardState, routes
        r = routes(DashboardState())
        for path in ("/api/layer_stack", "/api/tactile", "/api/chem"):
            assert path in r, f"GUI 缺 route {path}"

    def test_chem_route_returns(self):
        from embodied_ai.gui.server import DashboardState, routes
        r = routes(DashboardState())
        out = r["/api/chem"]({})
        assert "chem" in out and "estimate" in out
        assert out["chem"]["provenance"] == "synthetic"

    def test_tactile_route_returns(self):
        from embodied_ai.gui.server import DashboardState, routes
        r = routes(DashboardState())
        out = r["/api/tactile"]({})
        assert set(out) >= {"mu_hat", "f_t", "slip_risk", "source"}

    def test_layer_stack_route_returns(self):
        from embodied_ai.gui.server import DashboardState, routes
        r = routes(DashboardState())
        out = r["/api/layer_stack"]({})
        assert "C5" in out["layers"]
        assert out["by_stack"]["中枢决策栈"] == ["C1", "C2", "C3", "C4", "C5"]

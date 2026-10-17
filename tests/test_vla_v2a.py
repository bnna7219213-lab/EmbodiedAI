"""M3 V-to-action 纯视觉决策路径 测试 (plan_v5 Sprint M)。

反证: v2a ≡ vla(lang=None) 等价; 语言通道生效; modality="v2a" 时
min_block_size 不含语言 token; 无 torch 可跑; 坏输入形状报错。
"""
import numpy as np
import pytest

from embodied_ai.models.vla_config import VLAConfig
from embodied_ai.models.vla_v2a import VLAPolicy, v2a_demo


class TestModalityConfig:
    def test_default_modality_is_vla(self):
        assert VLAConfig().modality == "vla"

    def test_v2a_min_block_size_excludes_lang(self):
        vla = VLAConfig(modality="vla")
        v2a = VLAConfig(modality="v2a")
        assert v2a.lang_tokens == 0
        assert vla.lang_tokens == VLAConfig.LANG_TOKENS
        assert v2a.min_block_size == vla.min_block_size - VLAConfig.LANG_TOKENS

    def test_bad_modality_rejected(self):
        cfg = VLAConfig(modality="audio")   # dataclass 不校验, 由 Policy 校验
        with pytest.raises(ValueError):
            VLAPolicy(cfg)


class TestVLAPolicy:
    def test_v2a_equals_vla_none(self):
        cfg = VLAConfig(modality="v2a")
        p = VLAPolicy(cfg)
        rng = np.random.default_rng(1)
        vision = rng.normal(size=(cfg.num_views, cfg.vision_embd))
        state = rng.normal(size=cfg.state_dim)
        a1 = p.infer_v2a(vision, state)
        a2 = p.infer_vla(vision, None, state)
        assert np.allclose(a1, a2)

    def test_action_shape(self):
        cfg = VLAConfig()
        p = VLAPolicy(cfg)
        vision = np.zeros((cfg.num_views, cfg.vision_embd))
        state = np.zeros(cfg.state_dim)
        a = p.infer_vla(vision, np.zeros(cfg.LANG_TOKENS), state)
        assert a.shape == (cfg.action_horizon, cfg.action_dim)

    def test_language_channel_changes_output(self):
        cfg = VLAConfig()
        p = VLAPolicy(cfg)
        vision = np.zeros((cfg.num_views, cfg.vision_embd))
        state = np.zeros(cfg.state_dim)
        a_none = p.infer_vla(vision, None, state)
        a_lang = p.infer_vla(vision, np.ones(cfg.LANG_TOKENS) * 3.0, state)
        assert not np.allclose(a_none, a_lang)

    def test_deterministic(self):
        cfg = VLAConfig(seed=42)
        p1, p2 = VLAPolicy(cfg), VLAPolicy(cfg)
        vision = np.ones((cfg.num_views, cfg.vision_embd))
        state = np.ones(cfg.state_dim)
        assert np.allclose(p1.infer_v2a(vision, state),
                           p2.infer_v2a(vision, state))

    def test_bad_vision_shape_raises(self):
        cfg = VLAConfig()
        p = VLAPolicy(cfg)
        with pytest.raises(ValueError):
            p.infer_v2a(np.zeros(10), np.zeros(cfg.state_dim))

    def test_bad_state_shape_raises(self):
        cfg = VLAConfig()
        p = VLAPolicy(cfg)
        with pytest.raises(ValueError):
            p.infer_v2a(np.zeros((cfg.num_views, cfg.vision_embd)), np.zeros(3))

    def test_v2a_modality_rejects_language(self):
        """v2a 契约是"纯视觉构成判断依据" — 静默吃掉语言会让契约名存实亡。"""
        cfg = VLAConfig(modality="v2a")
        p = VLAPolicy(cfg)
        vision = np.zeros((cfg.num_views, cfg.vision_embd))
        with pytest.raises(ValueError, match="v2a"):
            p.infer_vla(vision, np.zeros(cfg.LANG_TOKENS), np.zeros(cfg.state_dim))
        # 同一 policy 走 vla 模态则接受语言
        p_vla = VLAPolicy(VLAConfig(modality="vla"))
        p_vla.infer_vla(vision, np.zeros(cfg.LANG_TOKENS), np.zeros(cfg.state_dim))

    def test_demo_assertions(self):
        r = v2a_demo()
        # 这些必须是实测值: 曾经 v2a_demo 直接返回字面量 True, verify 断言的
        # 就是那个字面量 —— 自我认证, 等价性坏了也照样 PASS。
        assert r["v2a_equals_vla_none"] is True
        assert r["v2a_rejects_language"] is True
        assert r["lang_changes_output"] is True
        assert r["min_block_size_v2a"] < r["min_block_size_vla"]

    def test_torch_shape_contract(self):
        """numpy 头与 torch VLAModel 的动作形状必须一致, 防止两套实现漂移。"""
        from embodied_ai.models.vla_v2a import torch_shape_contract

        contract = torch_shape_contract()
        if contract is None:
            pytest.skip("无 torch, 跳过形状契约交叉核验")
        torch_shape, np_shape = contract
        assert torch_shape == np_shape

    def test_no_torch_dependency(self):
        import embodied_ai.models.vla_v2a as m
        # vla_v2a 自身不 import torch（模块顶层验证; torch 仅在
        # torch_shape_contract() 内按需导入）
        assert "torch" not in m.__dict__

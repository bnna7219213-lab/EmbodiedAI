"""N3 OS SSH 双通道 测试 (plan_v6 Sprint N)。

反证: 状态机全转换表 (合法边+非法边); MAINTENANCE/PAUSED/PROVISIONED
拒收运动指令 (维护通道绝不遥控); dry-run 命令形状 (ssh/scp token);
provisioning 序列; 未进维护态 push/pull 报错。
"""
import pytest

from embodied_ai.os_channel import (
    ALLOWED_TRANSITIONS,
    ChannelKind,
    DryRunExecutor,
    ModeController,
    ModeState,
    OsChannelAdmin,
    SshCommandBuilder,
)


class TestModeController:
    def test_all_legal_transitions(self):
        """全合法边: 每条 ALLOWED_TRANSITIONS 边都能走通。"""
        for src, dsts in ALLOWED_TRANSITIONS.items():
            for dst in dsts:
                m = ModeController(initial=src)
                assert m.transition(dst) is True
                assert m.state is dst

    def test_illegal_transitions_raise(self):
        """全非法边: 不在转换表的 (src,dst) 一律 ValueError。"""
        states = list(ModeState)
        for src in states:
            for dst in states:
                if dst in ALLOWED_TRANSITIONS[src]:
                    continue
                m = ModeController(initial=src)
                with pytest.raises(ValueError):
                    m.transition(dst)

    def test_full_workflow(self):
        m = ModeController()
        m.transition(ModeState.PAUSED)
        m.transition(ModeState.MAINTENANCE)
        m.transition(ModeState.PROVISIONED)
        m.transition(ModeState.RUNTIME)
        assert m.history == [ModeState.RUNTIME, ModeState.PAUSED,
                             ModeState.MAINTENANCE, ModeState.PROVISIONED,
                             ModeState.RUNTIME]

    def test_motion_only_in_runtime(self):
        """铁律: 仅 RUNTIME 允许运动 — 维护通道绝不遥控。"""
        for state in ModeState:
            m = ModeController(initial=state)
            assert m.can_send_motion() is (state is ModeState.RUNTIME)

    def test_require_motion_raises_in_maintenance(self):
        m = ModeController(initial=ModeState.MAINTENANCE)
        with pytest.raises(PermissionError):
            m.require_motion_allowed()


class TestSshCommandBuilder:
    def test_ssh_cmd_shape(self):
        b = SshCommandBuilder(user="robot")
        cmd = b.build_ssh_cmd("host.local", ["uname", "-a"])
        assert cmd[0] == "ssh"
        assert "robot@host.local" in cmd
        assert cmd[-2:] == ["--", "uname", "-a"] or "uname" in cmd

    def test_scp_to_remote(self):
        b = SshCommandBuilder()
        cmd = b.build_scp_cmd("/tmp/a.py", "/opt/a.py", "host.local",
                              to_remote=True)
        assert cmd[0] == "scp"
        assert cmd[-2] == "/tmp/a.py"
        assert cmd[-1] == "robot@host.local:/opt/a.py"

    def test_scp_from_remote(self):
        b = SshCommandBuilder()
        cmd = b.build_scp_cmd("./logs", "/var/log/embodied/", "host.local",
                              to_remote=False)
        assert cmd[-2] == "robot@host.local:/var/log/embodied/"
        assert cmd[-1] == "./logs"

    def test_factory_script_content(self):
        s = SshCommandBuilder().factory_program_script("cognition_v2")
        assert "#!/bin/sh" in s
        assert "/opt/embodied/cognition" in s
        assert "cognition_v2" in s
        assert "manifest.txt" in s


class TestOsChannelAdmin:
    def _admin(self):
        return OsChannelAdmin("robot.local", executor=DryRunExecutor())

    def test_dry_run_records_commands(self):
        admin = self._admin()
        admin.enter_maintenance("fix bug")
        recorded = admin.executor.recorded
        assert any(a[0] == "ssh" for a in recorded)

    def test_full_maintenance_workflow(self):
        admin = self._admin()
        admin.enter_maintenance("fix historical bug")
        assert admin.mode.state is ModeState.MAINTENANCE
        admin.pull_debug_logs("/var/log/embodied/")
        admin.push_program("dist/new_feature.py", "/opt/embodied/")
        admin.provision_factory_program("cognition_v2")
        assert admin.mode.state is ModeState.PROVISIONED
        admin.return_to_runtime()
        assert admin.mode.state is ModeState.RUNTIME
        kinds = [a[0] for a in admin.executor.recorded]
        assert "ssh" in kinds and "scp" in kinds

    def test_maintenance_blocks_motion(self):
        admin = self._admin()
        admin.enter_maintenance("repair")
        assert admin.mode.can_send_motion() is False
        with pytest.raises(PermissionError):
            admin.mode.require_motion_allowed()

    def test_push_requires_maintenance(self):
        admin = self._admin()
        with pytest.raises(PermissionError):
            admin.push_program("a.py", "/opt/a.py")
        with pytest.raises(PermissionError):
            admin.pull_debug_logs("/var/log/")

    def test_enter_requires_reason(self):
        admin = self._admin()
        with pytest.raises(ValueError):
            admin.enter_maintenance("")

    def test_channel_semantics(self):
        """RUNTIME 态=协作规划通道; 维护态=SSH 维护通道。"""
        admin = self._admin()
        assert admin.channel is ChannelKind.RUNTIME
        admin.enter_maintenance("fix")
        assert admin.channel is ChannelKind.MAINTENANCE

    def test_status_observable(self):
        admin = self._admin()
        admin.enter_maintenance("fix")
        s = admin.status()
        assert s["mode"] == "maintenance"
        assert s["can_send_motion"] is False
        assert s["channel"] == "maintenance"

    def test_return_from_paused(self):
        admin = self._admin()
        admin.mode.transition(ModeState.PAUSED)
        admin.return_to_runtime()
        assert admin.mode.state is ModeState.RUNTIME

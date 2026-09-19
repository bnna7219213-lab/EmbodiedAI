# 上层 GUI 交互界面 — 具身智能体的"脸"

> 本文档回答核心问题: "如何设计一个人机交互界面, 既能控制机器人, 又能解释 AI 决策, 还能在嵌入式 Linux 上流畅运行?"
>
> 关键词: Qt/QML、Web 界面、3D 可视化、机械臂示教、实时状态监控、远程控制

---

## 1. GUI 的定位与需求

### 1.1 GUI 的双重受众

```
受众 1: 终端用户 (家庭/工厂/服务场景)
  需要: 简单语音/按钮控制, 动画状态显示, 异常报警
  
受众 2: 开发者/调试者
  需要: 实时状态可视化, 传感器数据流, 参数调优, 决策追踪

本计划两阶段策略:
  Phase 0-1: 开发者 Web 界面 (快速迭代, 在 PC/平板浏览器)
  Phase 2+: 用户端 App + 机器人头部 LCD 双模式
```

### 1.2 关键交互场景

```
场景 1: 语音交互
  用户: "给我拿一瓶水"
  机器人: (头部显示屏显示 "正在理解指令...")
         (3D 渲染手臂运动轨迹预演)
         (语言回复 + 执行动画)

场景 2: 手动示教 (拖拽 / 遥操作)
  开发者: 拖拽 UR5/机械臂 3D 模型末端到目标位置
  机器人: (关节实时跟随 → 记录轨迹 → 录为新技能)

场景 3: 决策可解释性
  用户: "你刚才为什么从那个方向过来?"
  机器人: (显示路径规划图 + 高亮 "最短安全路径" + 文字解释)

场景 4: 异常处理
  机器人: "我遇到了障碍物, 请问是否需要重新规划?"
  界面:  显示障碍物 + 弹出选项 [重试] [跳过] [手动控制]
```

---

## 2. Qt for Embedded Linux (推荐方案)

### 2.1 为什么选 Qt

```
嵌入式 GUI 方案对比:

┌──────────────┬───────────────────────────────────────────────────┐
│ 方案          │ 评估                                               │
├──────────────┼───────────────────────────────────────────────────┤
│ ★ Qt Quick   │ 嵌入式领域王者, QML 声明式, GPU 加速                │
│ (QML)        │ 与 ROS2 集成好, 跨平台, 代码少                     │
│              │ ★ 推荐 (NXP, 汽车, 医疗设备主流选择)              │
├──────────────┼───────────────────────────────────────────────────┤
│ Web(Electron)│ 开发快, 但占用大 (200MB+ 内存), 性能低            │
│              │ 适合 Phase 0 开发阶段, 不推荐最终产品               │
├──────────────┼───────────────────────────────────────────────────┤
│ LVGL         │ 极轻量, MCU 级, 功能有限                          │
│              │ 适合关节板上的 OLED 调试屏                          │
├──────────────┼───────────────────────────────────────────────────┤
│ GTK/Flutter  │ GTK 过时; Flutter 新但生态未熟                     │
└──────────────┴───────────────────────────────────────────────────┘

Jetson Orin (ARM + GPU):
  Qt + QML + OpenGL ES 3.0
  → QML 的 Scene3D 组件可直接做 3D 机器人模型
  → 触摸屏 (MIPI DSI 7 英寸)
  → 匹配 Android 级流畅度 (60fps)
```

### 2.2 Qt + ROS2 架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    Qt GUI Application (C++ / QML)                │
│                                                                  │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐    │
│  │ MainWindow.qml  │  │ StatusPanel.qml │  │ TeachPanel.qml  │    │
│  │ (主界面/状态)    │  │ (传感器仪表)    │  │ (示教拖拽)      │    │
│  └────────┬───────┘  └────────┬───────┘  └────────┬───────┘    │
│           │                   │                   │             │
│  ┌────────┴───────────────────┴───────────────────┴────────┐   │
│  │              ROS2Interface (C++ rclcpp wrapper)          │   │
│  │  - 订阅 /robot/state, /tactile/processed, /decision_log  │   │
│  │  - 发布 /gui/command, /gui/teach_waypoint                │   │
│  │  - 调用 /task_manager/srv/start_task                     │   │
│  └──────────────────────────┬────────────────────────────┘   │
└─────────────────────────────┼────────────────────────────────┘
                              │
                          DDS 总线
                              │
┌─────────────────────────────┼────────────────────────────────┐
│                             ▼                                 │
│                     ROS2 Middleware                           │
│                                                              │
│   /task_planner   /motion_planner   /safety_monitor          │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## 3. QML 界面设计

### 3.1 主界面布局

```qml
// MainWindow.qml

import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import QtQuick3D 1.15
import Robot.Ros2 1.0

ApplicationWindow {
    id: mainWindow
    visible: true
    width: 1280
    height: 800
    title: "EmbodiedAI Control"
    
    ROS2Interface { id: ros2 }
    
    // 整体布局: 3D 视图 | 状态面板 | 控制台
    RowLayout {
        anchors.fill: parent
        
        // 左: 3D 机器人模型 + 轨迹可视化
        Rectangle {
            Layout.preferredWidth: parent.width * 0.6
            Layout.fillHeight: true
            color: "#1a1a2e"
            
            View3D {
                id: robotView
                anchors.fill: parent
                
                Environment { id: env }
                DirectionalLight { position: Qt.vector3d(-500, 500, -1000) }
                PerspectiveCamera { position: Qt.vector3d(800, 600, 1200) }
                
                RobotScene {
                    id: robot3D
                    joints: ros2.joint_states
                    ee_pose: ros2.eef_pose
                    
                    PathVisualization {
                        trajectory: ros2.planned_trajectory
                        opacity: 0.5
                    }
                    
                    ObstacleMarkers {
                        obstacles: ros2.detected_obstacles
                    }
                }
            }
            
            // 3D 视图上的状态浮层
            Rectangle {
                anchors.top: parent.top
                anchors.right: parent.right
                anchors.margins: 10
                width: 200
                height: 120
                color: "#33000000"
                radius: 8
                
                Column {
                    anchors.centerIn: parent
                    spacing: 4
                    
                    Text { 
                        color: ros2.safety_level > 0 ? "red" : "lime"
                        text: ros2.mode 
                        font.bold: true 
                    }
                    Text { color: "white"; text: ros2.current_task }
                    Text { color: "white"; text: "FPS: " + ros2.loop_freq.toFixed(1) + "Hz" }
                }
            }
        }
        
        // 中右: 操作面板
        ColumnLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            
            // 指令输入
            TextArea {
                id: commandInput
                placeholderText: "Type command or press mic to speak..."
                Layout.fillWidth: true
                Layout.preferredHeight: 120
                
                background: Rectangle { 
                    border.color: commandInput.focus ? "#4a9eff" : "#555"
                    radius: 6
                }
                
                Keys.onReturnPressed: ros2.sendCommand(commandInput.text)
            }
            
            // 按钮行
            RowLayout {
                Button { 
                    text: "Execute" 
                    highlighted: true
                    onClicked: ros2.sendCommand(commandInput.text)
                }
                Button { text: "Stop (E-stop)"; onClicked: ros2.eStop() }
                Button { text: "Home"; onClicked: ros2.home() }
                Button { text: "Teach Mode"; onClicked: teachDialog.open() }
            }
            
            // 传感器数据表格 (折叠面板)
            SensorDashboard {
                Layout.fillWidth: true
                Layout.fillHeight: true
                tactile: ros2.tactile_state
                joints: ros2.joint_states
                vision: ros2.vision_status
            }
            
            // 决策日志 (下边)
            DecisionLogView {
                Layout.fillWidth: true
                Layout.preferredHeight: 150
                model: ros2.decision_log_model
            }
        }
    }
    
    // 异常弹出框
    ErrorDialog {
        id: errorDialog
        model: ros2.error_list
    }
    
    // 语音对话框
    VoiceDialog {
        id: voiceDialog
        visible: false
        onAccepted: ros2.sendCommand(voiceDialog.text)
    }
}
```

### 3.2 3D 机器人模型 (QML 3D)

```qml
// RobotScene.qml

import QtQuick 2.15
import QtQuick3D 1.15

Node {
    id: root
    
    // 关节状态 (来自 ROS2)
    property var joints: []
    property var ee_pose: null
    
    // 机器人骨骼链: 基于 URDF 或手工建模
    Model {
        id: torso
        source: "meshes/torso.mesh"    // OBJ 或 FBT 格式
        position: Qt.vector3d(0, 0, 0.5)
    }
    
    Model {
        id: left_shoulder
        source: "meshes/shoulder_L.mesh"
        position: torso.position + Qt.vector3d(0, 0.15, 0.25)
        rotation: fromEuler(0, 0, joints[0] * 180/Math.PI)
    }
    
    Model {
        id: left_upper_arm
        source: "meshes/upper_arm.mesh"
        position: left_shoulder.position + Qt.vector3d(0, 0, -0.15)
        rotation: fromEuler(joints[1] * 180/Math.PI, 0, 0)
    }
    
    Model {
        id: left_forearm
        source: "meshes/forearm.mesh"
        position: left_upper_arm.position + Qt.vector3d(0, 0, -0.3)
        rotation: fromEuler(joints[2] * 180/Math.PI, 0, 0)
    }
    
    // ... 继续链条直到指尖
    
    // 末端执行器 (手爪)
    Model {
        id: left_gripper
        source: "meshes/gripper.mesh"
        // 位置由 fk 计算或从 ROS2 /robot/state 读取
        position: root.ee_pose ? root.ee_pose.position : Qt.vector3d(0.4, 0.5, 0.3)
    }
    
    // 规划的轨迹可视化
    PathVisualization {
        trajectory: []  // 由 ROS2 话题更新
    }
    
    // 检测到的障碍物
    ObstacleMarkers {
        obstacles: []
        color: "red"
    }
}
```

### 3.3 示教面板

```qml
// TeachPanel.qml

Item {
    id: teachPanel
    property var teach_points: []
    property bool recording_mode: false
    
    // 1. 拖拽示教模式
    MouseArea {
        id: dragArea
        anchors.fill: parent
        
        onPositionChanged: {
            if (recording_mode && drag.active) {
                var waypoint = screen_to_robot(mouse.x, mouse.y)
                teach_points.push(waypoint)
                ros2.publish_teach_point(waypoint)
                visual.marker_add(waypoint)
            }
        }
        
        onPressed: {
            recording_mode = true
            teach_points = []
            recording_start()
        }
        
        onReleased: {
            recording_mode = false
            recording_stop()
            show_save_dialog()
        }
    }
    
    // 2. 坐标显示
    Text {
        text: "Teach Points: " + teach_points.length + 
              "\nCurrent: X=" + current_x + " Y=" + current_y + " Z=" + current_z
    }
    
    // 3. 保存/回放/删除
    RowLayout {
        Button { text: "Save Skill"; onClicked: save_skill() }
        Button { text: "Preview"; onClicked: preview_skill() }
        Button { text: "Record Voice Note"; onClicked: record_voice_note() }
        Button { text: "Clear"; onClicked: clear_points() }
    }
}
```

---

## 4. 语音交互系统

### 4.1 嵌入式语音方案

```
语音交互方案对比:

┌──────────────┬────────────────────────────────────────────────────┐
│ Vosk +       │ 离线, 小模型, 命令词识别                              │
│ Speech       │ 不需要网络, SentencePiece 分词,                       │
│ Recognition  │ 适合固定命令: "Stop", "Home", "Take cup"             │
│              │ CPU 低 (< 1 核), 识别率 > 95%                         │
│              │ ★ 推荐用于固定命令识别                                │
├──────────────┼────────────────────────────────────────────────────┤
│ Whisper      │ 离线, 大模型 (74M-1.5B 参数)                        │
│ (edge)       │ 通用 ASR, 支持自然语言                                │
│              │ Jetson Orin 可跑 base.en (74M, 10s 音频 = 2s 推理)    │
│              │ GPU 需要 ~ 2GB, 延迟 ~ 200ms                          │
│              │ ★ 推荐用于自然语言指令                                │
├──────────────┼────────────────────────────────────────────────────┤
│ VITS /       │ 语音合成 (AI 说话)                                    │
│ Coqui TTS    │ 离线, 高质量, Jetson 上 50ms 延迟                     │
│              │ ★ 推荐用于机器人语言回复                              │
├──────────────┼────────────────────────────────────────────────────┤
│ WebRTC AEC   │ 回声消除 (机器人自己说话时, 麦克风不收到自己)          │
│              │ 必须, 否则"自己听到自己", 用户体验极差                │
└──────────────┴────────────────────────────────────────────────────┘
```

### 4.2 语音交互管线

```python
# voice_pipeline.py

import vosk
import whisper
import numpy as np
import sounddevice as sd
from TTS.api import TTS
import webrtcvad

class VoicePipeline:
    def __init__(self):
        # 1. 唤醒词 (Porcupine / Sensory)
        self.wake_word_engine = porcupine.create(
            keyword_paths=["hey-robot.ppn"],  # 中文可自定义训练
            sensitivities=[0.7]
        )
        
        # 2. ASR (Whisper base.en)
        self.whisper_model = whisper.load_model("base.en")  # 74M params
        
        # 3. TTS (VITS)
        self.tts = TTS(model_name="vits/ljspeech", gpu=True)
        
        # 4. VAD (WebRTC)
        self.vad = webrtcvad.Vad(3)  # 0-3, 3 最激进
        
        # 5. 音频配置
        self.FS = 16000  # Hz
        self.CHUNK = 512
        
    def listen_and_transcribe(self):
        """录音并识别, 等待 VAD 检测到语音结束"""
        audio_buffer = []
        silence_count = 0
        
        with sd.InputStream(samplerate=self.FS, channels=1, 
                           blocksize=self.CHUNK, dtype='int16') as stream:
            while True:
                chunk, _ = stream.read(self.CHUNK)
                pcm_data = chunk.tobytes()
                
                is_speech = self.vad.is_speech(pcm_data, self.FS)
                audio_buffer.append(chunk)
                
                if is_speech:
                    silence_count = 0
                else:
                    silence_count += 1
                
                # 1.5s 静音 = 结束
                if silence_count > 50 and len(audio_buffer) > 20:
                    break
        
        audio = np.concatenate(audio_buffer)
        
        # Whisper 识别
        result = self.whisper_model.transcribe(audio, language="en")
        return result["text"].strip()
    
    def speak(self, text):
        """TTS 合成并播放到机器人扬声器"""
        # 合成
        wav = self.tts.tts(text)
        
        # 播放
        sd.play(np.array(wav), self.FS)
        sd.wait()
    
    def run(self):
        """主交互循环"""
        print("Voice pipeline ready. Say 'Hey robot' to start...")
        
        while True:
            # 录音 pcm
            with sd.InputStream(samplerate=self.FS, channels=1, 
                               blocksize=self.CHUNK) as stream:
                pcm_chunk, _ = stream.read(self.FS * 3)  # 3s max
            
            # 唤醒词检测
            if self.wake_word_engine.process(pcm_chunk) >= 0:
                self.speak("Yes? I'm listening.")
                
                # 语音识别
                command = self.listen_and_transcribe()
                print(f"User said: '{command}'")
                
                # 发送到 ROS2
                # ros2.publish('/voice/command', command)
                
                # 语言回复
                response = self.process_command(command)
                self.speak(response)
```

---

## 5. Web 远程界面

### 5.1 WebSocket + ROS2 Bridge

```
远程访问 (机房/云端监控小车/家庭部署):

  Browser (PC / Pad / Phone)
      │
      ▼ WebSocket (wss://robot.local:9090)
      │
  ros2-web-bridge (ROS2 节点)
      │
      ▼
  ROS2 Topics / Services
      │
  机器人本体

技术栈:
  - ros2-web-bridge: rosbridge_suite (WebSocket ↔ ROS2)
  - 前端: React + Three.js (3D 可视化)
  - 安全: wss + 用户认证 + VPN (可选)
```

### 5.2 微服务架构

```
完整通信架构:

        ┌────────────────────────────────────────────────┐
        │                远程控制端                       │
        │   Web App (React)  /  App (Flutter)            │
        │   协议: WebSocket + ROS2 rosbridge             │
        └───────────────────────┬────────────────────────┘
                                │
                           局域网 / VPN
                                │
        ┌───────────────────────┼────────────────────────┐
        │                  机器人本体                      │
        │                       │                        │
        │  ┌────────────────────┼────────────────────┐  │
        │  │ ros2-web-bridge    │   Qt GUI (本机)   │  │
        │  │ (远程连接)         │   (本地 LCD)      │  │
        │  └────────┬───────────┘                   │  │
        │           │                               │  │
        │       DDS 总线 (ROS2)                      │  │
        │           │                               │  │
        │  /task_planner /motion_planner /sensors   │  │
        │                                           │  │
        │  ┌────────────────────────────────────┐  │  │
        │  │ Jetson Orin (CPU + GPU + NPU)     │  │  │
        │  └────────────────────────────────────┘  │  │
        └───────────────────────────────────────────┘
```

---

## 6. 头部 LCD 显示 (嵌入式面板)

### 6.1 面部动画显示

```
7 英寸 LCD (MIPI DSI 1024×600):
  
  用途: 
    - 机器人"眼睛": 表情动画 (开心/疑惑/工作中)
    - 状态文字: "正在抓取杯中..."
    - 决策解释: 显示注意力热力图
    - 语音波形: 实时波形动画
  
  实现:
    - Qt for MCU / DRM (无 X11/Wayland)
    - 或直接 DRM/KMS (framebuffer) + 自绘
    - LVGL (极轻量, 适合无 GPU 场景)
```

---

## 7. 综合 GUI 开发路线图

```
Phase 0 (调试):
    Web 界面 (ros2-web-bridge + React)
    - 状态监控面板 (关节、传感器)
    - CAN 总线调试
    - 命令行终端

Phase 1 (单臂):
    Qt QML 本地界面
    - 3D 模型 + 轨迹
    - 拖拽示教
    - 决策日志显示
    - 头部 LVGL 表情动画

Phase 2 (双臂 + 移动):
    语音交互 (Vosk + Whisper)
    - 语音命令控制
    - AI 语言回复 (TTS)
    - 远程 Web 控制 (平板)
    
Phase 3 (完整):
    用户端 App (Flutter + WebSocket)
    - 家庭用户远程控制
    - 健康状态报告
    - OTA 固件更新 (ab 分区升级)
    - 多机协同 (多台机器人, 一个平板控制)
```

---

## 附录 A: Qt for Embedded Linux 在 Jetson Orin 上的构建

```bash
# 交叉编译 Qt 6.5 for Jetson Orin (aarch64)

# 1. 安装交叉编译链 (已有 Jetson SDK Manager 提供)
export CROSS_COMPILE=/opt/toolchain/bin/aarch64-linux-gnu-

# 2. Qt 源码 + 配置
git clone https://code.qt.io/qt/qt5.git -b 6.5
cd qt5
./init-repository

# 3. 配置 (关键: eglfs 平台插件)
./configure \
    -release \
    -opengl es2 \
    -device linux-jetson-nano-g++ \
    -device-option CROSS_COMPILE=$CROSS_COMPILE \
    -sysroot /opt/sysroot-jetson \
    -prefix /usr/local/qt6-jetson \
    -qt-libpng -qt-libjpeg -qt-zlib \
    -qt-pcre -qt-harfbuzz \
    -skip qt3d -skip qtwebengine \
    -nomake examples -nomake tests

# 4. 编译 + 部署
make -j$(nproc)
make install
rsync /usr/local/qt6-jetson jetson@robot:/usr/local/qt6-jetson

# 5. 运行参数 (eglfs 独占显示, 无需 X11)
export QT_QPA_PLATFORM=eglfs
export QT_QPA_EGLFS_INTEGRATION=eglfs_kms
export QT_QPA_EGLFS_KMS_CONFIG=/etc/kms.conf

# /etc/kms.conf 内容:
# {
#   "device": "/dev/dri/card0",
#   "outputs": [
#     { "name": "DSI-1", "mode": "1024x600" }
#   ]
# }

./robot_gui --platform eglfs
```

# 嵌入式 Linux on ARM — 具身智能体的软件基石

> 本文档覆盖从 ARM 裸机汇编到完整 Linux 运行时的全栈技术，
> 以及如何通过内核驱动、UBOOT、根文件系统来驱动物理硬件（电机、传感器、显示屏）。

---

## 第 1 部分：ARM 汇编与裸机编程

### 1.1 ARM Cortex-A 与 Cortex-M 在人形机器人上的分工

```
机器人芯片分工:

Cortex-A 系列 (应用处理器) ──────────── 运行 Linux + AI 推理
  ┌──────────────────────────────────────────────┐
  │ Jetson Orin / RK3588 / i.MX8                  │
  │ - Cortex-A78 / A76 核心                       │
  │ - 运行 Linux Kernel + ROS2 + VLA 模型          │
  │ - 频率: 1.5~2.4 GHz                           │
  │ - 职责: AI 决策 / 视觉处理 / 规划 / GUI         │
  └──────────────────────────────────────────────┘

Cortex-M 系列 (微控制器) ────────────── 运行裸机/RTOS + 电机控制
  ┌──────────────────────────────────────────────┐
  │ STM32G4 / STM32H7 (Cortex-M4/M7)             │
  │ - 单核或双核                                  │
  │ - 运行裸机 C / FreeRTOS                       │
  │ - 频率: 170~480 MHz                           │
  │ - 职责: FOC 电机控制 / 传感器采集 / 安全保护    │
  └──────────────────────────────────────────────┘

关键区分:
  Cortex-A 跑 Linux → 高级决策 (慢, 周期 50ms)
  Cortex-M 跑裸机 → 低级控制 (快, 周期 100μs)
  两者通过 CAN / SPI / UART 通信
```

### 1.2 ARM 汇编基础（与机器人驱动直接相关的指令）

```asm
; ============================================
; ARM Cortex-M4 (STM32G4) 裸机驱动示例
; ============================================

; --- GPIO 控制: 点亮 LED (调试指示) ---
; 假设 LED 接在 GPIOA Pin5 (类似 Nucleo 板上的用户 LED)

  .equ RCC_BASE,      0x40021000
  .equ RCC_AHB2ENR,   (RCC_BASE + 0x4C)   ; AHB2 外设时钟使能
  .equ GPIOA_BASE,    0x48000000
  .equ GPIOA_MODER,   (GPIOA_BASE + 0x00) ; 模式寄存器
  .equ GPIOA_ODR,     (GPIOA_BASE + 0x14) ; 输出数据寄存器
  .equ GPIOA_BSRR,    (GPIOA_BASE + 0x18) ; 位设置/复位寄存器

  .global Reset_Handler
Reset_Handler:
  ; 1. 使能 GPIOA 时钟 (RCC_AHB2ENR 的 bit0)
  LDR  R0, =RCC_AHB2ENR
  LDR  R1, [R0]
  ORR  R1, R1, #0x01        ; bit0 = 1
  STR  R1, [R0]

  ; 2. 设置 PA5 为输出模式 (MODER5 = 01)
  LDR  R0, =GPIOA_MODER
  LDR  R1, [R0]
  BIC  R1, R1, #(0x03 << 10) ; 清除 bit10-11
  ORR  R1, R1, #(0x01 << 10) ; 设为输出
  STR  R1, [R0]

  ; 3. 循环闪烁
loop:
  LDR  R0, =GPIOA_BSRR
  MOV  R1, #(1 << 5)        ; SET bit5 (点亮)
  STR  R1, [R0]

  LDR  R2, =500000          ; 延时计数
delay1:
  SUBS R2, R2, #1
  BNE  delay1

  MOV  R1, #(1 << 21)       ; RESET bit5 = bit21 (低16位写SET, 高16位写RESET)
  STR  R1, [R0]

  LDR  R2, =500000
delay2:
  SUBS R2, R2, #1
  BNE  delay2

  B    loop
```

### 1.3 PWM 生成（电机控制的最底层）

```asm
; ============================================
; STM32G4 TIM1 生成三相互补 PWM (用于电机驱动)
; ============================================

  .equ TIM1_BASE,     0x40010000
  .equ TIM1_CR1,      (TIM1_BASE + 0x00)
  .equ TIM1_CCMR1,    (TIM1_BASE + 0x18)  ; 捕获/比较模式
  .equ TIM1_CCER,     (TIM1_BASE + 0x20)  ; 捕获/比较使能
  .equ TIM1_CCR1,     (TIM1_BASE + 0x34)  ; 比较寄存器 1 (占空比)
  .equ TIM1_ARR,      (TIM1_BASE + 0x2C)  ; 自动重装载 (频率)
  .equ TIM1_BDTR,     (TIM1_BASE + 0x44)  ; 死区时间 (防止上下臂直通)

init_pwm:
  ; ARR = 1680 → 168MHz / 1680 = 100kHz PWM (实际用 20kHz 需 ARR=8400)
  LDR  R0, =TIM1_ARR
  MOV  R1, #8400
  STR  R1, [R0]

  ; 初始占空比 50% = 4200
  LDR  R0, =TIM1_CCR1
  MOV  R1, #4200
  STR  R1, [R0]

  ; 死区时间 = 100ns → DTG = 0x8F (156ns @ 168MHz)
  LDR  R0, =TIM1_BDTR
  MOV  R1, #0x8F
  STR  R1, [R0]

  ; 使能 PWM 输出 + 主输出使能 (MOE)
  LDR  R0, =TIM1_CCER
  MOV  R1, #0x05            ; CC1E + CC1NE (互补输出)
  STR  R1, [R0]

  LDR  R0, =TIM1_CR1
  MOV  R1, #0x81            ; CEN = 1, ARPE = 1
  STR  R1, [R0]

  BX   LR

; --- 动态更新占空比 (在 ADC 中断中调用 FOC 算法后) ---
  .global update_pwm_duty
update_pwm_duty:
  ; R0 = 占空比值 (0-arr)
  LDR  R1, =TIM1_CCR1
  STR  R0, [R1]
  BX   LR
```

### 1.4 ADC 采样（电流/力矩反馈的最底层）

```asm
; ============================================
; STM32G4 ADC1 触发采样 (与 PWM 同步)
; ============================================

  .equ ADC1_BASE,     0x50000000
  .equ ADC1_CR,       (ADC1_BASE + 0x08)
  .equ ADC1_DR,       (ADC1_BASE + 0x40)
  .equ ADC1_ISR,      (ADC1_BASE + 0x00)

  .global ADC_IRQHandler
ADC_IRQHandler:
  ; 读取转换结果 (12-bit, 0-4095 → 0-3.3V)
  LDR  R0, =ADC1_DR
  LDR  R1, [R0]             ; R1 = 电流采样值 (已对齐)

  ; 清除 EOC 标志
  LDR  R0, =ADC1_ISR
  MOV  R2, #0x02
  STR  R2, [R0]

  ; 调用 FOC 算法 (C 函数)
  BL   foc_calculate        ; 传入 R1 (电流值), 返回新占空比

  ; 更新 PWM
  BL   update_pwm_duty

  ; 清除中断挂起
  BX   LR
```

---

## 第 2 部分：硬件外设驱动 (UART, I2C, SPI, LCD)

### 2.1 UART — 关节板与主控的调试/通信通道

```
UART 在机器人中的用途:
  ┌─────────────────────────────────────────────────────────┐
  │ 1. 调试输出: printk / printf 到串口终端                  │
  │ 2. GPS/定位: UART 连接 RTK GPS 模块                      │
  │ 3. 无线模块: ESP8266/蓝牙 HC-05 通过 UART               │
  │ 4. 激光雷达: Slamtec RPLIDAR 通过 UART                   │
  │ 5. 固件更新: UART bootloader (DFY 模式)                  │
  └─────────────────────────────────────────────────────────┘

STM32G4 UART1 配置 (115200 8N1):
  TX → PA9 (复用功能 AF7)
  RX → PA10 (复用功能 AF7)
  波特率: 115200 (APB2 = 72MHz → BRR = 72M/115200 = 625 = 0x271)
```

```c
// ===== 嵌入式 C 驱动 (STM32 HAL 之前的原生寄存器方式) =====

#define USART1_BASE   0x40013800
#define USART1_BRR    (*(volatile uint32_t *)(USART1_BASE + 0x0C))
#define USART1_CR1    (*(volatile uint32_t *)(USART1_BASE + 0x00))
#define USART1_ISR    (*(volatile uint32_t *)(USART1_BASE + 0x1C))
#define USART1_TDR    (*(volatile uint32_t *)(USART1_BASE + 0x28))
#define USART1_RDR    (*(volatile uint32_t *)(USART1_BASE + 0x24))

void uart_init(uint32_t baud) {
    // 1. 使能 USART1 时钟 (RCC_APB2ENR)
    *(volatile uint32_t *)0x40021060 |= (1 << 14);
    
    // 2. 配置 GPIOA 9/10 为复用功能
    *(volatile uint32_t *)0x48000000 |= (0x02 << 18);  // PA9 AF
    *(volatile uint32_t *)0x48000000 |= (0x02 << 20);  // PA10 AF
    
    // 3. 波特率
    uint32_t fclk = 72000000;  // APB2 时钟
    USART1_BRR = fclk / baud;
    
    // 4. 使能 TX + RX + USART (TE | RE | UE)
    USART1_CR1 = (1 << 3) | (1 << 2) | (1 << 0);
}

void uart_send(char c) {
    while (!(USART1_ISR & (1 << 7)));  // 等待 TXE
    USART1_TDR = c;
}

void uart_print(const char *str) {
    while (*str) {
        if (*str == '\n') uart_send('\r');
        uart_send(*str++);
    }
}

// Linux 侧通过 /dev/ttyUSB0 读取
// 设备树描述 (device tree):
// &usart1 {
//     pinctrl-0 = <&usart1_tx_pa9 &usart1_rx_pa10>;
//     status = "okay";
// };
```

### 2.2 I2C — 传感器总线（IMU, 触觉, 温度, EEPROM）

```
I2C 在机器人中的连接:
  ┌──────── I2C 总线 1 (APB, 400kHz) ────────┐
  │                                            │
  │  ICM-42688 (IMU)       地址: 0x68          │
  │  ADS1256  (力传感器 ADC) 地址: 0x01         │
  │  AT24C256 (EEPROM 参数) 地址: 0x50         │
  │  NTC 温度芯片           地址: 0x48         │
  │  触觉传感器阵列         地址: 0x20-0x27     │
  └───────────────────────────────────────────┘
```

```c
// ===== I2C 原生寄存器驱动 (STM32 I2C1) =====

#define I2C1_BASE    0x40005400
#define I2C1_CR1     (*(volatile uint32_t *)(I2C1_BASE + 0x00))
#define I2C1_CR2     (*(volatile uint32_t *)(I2C1_BASE + 0x04))
#define I2C1_TXDR    (*(volatile uint32_t *)(I2C1_BASE + 0x28))
#define I2C1_RXDR    (*(volatile uint32_t *)(I2C1_BASE + 0x24))
#define I2C1_ISR     (*(volatile uint32_t *)(I2C1_BASE + 0x1C))
#define I2C1_TIMINGR (*(volatile uint32_t *)(I2C1_BASE + 0x10))

void i2c_init_400khz(void) {
    // 时序寄存器: 400kHz @ 72MHz APB1
    // PRESC=1, SCLDEL=4, SDADEL=2, SCLH=0x0F, SCLL=0x13
    I2C1_TIMINGR = (1 << 28) | (4 << 20) | (2 << 16) | (0x0F << 8) | 0x13;
    I2C1_CR1 |= 0x01;  // PE=1 使能
}

// 读取 IMU ICM-42688 的 WHO_AMIC 寄存器 (0x75, 应返回 0x47)
uint8_t i2c_read_reg(uint8_t dev_addr, uint8_t reg_addr) {
    // TXE 等待
    while (!(I2C1_ISR & 0x01));
    
    // 配置: 写模式, 1 字节, 无 STOP
    I2C1_CR2 = (dev_addr << 1) | (1 << 16) | (0 << 10);  // RD_WRN=0
    I2C1_TXDR = reg_addr;
    
    // 等待 TC
    while (!(I2C1_ISR & 0x40));
    
    // 重新配置: 读模式, 1 字节, STOP
    I2C1_CR2 = (dev_addr << 1) | (1 << 16) | (1 << 10) | (1 << 13) | (1 << 14);
    
    // 等待 RXNE
    while (!(I2C1_ISR & 0x04));
    
    return I2C1_RXDR;
}

// 对应 Linux I2C 驱动 (device tree + kernel driver):
//
// kernel/drivers/i2c/sensors/icm42688.c:
// static const struct of_device_id icm42688_match[] = {
//     { .compatible = "invense,icm42688" },
//     { }
// };
// 
// imu@68 {
//     compatible = "invense,icm42688";
//     reg = <0x68>;
//     interrupt-parent = <&gpioa>;
//     interrupts = <0 IRQ_TYPE_EDGE_FALLING>;  // Data Ready
// };
```

### 2.3 SPI — 高速器件（绝对编码器、NOR Flash、LCD）

```
SPI 在机器人中的应用:
  ┌────────── SPI1 (60MHz) ──────────┐
  │                                    │
  │  TLE5012B (磁编码器) 片选 PA4       │ → 关节角度 14-bit
  │  W25Q128 (Nor Flash)  片选 PA15    │ → 存储校准参数
  │  ADS1256 (24bit ADC)  片选 PB0     │ → 力传感器
  └────────────────────────────────────┘
```

```c
// ===== SPI 驱动: 读取 TLE5012B 编码器 =====

#define SPI1_BASE    0x40013000
#define SPI1_CR1     (*(volatile uint32_t *)(SPI1_BASE + 0x00))
#define SPI1_SR      (*(volatile uint32_t *)(SPI1_BASE + 0x08))
#define SPI1_DR      (*(volatile uint32_t *)(SPI1_BASE + 0x0C))

uint16_t spi_read_encoder(void) {
    // 拉低片选 (PA4)
    *(volatile uint32_t *)0x48000018 = (1 << 20);  // RESET
    
    // 发送读命令 0x8000 (读角度寄存器)
    while (!(SPI1_SR & 0x02));   // TXE
    SPI1_DR = 0x8021;            // 读命令 + 安全字
    while (!(SPI1_SR & 0x01));   // RXNE
    uint16_t angle = SPI1_DR;     // 14-bit 角度值
    
    // 拉高片选
    *(volatile uint32_t *)0x48000018 = (1 << 4);   // SET
    
    return angle & 0x7FFF;        // 14-bit, 0-16383 对应 0-360°
}

// Linux SPI 子系统的使用:
// struct spi_device *spi = to_spi_device(dev);
// struct spi_transfer xfer = {
//     .tx_buf = tx,
//     .rx_buf = rx,
//     .len = 2,
//     .speed_hz = 1000000,
// };
// spi_sync(spi, &xfer);
```

### 2.4 LCD — 头部/胸部的交互显示屏

```
LCD 接口: MIPI DSI (Jetson Orin 原生支持)
  - 分辨率: 1920×1080 (头部 7 英寸)
  - 刷新率: 60 Hz
  - 触摸: I2C 接口的 Goodix GT911
  
帧缓冲显示流程:
  GPU 渲染 → DRM 驱动 → MIPI DSI 控制器 → 物理显示

如果不使用 GPU, 直接用 framebuffer:
  /dev/fb0 (Linux frame buffer)
  写入: ioctl FBIOPUT_VSCREENINFO
  直接操作: mmap /dev/fb0 → 写像素
```

---

## 第 3 部分：UBOOT — 启动引导

### 3.1 UBOOT 启动流程

```
上电后执行顺序 (RK3588 为例):
                                                             
  ┌─────────────────────────────────────────────────────────┐
  │ Step 1: BootROM (芯片内部 ROM)                           │
  │   → 初始化 DDR, 读取 SPI Flash 的 UBOOT                  │
  │   → 耗时: ~100ms                                        │
  ├─────────────────────────────────────────────────────────┤
  │ Step 2: U-Boot SPL (Secondary Program Loader)            │
  │   → 更复杂的 DDR 初始化, 加载完整 U-Boot                  │
  │   → 输出 "U-Boot SPL 2024.01"                           │
  ├─────────────────────────────────────────────────────────┤
  │ Step 3: U-Boot Proper                                   │
  │   → 初始化外设, 显示 logo, 倒计时                        │
  │   → 执行 bootcmd 加载 kernel                             │
  │   → 可交互: 按任意键进入 U-Boot 命令行                    │
  ├─────────────────────────────────────────────────────────┤
  │ Step 4: Linux Kernel                                    │
  │   → 解压内核, 初始化设备树, 驱动                          │
  │   → 输出内核 log (dmesg)                                │
  │   → 挂载根文件系统, 启动 init (systemd)                   │
  ├─────────────────────────────────────────────────────────┤
  │ Step 5: Userspace                                      │
  │   → systemd 启动 ROS2 节点                                │
  │   → 启动 VLA 模型                                        │
  │   → 启动控制环路                                         │
  │   → 启动 GUI 显示界面                                    │
  └─────────────────────────────────────────────────────────┘
  
  总启动时间目标: < 15 秒 (裸机 MCU: < 1秒)
```

### 3.2 UBOOT 环境变量（机器人关键配置）

```bash
# U-Boot 命令行 (串口调试时进入)
# 查看环境变量
printenv

# 关键变量:
bootcmd=load mmc 1:1 ${kernel_addr_r} /boot/Image; load mmc 1:1 ${fdt_addr_r} /boot/dtbs/rk3588-robot.dtb; booti ${kernel_addr_r} - ${fdt_addr_r}

bootargs=root=/dev/mmcblk1p2 rootwait console=ttyS2,115200n8 earlycon mmc=1:1:0x100000 gpio=15:0 can0_bitrate=1000000 rt_prio=98 isolcpus=4,5,6,7

kernel_addr_r=0x02080000      # 内核加载地址
fdt_addr_r=0x0a100000        # 设备树加载地址
ramdisk_addr_r=0x0a200000    # initrd 地址

# 机器人专用: 配置实时核心
# isolcpus=4,5,6,7 隔离 CPU 4-7 专门用于控制
# rt_prio=98 实时优先级
# can0_bitrate=1000000 CAN 1Mbps

# 启动失败时恢复:
setenv bootcmd "load usb 0:1 ${kernel_addr_r} /boot/Image; booti ${kernel_addr_r} - ${fdt_addr_r}"
saveenv
reset
```

### 3.3 UBOOT 设备树 (Device Tree) — 硬件描述的圣经

```
设备树定义了"板上有什么硬件"和"它在哪里"

rk3588-robot.dts (关键部分):

/dts-v1/;

/ {
    model = "EmbodiedAI Robot Main Board";
    compatible = "embodied,robot-main", "rockchip,rk3588";
    
    /* 定义 8 个 CPU 核心 */
    cpu@0 { cpu-frequency = <1800000>; };  // 小核
    cpu@4 { cpu-frequency = <2400000>; };  // 大核 (AI 专用)
    
    /* 内存 */
    memory@0 {
        device_type = "memory";
        reg = <0x0 0x80000000>;  // 8GB LPDDR5
    };
    
    /* GPIO 定义: 电源使能、故障输入 */
    gpio_alerts {
        compatible = "gpio-leds";
        motor_fault {
            gpios = <&gpio0 RK_PA0 GPIO_ACTIVE_LOW>;
            label = "motor-fault";
        };
        ai_active {
            gpios = <&gpio0 RK_PA1 GPIO_ACTIVE_HIGH>;
            label = "ai-active";
        };
    };
};

/* CAN 控制器连接到关节板 */
&can0 {
    pinctrl-0 = <&can0m0_pins>;
    status = "okay";
    clock-frequency = <1000000>;  // 1Mbps
};

/* SPI 连接到编码器 */
&spi0 {
    status = "okay";
    cs-gpios = <&gpio1 RK_PB0 GPIO_ACTIVE_LOW>;
    
    encoder@0 {
        compatible = "ti,tle5012b";
        reg = <0>;
        spi-max-frequency = <10000000>;  // 10MHz
    };
};

/* I2C 连接到 IMU */
&i2c1 {
    status = "okay";
    clock-frequency = <400000>;  // 400kHz
    
    imu@68 {
        compatible = "invense,icm42688";
        reg = <0x68>;
        interrupt-parent = <&gpio2>;
        interrupts = <12 IRQ_TYPE_EDGE_FALLING>;
    };
    
    /* 触觉传感器 */
    touch_left@20 {
        compatible = "azoteq,iqs550";
        reg = <0x20>;
        interrupt-parent = <&gpio2>;
        interrupts = <15 IRQ_TYPE_EDGE_FALLING>;
    };
};

/* LCD 接口 */
&dsi {
    status = "okay";
    
    panel: panel@0 {
        compatible = "ilitek,ili9881d";
        backlight = <&backlight>;
        reset-gpios = <&gpio3 RK_PB6 GPIO_ACTIVE_LOW>;
        
        port {
            panel_in: endpoint {
                remote-endpoint = <&dsi_out>;
            };
        };
    };
};
```

---

## 第 4 部分：根文件系统 (Root Filesystem)

### 4.1 构建方式选择

```
┌─────────────────────────────────────────────────────────────────┐
│                       根文件系统构建方式                          │
├───────────────────┬─────────────────────────────────────────────┤
│ Buildroot         │ 简单, 体积小 (~100MB), 适合资源受限          │
│                   │ make menuconfig → 勾选需要的包 → make        │
│                   │ 优点: 构建快, 镜像小, 适合量产                │
│                   │ 缺点: 包管理弱, 适合固定功能                  │
├───────────────────┼─────────────────────────────────────────────┤
│ Yocto             │ 强大, 灵活, 学习曲线陡                       │
│                   │ bitbake linux-yocto + image-robot             │
│                   │ 优点: 包管理好, 社区大, 适合产品线            │
│                   │ 缺点: 构建慢 (首次数小时), 体积大             │
├───────────────────┼─────────────────────────────────────────────┤
│ Debian/Ubuntu     │ 生态最好, 直接用                            │
│                   │ debootstrap --arch=arm64                     │
│                   │ 优点: 包最多, 开发便利                       │
│                   │ 缺点: 体积大, 实时性差, 启动慢               │
└───────────────────┴─────────────────────────────────────────────┘

推荐: Buildroot (Phase 0-2, 专注功能) → Yocto (Phase 3+, 需要软件生态)
```

### 4.2 Buildroot 机器人专用配置

```bash
# buildroot/configs/robot_defconfig

# 架构
BR2_aarch64=y
BR2_cortex_a76=y  # RK3588 大核

# 工具链
BR2_TOOLCHAIN_EXTERNAL=y
BR2_TOOLCHAIN_EXTERNAL_ARM_AARCH64=y

# 内核 (自定义实时内核)
BR2_LINUX_KERNEL=y
BR2_LINUX_KERNEL_CUSTOM_GIT=y
BR2_LINUX_KERNEL_CUSTOM_REPO_URL="https://github.com/embodied-robot/linux-rt"
BR2_LINUX_KERNEL_CUSTOM_REPO_VERSION="6.1.60-rt15"
BR2_LINUX_KERNEL_PREEMPT_RT=y   # 实时抢占补丁
BR2_LINUX_KERNEL_USE_CUSTOM_CONFIG=y
BR2_LINUX_KERNEL_CUSTOM_CONFIG_FILE="board/robot/linux.config"

# 文件系统
BR2_TARGET_ROOTFS_EXT2=y
BR2_TARGET_ROOTFS_EXT2_4=y
BR2_TARGET_ROOTFS_EXT2_SIZE="512M"

# 关键软件包 (最小必要包)
BR2_PACKAGE_BUSYBOX_SHOW_OTHERS=y
BR2_PACKAGE_BUSYBOX_CONFIG="board/robot/busybox.config"

# 网络
BR2_PACKAGE_DROPBEAR=y           # SSH 服务
BR2_PACKAGE_IPROUTE2=y
BR2_PACKAGE_CAN_UTILS=y          # CAN 总线工具

# Python + ROS2
BR2_PACKAGE_PYTHON3=y
BR2_PACKAGE_PYTHON3_NUMPY=y
BR2_PACKAGE_PYTHON3_PYQT5=y       # GUI

# 实时性能
BR2_PACKAGE_LINUX_TOOLS_PERF=y
BR2_PACKAGE_LINUX_TOOLS_STRESS_NG=y

#  robot 自定义包
BR2_PACKAGE_ROBOT_MOTOR_DRV=y    # 电机控制驱动
BR2_PACKAGE_ROBOT_VLA_RT=y       # VLA 推理引擎
BR2_PACKAGE_ROBOT_GUI=y          # 交互界面
```

### 4.3 根文件系统目录结构

```
/                       # 根
├── boot/               # 内核镜像 + 设备树
│   ├── Image           # Linux 内核 (压缩 zImage)
│   ├── initrd.img      # 初始 RAM 盘
│   └── dtbs/           # 设备树 blob
│       ├── rk3588-robot.dtb     # 主控板
│       └── stm32g4-joint.dtb    # 关节板 (如果使用 RTOS)
│
├── dev/                # 设备文件 (kernel 自动管理)
│   ├── can0            # CAN 总线
│   ├── i2c-1           # I2C 总线
│   ├── spi0.0          # SPI 设备
│   ├── ttyS0           # 串口 (调试)
│   ├── gpiochip0       # GPIO
│   ├── dma_heap        # DMA 内存
│   ├── nvidia-gpu      # GPU 设备
│   └── video0          # 摄像头
│
├── sys/                # 系统信息 (内核接口)
│   ├── class/gpio/     # GPIO 控制
│   ├── class/pwm/      # PWM 控制
│   ├── class/i2c-dev/  # I2C 用户空间
│   ├── class/spi-dev/  # SPI 用户空间
│   ├── class/input/    # 输入设备 (触摸屏)
│   └── class/video4linux/ # 视频设备
│
├── etc/                # 配置文件
│   ├── fstab           # 挂载表
│   ├── inittab         # 启动脚本 (SysV init)
│   ├── network/        # 网络配置
│   ├── robot/          # 机器人专用配置
│   │   ├── joints.yaml     # 关节参数表
│   │   ├── vla_model.path  # 模型路径
│   │   ├── safety_limits.yaml # 安全限制
│   │   └── tf_transforms.yaml # 坐标变换
│   ├── systemd/        # systemd 服务
│   │   ├── system/
│   │   │   ├── robot-control.service   # 主控服务
│   │   │   ├── vla-inference.service   # AI 推理服务
│   │   │   ├── gui-interface.service   # GUI 界面
│   │   │   ├── joint-comm.service      # 关节通信
│   │   │   └── safety-monitor.service  # 安全监控
│   │   └── multi-user.target.wants/     # 自启动
│   └── profile.d/      # 环境变量
│
├── lib/                # 共享库
│   ├── modules/        # 内核模块
│   │   └── 6.1.60-rt15/
│   │       ├── kernel/drivers/robot/   # 自定义驱动
│   │       ├── extra/                  # 第三方模块
│   │       └── can/                    # CAN 驱动
│   ├── firmware/       # 固件 (WiFi, GPU 等)
│   └── ld-linux-aarch64.so.1  # 动态链接器
│
├── usr/
│   ├── bin/            # 用户程序
│   │   ├── robot-control       # 主控制程序
│   │   ├── vla-infer           # VLA 推理
│   │   ├── tactile-process     # 触觉处理
│   │   └── safety-daemon       # 安全守护
│   ├── lib/            # 库
│   │   ├── librockchip_mpp.so  # 硬件编解码
│   │   ├── libtorch.so         # PyTorch 推理
│   │   └── libtensorrt.so      # TensorRT
│   ├── share/          # 数据文件
│   │   ├── robot-meshes/       # 3D 模型文件
│   │   ├── rl-policies/        # RL 策略文件
│   │   └── gui-assets/         # GUI 资源
│   └── local/
│       └── models/     # AI 模型
│           ├── vla-7b-int4.engine   # TensorRT 引擎
│           ├── tactile-cnn.onnx     # 触觉分类模型
│           └── diffusion-policy/    # 扩散策略
│
├── opt/
│   ├── ros2/           # ROS2 安装
│   │   ├── bin/
│   │   ├── lib/
│   │   └── share/
│   └── nnfu/           # 神经网络加速固件
│       └── rk3588_npu_fw.bin
│
├── run/                # 运行时状态 (tmpfs, 重启清空)
│   ├── robot.state     # 机器人当前状态
│   ├── vla.cache       # 推理 KV cache
│   └── logs/
│
├── tmp/                # 临时文件 (tmpfs)
│
├── var/                # 可变数据
│   ├── log/            # 系统日志
│   │   ├── messages    # syslog
│   │   ├── robot/      # 机器人运行日志
│   │   └── vla/        # AI 推理日志
│   └── lib/            # 变量库数据
│
├── home/
│   └── robot/          # 机器人用户
│       ├── workspace/
│       └── .config/
│
└── proc/               # 进程信息 (内核虚拟)
    ├── cpuinfo
    ├── interrupts
    ├── meminfo
    └── driver/         # 驱动调试接口
```

---

## 第 5 部分：Linux 内核调试

### 5.1 内核配置（机器人实时性关键）

```kconfig
# 实时抢占 (最关键的配置)
CONFIG_PREEMPT_RT=y                # 全抢占实时内核
CONFIG_HZ_1000=y                  # 1000Hz tick
CONFIG_NO_HZ_FULL=y               # 全动态 tick (减少调度抖动)
CONFIG_CPU_FREQ_DEFAULT_GOV_PERFORMANCE=y  # 固定 CPU 频率 (不让调频引入延迟)

# CPU 隔离 (关键核心专用于控制)
CONFIG_CPU_ISOLATION=y
# 启动参数 isolcpus=4,5,6,7 隔离 4 个大核

# 实时调度
CONFIG_RT_GROUP_SCHED=y

# CAN 总线支持
CONFIG_CAN=y
CONFIG_CAN_RAW=y
CONFIG_CAN_BCM=y
CONFIG_CAN_GW=y
CONFIG_CAN_DEV=y
CONFIG_CAN_MCP251x=y              # SPI 转 CAN 控制器
CONFIG_CAN_FLEXCAN=y              # i.MX 原生 CAN

# SPI / I2C / GPIO
CONFIG_SPI=y
CONFIG_SPI_SPIDEV=y               # 用户空间 SPI
CONFIG_I2C=y
CONFIG_I2C_CHARDEV=y              # 用户空间 I2C
CONFIG_GPIO_SYSFS=y               # GPIO sysfs 接口 (已过时但调试方便)

# 机器人专用: 高精度定时器
CONFIG_HIGH_RES_TIMERS=y

# 散热与电源
CONFIG_THERMAL=y
CONFIG_CPU_THERMAL=y
CONFIG_REGULATOR=y
CONFIG_BATTERY_BQ27XXX=y

# 调试 (生产环境关闭)
CONFIG_DEBUG_KERNEL=y             # 开发阶段
CONFIG_DEBUG_PREEMPT=y            # 抢占调试
CONFIG_PREEMPT_TRACER=y           # 抢占延迟追踪
# CONFIG_DEBUG_KERNEL=n           # 生产: 关闭所有调试
```

### 5.2 驱动调试工具链

```bash
# ===== 开发阶段: 常用调试命令 =====

# 1. CAN 总线监控
# 实时监控所有 CAN 帧
candump can0

# 发送测试帧 (ID=0x100, 数据 0x01 0xA5)
cansend can0 100#01A5000000000000

# CAN 总线状态 (错误计数)
ip -details link show can0
# 关注: "state ERROR-ACTIVE" "tx-errors" "rx-errors"

# 2. I2C 调试
# 扫描总线上的设备 (看到 0x68 表示 IMU 在线)
i2cdetect -y 1

# 读取 IMU WHO_AM_I 寄存器
i2cget -y 1 0x68 0x75

# 写入: 唤醒 IMU (POWER_MGMT0 = 0x4E, 写入 0x0F)
i2cset -y 1 0x68 0x4E 0x0F

# 3. SPI 调试
# 用 spidev_test 收发测试
spidev_test -v -D /dev/spidev0.0
# 应看到发送和接收的数据

# 4. GPIO 调试 (sysfs 方式, 不推荐生产)
echo 15 > /sys/class/gpio/export          # 导出 GPIO15
echo out > /sys/class/gpio/gpio15/direction
echo 1 > /sys/class/gpio/gpio15/value     # 高电平 (使能编码器 SPI CS)
cat /sys/class/gpio/gpio15/value          # 读取

# 5. PWM (通过 sysfs, 调试用)
echo 0 > /sys/class/pwm/pwmchip0/export
echo 20000000 > /sys/class/pwm/pwmchip0/pwm0/period       # 20kHz
echo 10000000 > /sys/class/pwm/pwmchip0/pwm0/duty_cycle  # 50%
echo 1 > /sys/class/pwm/pwmchip0/pwm0/enable

# 6. 中断统计
cat /proc/interrupts | grep -E "can|spi|i2c|gpio"
# 观察中断计数是否增长

# 7. 实时性能测量
# 安装 cyclictest (rt-tests)
cyclictest -m -S -p 98 -i 200 -l 10000
# 输出: Max Latencies (us)
# 目标: worst-case < 100us (1000us = 1ms 不可接受)

# 8. 内核日志
dmesg -wH                          # 实时查看
dmesg | grep -iE "can|spi|i2c|gpio|robot"
dmesg -C                           # 清除日志

# 9. ftrace (内核函数追踪)
echo function_graph > /sys/kernel/debug/tracing/current_tracer
echo foc_calculate > /sys/kernel/debug/tracing/set_ftrace_filter
echo 1 > /sys/kernel/debug/tracing/tracing_on
cat /sys/kernel/debug/tracing/trace       # 查看函数调用图

# 10. perf (性能分析)
perf stat -e cycles,instructions,cache-misses ./robot-control
perf record -g ./robot-control
perf report                                          # 交互式分析
```

### 5.3 内核模块开发示例（CAN 电机驱动）

```c
// drivers/can/robot_joint_can.c
// 一个通过 CAN 总线接收目标位置，通过 SPI 读取编码器的简单内核模块

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/can/core.h>
#include <linux/can/dev.h>
#include <linux/netdevice.h>
#include <linux/spi/spi.h>
#include <linux/workqueue.h>

#define JOINT_CAN_ID_BASE  0x100
#define JOINT_SPI_BUS      0
#define NUM_JOINTS         12

struct joint_device {
    uint8_t  id;
    int16_t  target_angle;    // 来自 CAN (主控)
    int16_t  current_angle;   // 来自 SPI 编码器
    int16_t  current_torque;  // 来自 ADC
    struct spi_device *spi;
    struct work_struct update_work;
};

static struct joint_device joints[NUM_JOINTS];

// CAN 接收回调: 当收到目标角度帧时
static int joint_can_rx_handler(struct sk_buff *skb, void *data)
{
    struct can_frame *cf = (struct can_frame *)skb->data;
    uint8_t joint_id = cf->can_id - JOINT_CAN_ID_BASE;
    
    if (joint_id >= NUM_JOINTS)
        return -EINVAL;
    
    // 检查安全限制 (内核态!)
    int16_t target = (cf->data[0] << 8) | cf->data[1];
    if (target > MAX_ANGLE || target < MIN_ANGLE) {
        pr_err("Joint %d: angle %d exceeds limits!", joint_id, target);
        return -EINVAL;
    }
    
    joints[joint_id].target_angle = target;
    
    // 调度 PID 计算 (在 workqueue 中执行, 不阻塞 CAN RX)
    schedule_work(&joints[joint_id].update_work);
    
    return 0;
}

// 周期性 PID 控制 (在 workqueue 中)
static void joint_update_work(struct work_struct *work)
{
    struct joint_device *j = container_of(work, struct joint_device, update_work);
    
    // 1. SPI 读取编码器 (当前角度)
    uint8_t tx[2] = {0x80, 0x21};  // TLE5012B 读命令
    uint8_t rx[2];
    struct spi_transfer xfer = { .tx_buf = tx, .rx_buf = rx, .len = 2 };
    spi_sync(j->spi, &xfer);
    j->current_angle = (rx[0] << 8) | rx[1];
    
    // 2. PID 计算
    int16_t error = j->target_angle - j->current_angle;
    int16_t pwm = pid_calculate(j->id, error);  // PID 参数 per-joint
    
    // 3. PWM 输出 (通过 SPI 写到 DAC 或直接用 PWM 子系统)
    pwm_set_duty(j->id, pwm);
}

static int __init robot_joint_init(void) {
    int i;
    
    // 注册 CAN 接收过滤器
    for (i = 0; i < NUM_JOINTS; i++) {
        joints[i].id = i;
        joints[i].spi = spi_new_device(spi_bus, &spi_info[i]);
        INIT_WORK(&joints[i].update_work, joint_update_work);
    }
    
    pr_info("Robot joint driver: %d joints online\n", NUM_JOINTS);
    return 0;
}

static void __exit robot_joint_exit(void) {
    int i;
    for (i = 0; i < NUM_JOINTS; i++) {
        cancel_work_sync(&joints[i].update_work);
        spi_unregister_device(joints[i].spi);
    }
}

module_init(robot_joint_init);
module_exit(robot_joint_exit);

MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("EmbodiedAI Robot Joint CAN Driver");
MODULE_AUTHOR("EmbodiedAI Project");
```

---

## 第 6 部分：把一切串起来的系统架构

```
完整嵌入式 Linux 软件栈 (从底层到顶层):

┌─────────────────────────────────────────────────────┐
│ 第 5 层: 应用层 (Python/C++)                         │
│   - ROS2 节点 / VLA 推理 / GUI / 语音交互             │
│   - 运行在: CPU 0-3 (本地核心)                        │
├─────────────────────────────────────────────────────┤
│ 第 4 层: 中间件 (C/C++)                              │
│   - ROS2 DDS / rclcpp / image_transport              │
│   - TensorRT 推理引擎 / CANopen 协议栈                │
├─────────────────────────────────────────────────────┤
│ 第 3 层: Linux 内核 (C)                              │
│   - 实时调度 / CAN 驱动 / SPI 驱动 / I2C 驱动         │
│   - GPU (NPU) 驱动 / camera (V4L2) 驱动              │
│   - 运行在: CPU 4-7 (隔离核心)                        │
├─────────────────────────────────────────────────────┤
│ 第 2 层: 硬件抽象 (寄存器操作)                        │
│   - Jetson Orin SoC 寄存器 / 设备树                    │
│   - GPIO, PWM, DMA, I2C, SPI, CAN, MIPI             │
├─────────────────────────────────────────────────────┤
│ 第 1 层: 物理硬件                                    │
│   - PCB 走线 / 信号 / 电源 / 电机 / 传感器            │
└─────────────────────────────────────────────────────┘

     (Cortex-M 关节板通过 CAN 总线挂到第 3 层)
     
┌─────────────────────────────────────────────────────┐
│ 关节 MCU (STM32G4) 独立运行                          │
│   - FOC 电机控制 (10kHz)                              │
│   - 编码器读取 (SPI)                                  │
│   - 自动关机硬件 (< 1μs)                              │
│   - CAN 通信 (接收目标, 发送状态)                     │
│   - 运行在: 裸机 / FreeRTOS                          │
└─────────────────────────────────────────────────────┘
```

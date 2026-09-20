"""触觉-视觉多速率 EKF — 拆自 climbing_dynamics_core.py (K1 拆分, 逻辑未改)。"""
import numpy as np

class MultiRateEKF:
    """§4.3 触觉-视觉多速率 EKF，μ 状态增广

    状态 x = [f_t (切向力), μ, v_t (切向速度)]（单接触最小实例）
    过程(100Hz):  stick: f_t += k_t·v_t·dt (形变累积), v_t 恒定
                  slip:  f_t ≈ μ·f_n (贴锥面)
    触觉观测(100Hz): z = f_t + noise
    视觉"观测"(20Hz): 材质分类给出 μ 的软测量（先验注入，§4.4 方式A）
    视觉丢帧: 只做时间更新（P 增长）
    """
    def __init__(self, mu_prior_mean, mu_prior_std, f_n, k_t=5000.0,
                 R_tac=0.25, R_vis_mu=0.02, dt=0.01):
        self.x = np.array([5.0, mu_prior_mean, 0.1])   # [f_t, μ, v_t]
        self.P = np.diag([1.0, mu_prior_std**2, 0.1**2])
        self.f_n, self.k_t = f_n, k_t
        self.R_tac, self.R_vis_mu, self.dt = R_tac, R_vis_mu, dt
        self.q_mu = 0.002   # μ 随机游走过程噪声

    def predict(self):
        f_t, mu, v_t = self.x
        if f_t < mu * self.f_n:          # stick: 切向形变累积
            f_next = f_t + self.k_t * v_t * self.dt
        else:                            # slip: 贴锥面
            f_next = mu * self.f_n
        F = np.eye(3)
        F[0, 2] = self.k_t * self.dt if f_t < mu * self.f_n else 0.0
        Q = np.diag([0.05, self.q_mu, 0.02])
        self.x = np.array([f_next, mu, v_t])
        self.P = F @ self.P @ F.T + Q
        return self.x, self.P

    def update_tactile(self, z_f):
        H = np.array([[1.0, 0.0, 0.0]])
        R = np.array([[self.R_tac]])
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + (K @ (z_f - H @ self.x)).ravel()
        I_KH = np.eye(3) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        # slip 边界观测: 若 z_f 贴近 μ f_n，则 μ 也可观
        if abs(z_f - self.x[1] * self.f_n) < 1.0:
            H2 = np.array([[0.0, self.f_n, 0.0]])
            S2 = H2 @ self.P @ H2.T + np.array([[self.R_tac * 4]])
            K2 = self.P @ H2.T @ np.linalg.inv(S2)
            self.x = self.x + (K2 @ (z_f - H2 @ self.x)).ravel()
            I_KH2 = np.eye(3) - K2 @ H2
            self.P = I_KH2 @ self.P @ I_KH2.T + K2 @ np.array([[self.R_tac*4]]) @ K2.T
        return self.x, self.P

    def update_visual_material(self, mu_meas):
        """§4.4 视觉材质先验注入（软测量）"""
        H = np.array([[0.0, 1.0, 0.0]])
        R = np.array([[self.R_vis_mu]])
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + (K @ (mu_meas - H @ self.x)).ravel()
        I_KH = np.eye(3) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        return self.x, self.P

def ekf_demo():
    """§6.5: 视觉误判材质(μ 先验 0.55) → 触觉 100Hz 修正到真值 0.75"""
    rng = np.random.default_rng(42)
    mu_true = 0.75; f_n = 40.0
    ekf = MultiRateEKF(mu_prior_mean=0.55, mu_prior_std=0.15, f_n=f_n)
    f_t_true = 25.0; v_t = 0.003   # 缓慢加载
    print("§6.5 多速率 EKF（μ 在线估计）")
    for step in range(100):        # 1s @ 100Hz
        # 真实系统
        if f_t_true < mu_true * f_n:
            f_t_true += 5000 * v_t * 0.01
        else:
            f_t_true = mu_true * f_n
        ekf.predict()
        z = f_t_true + rng.normal(0, 0.5)
        ekf.update_tactile(z)
        if step % 5 == 0 and step < 40:  # 视觉 20Hz, 前 0.4s 有观测
            if rng.random() > 0.3:       # 30% 丢帧
                ekf.update_visual_material(0.55 + rng.normal(0, 0.1))
        if step in (10, 40, 99):
            print(f"  t={step*0.01:4.2f}s  μ̂={ekf.x[1]:.3f} (真值 {mu_true})  "
                  f"σ_μ={np.sqrt(ekf.P[1,1]):.3f}")
    print(f"  μ 收敛: 误差 {abs(ekf.x[1]-mu_true):.3f}, σ_μ {np.sqrt(ekf.P[1,1]):.3f} "
          f"(从 0.15 起)—— 触觉是 μ 的在线观测源")
    return ekf.x[1], np.sqrt(ekf.P[1, 1])

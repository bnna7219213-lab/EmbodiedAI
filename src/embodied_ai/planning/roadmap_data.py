"""路线图里程碑数据 (原 mcts_integration.py 尾部的 milestones 字典)。

非执行代码, 作为 structured data 保留。"""
milestones_phase1 = {
    'Month 1': {
        'code': [
            '扩展接触模型（点→面→软接触）',
            '实现安全约束检查器',
            '搭建遥操作数据采集系统',
        ],
        'hardware': [
            '集成GelSight触觉传感器',
            '部署RealSense D435',
            '设置安全防护（急停、限力）',
        ],
        'deliverable': '能在简单场景遥操作（抓刚性物体）',
    },
    
    'Month 2': {
        'code': [
            '实现扩展的MCTS（SafeHouseholdMCTS）',
            '添加功能性约束（杯口朝上等）',
            '集成PyBullet仿真器',
        ],
        'data': '采集20小时遥操作演示',
        'deliverable': 'MCTS能规划简单抓取序列',
    },
    
    'Month 3': {
        'code': [
            '训练基础Diffusion Policy（2-3个技能）',
            '实现分层架构（MCTS + Diffusion）',
            '添加实时监控与错误恢复',
        ],
        'eval': '真机测试: 抓取成功率>60%',
        'deliverable': '能执行"抓→放"的完整任务',
    },
}

milestones_phase2 = {
    'Month 4-5': {
        'code': [
            '实现LearnedContactModel',
            '训练触觉特征编码器（自监督）',
            '集成到MCTS作为启发式',
        ],
        'data': '继续采集演示（累计50小时）',
        'metric': '接触点预测准确率>75%',
    },
    
    'Month 6-7': {
        'code': [
            '部署NeRF场景重建',
            '实现NeRFGuidedMCTS',
            '训练NeRF价值网络',
        ],
        'hardware': '升级到多相机系统（3-4个视角）',
        'metric': 'NeRF重建精度<5mm',
    },
    
    'Month 8-9': {
        'integration': [
            '完整系统集成（4层架构）',
            '大规模测试（100+试验）',
            '调优超参数',
        ],
        'eval': {
            'success_rate': '>80% on seen objects',
            'generalization': '>50% on novel objects',
            'safety': '0 human collisions in 100 trials',
        },
        'deliverable': '论文投稿（ICRA/IROS）',
    },
}
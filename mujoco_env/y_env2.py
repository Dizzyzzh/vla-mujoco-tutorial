"""
统一环境入口兼容模块：
直接复用 mujoco_env.y_env.SimpleEnv，避免项目维护两套重复的 MuJoCo 环境代码。
对于 7.pi0.ipynb / 8.smolvla.ipynb 等旧导入（from mujoco_env.y_env2 import SimpleEnv2），
只需将 xml_path 指向 './asset/example_scene_y.xml' 即可无缝运行新场景。
"""

from mujoco_env.y_env import SimpleEnv  # 从主环境模块导入唯一环境类 SimpleEnv

SimpleEnv2 = SimpleEnv  # 暴露 SimpleEnv2 别名以保持对旧 Notebook 导入语句的向后兼容

__all__ = ["SimpleEnv", "SimpleEnv2"]  # 声明对外导出的公共符号列表

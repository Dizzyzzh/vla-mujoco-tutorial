import unittest

import numpy as np
import torch

from mujoco_env.y_env import SimpleEnv
from scripts.train_utils import VisualAugmentor, build_warmup_cosine_scheduler


class TestSortingEnvAndUtils(unittest.TestCase):
    def test_env_modes_reset_and_diagnostics(self) -> None:
        env = SimpleEnv(
            "./asset/example_scene_y.xml",
            action_type="joint_angle",
            headless=True,
            verbose=False,
        )
        try:
            # 1. 标准 4 任务模式与 OOD 场景重置
            env.reset(seed=42, task_idx=0, ood_distractor=False, ood_position=False)
            self.assertEqual(env.obj_target, "body_obj_apple")
            self.assertEqual(env.tray_target, "tray_blue")

            # 2. 同义改写指令解析 + 红牛罐干扰物 + 位置外推模式
            env.reset(
                seed=43,
                instruction="Grab the citrus fruit and place it into the near tray.",
                ood_distractor=True,
                ood_position=True,
            )
            self.assertEqual(env.obj_target, "body_obj_orange")
            self.assertEqual(env.tray_target, "tray_green")

            state = env.get_joint_state()
            self.assertEqual(state.shape, (7,))
            agent_img, wrist_img = env.grab_image(include_side=False)
            self.assertEqual(agent_img.shape, (480, 640, 3))
            self.assertEqual(wrist_img.shape, (480, 640, 3))

            # 3. 验证单步控制、碰撞检测与诊断字典字段
            env.step(state)
            env.step_env()
            self.assertIsInstance(env.check_barrier_collision(), bool)
            self.assertFalse(env.check_success())
            diag = env.get_eval_diagnostics()
            for key in ("tcp_to_target_xy", "target_z", "redbull_active", "success"):
                self.assertIn(key, diag)
        finally:
            env.close()

    def test_visual_augmentor_4d_and_5d(self) -> None:
        aug = VisualAugmentor(color_jitter=0.2, noise_std=0.02, erasing_prob=0.5, augment_wrist=False)
        img_4d = torch.rand(2, 3, 64, 64)
        img_5d = torch.rand(2, 1, 3, 64, 64)
        wrist_5d = torch.rand(2, 1, 3, 64, 64)

        out_4d = aug({"observation.image": img_4d.clone()})["observation.image"]
        self.assertEqual(out_4d.shape, img_4d.shape)
        self.assertGreaterEqual(float(out_4d.min()), 0.0)
        self.assertLessEqual(float(out_4d.max()), 1.0)

        out_5d = aug({"observation.image": img_5d.clone(), "observation.wrist_image": wrist_5d.clone()})
        self.assertEqual(out_5d["observation.image"].shape, img_5d.shape)
        # augment_wrist=False 时应原样保留腕部图像
        self.assertTrue(torch.equal(out_5d["observation.wrist_image"], wrist_5d))

    def test_warmup_cosine_scheduler(self) -> None:
        param = torch.nn.Parameter(torch.zeros(1))
        opt = torch.optim.AdamW([param], lr=1e-3)
        sched = build_warmup_cosine_scheduler(opt, total_steps=100, warmup_steps=10, min_lr_ratio=0.1)
        lrs = []
        for _ in range(100):
            lrs.append(sched.get_last_lr()[0])
            opt.step()
            sched.step()
        self.assertLess(lrs[0], lrs[10])
        self.assertAlmostEqual(lrs[10], 1e-3, places=6)
        self.assertGreater(lrs[10], lrs[-1])


if __name__ == "__main__":
    unittest.main()

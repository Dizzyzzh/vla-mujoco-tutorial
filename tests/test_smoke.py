import unittest
import numpy as np

from mujoco_env.expert import ExpertConfig, PrivilegedSortingExpert
from mujoco_env.y_env import SimpleEnv


class TestSortingEnvAndExpert(unittest.TestCase):
    def test_env_modes_and_expert_rollout(self) -> None:
        env = SimpleEnv(
            "./asset/example_scene_y.xml",
            action_type="joint_angle",
            headless=True,
            verbose=False,
        )
        try:
            # 1. Check standard & OOD reset + language synonym parsing
            env.reset(
                seed=42,
                instruction="Grab the red fruit and drop it into the far container.",
                ood_distractor=True,
                ood_position=True,
            )
            self.assertEqual(env.obj_target, "body_obj_apple")
            self.assertEqual(env.tray_target, "tray_blue")
            state = env.get_joint_state()
            self.assertEqual(state.shape, (7,))
            agent_img, wrist_img = env.grab_image(include_side=False)
            self.assertEqual(agent_img.shape, (480, 640, 3))
            self.assertEqual(wrist_img.shape, (480, 640, 3))

            # 2. Verify privileged expert completes a full obstacle-crossing episode cleanly
            env.reset(seed=100, task_idx=0, ood_distractor=False, ood_position=False)
            expert = PrivilegedSortingExpert(ExpertConfig(randomize=False))
            expert.reset(env, np.random.default_rng(100))

            substeps = int(round(1.0 / (env.env.dt * 20)))
            hit_barrier = False
            success = False
            for _ in range(260):
                action, failed, reason = expert.get_action(env)
                self.assertFalse(failed, f"Expert failed unexpectedly: {reason}")
                env.step(action)
                for _ in range(substeps):
                    env.step_env()
                    if env.check_barrier_collision():
                        hit_barrier = True
                if env.check_success():
                    success = True
                    break

            self.assertFalse(hit_barrier, "Expert trajectory collided with the barrier wall")
            self.assertTrue(success, "Expert trajectory did not satisfy task success criteria")
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()

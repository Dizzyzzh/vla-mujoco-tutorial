import numpy as np
from datetime import datetime
from PIL import Image
import lerobot
from mujoco_env.y_env import SimpleEnv
from lerobot.datasets.lerobot_dataset import LeRobotDataset

SEED = 0
# SEED = None <- Uncomment this line to randomize the object positions
REPO_NAME = "omy_pnp"
NUM_DEMO = 1  # Number of demonstrations to collect
TASK_NAME = "Put mug cup on the plate"
XML_PATH = "./asset/example_scene_y.xml"
CREATE_NEW_DATASET = (
    True  # Set to False if you want to resume recording on an existing dataset
)
OLD_DATASET_PATH = (
    "./data/demo_data_20261117_161335"  # Only used when CREATE_NEW_DATASET = False
)

ROOT = f"./data/demo_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
print(f"lerobot version: {lerobot.__version__}")

if CREATE_NEW_DATASET:
    print(f"Create new dataset: {ROOT}")
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        root=ROOT,
        robot_type="omy",
        fps=20,  # 20 frames per second
        features={
            "observation.image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channels"],
            },
            "observation.wrist_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channels"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (6,),
                "names": ["state"],  # x, y, z, roll, pitch, yaw
            },
            "action": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["action"],  # 6 joint angles and 1 gripper
            },
            "obj_init": {
                "dtype": "float32",
                "shape": (6,),
                "names": ["obj_init"],  # initial object pose, not used in training
            },
        }
    )
else:
    print(f"Resume from previous dataset: {OLD_DATASET_PATH}")
    dataset = LeRobotDataset.resume(
        repo_id=REPO_NAME,
        root=OLD_DATASET_PATH,
    )
    print(f"Existing episodes: {dataset.meta.total_episodes}")

# Define the environment
PnPEnv = SimpleEnv(XML_PATH, seed=SEED)
action = np.zeros(7)
episode_id = 0
record_flag = False  # Start recording when the robot starts moving

try:
    while PnPEnv.env.is_viewer_alive() and episode_id < NUM_DEMO:
        PnPEnv.step_env()
        if PnPEnv.env.loop_every(HZ=20):
            done = PnPEnv.check_success()
            if done:
                if dataset.has_pending_frames():
                    dataset.save_episode()
                    episode_id += 1
                    print(f"Saved episode {episode_id}/{NUM_DEMO} to {dataset.root}")
                else:
                    print(
                        "Success detected before any frame was recorded, skip saving."
                    )
                PnPEnv.reset(seed=SEED)
                record_flag = False
                action = np.zeros(7)
            else:
                action, reset = PnPEnv.teleop_robot()

                if not record_flag and np.any(np.abs(action) > 1e-6):
                    record_flag = True
                    print("Start recording")
                    
                if reset:
                    # Press 'z' to reset the environment and clear the current episode buffer
                    PnPEnv.reset(seed=SEED)
                    dataset.clear_episode_buffer()
                    record_flag = False
                    action = np.zeros(7)

                ee_pose = np.asarray(PnPEnv.get_ee_pose(), dtype=np.float32)
                agent_image, wrist_image = PnPEnv.grab_image()
                agent_image = np.asarray(
                    Image.fromarray(agent_image).resize((256, 256)), dtype=np.uint8
                )
                wrist_image = np.asarray(
                    Image.fromarray(wrist_image).resize((256, 256)), dtype=np.uint8
                )
                joint_q = np.asarray(PnPEnv.step(action), dtype=np.float32)

                if record_flag:
                    dataset.add_frame(
                        {
                            "observation.image": agent_image,
                            "observation.wrist_image": wrist_image,
                            "observation.state": ee_pose,
                            "action": joint_q,
                            "obj_init": np.asarray(
                                PnPEnv.obj_init_pose, dtype=np.float32
                            ),
                            "task": TASK_NAME,
                        }
                    )
            PnPEnv.render(teleop=True)
finally:
    if dataset.has_pending_frames():
        print("Discard unfinished episode.")
        dataset.clear_episode_buffer()
    dataset.finalize()
    PnPEnv.env.close_viewer()

print(f"Dataset saved to: {dataset.root}")

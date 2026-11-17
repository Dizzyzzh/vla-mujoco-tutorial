import numpy as np
from datetime import datetime
from PIL import Image
from mujoco_env.y_env import SimpleEnv
from lerobot.datasets.lerobot_dataset import LeRobotDataset

SEED = 0
# SEED = None <- Uncomment this line to randomize the object positions
REPO_NAME = "omy_pnp"
NUM_DEMO = 1  # Number of demonstrations to collect
TASK_NAME = "Put mug cup on the plate"
XML_PATH = "./asset/example_scene_y.xml"
CREATE_NEW_DATASET = True  # Set to False if you want to load from a previous dataset
OLD_DATASET_PATH = "./data/demo_data_20261117_161335"  # 旧数据集路径，如果 CREATE_NEW_DATASET = False，则会从该路径加载数据集

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
ROOT = f"./data/demo_data_{timestamp}"
print(f"Save dataset to: {ROOT}")

if CREATE_NEW_DATASET:
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
                "names": ["height", "width", "channel"],
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
                "names": [
                    "obj_init"
                ],  # just the initial position of the object. Not used in training.
            },
        },
    )
else:
    print("Load from previous dataset")
    dataset = LeRobotDataset(REPO_NAME, root=OLD_DATASET_PATH)


# Define the environment
PnPEnv = SimpleEnv(XML_PATH, seed=SEED)

action = np.zeros(7)
episode_id = 0
record_flag = False  # Start recording when the robot starts moving
while PnPEnv.env.is_viewer_alive() and episode_id < NUM_DEMO:
    PnPEnv.step_env()
    if PnPEnv.env.loop_every(HZ=20):
        # check if the episode is done
        done = PnPEnv.check_success()
        if done:
            # Save the episode data and reset the environment
            dataset.save_episode()
            PnPEnv.reset(seed=SEED)
            episode_id += 1
            # Reset recording state for the next episode
            record_flag = False
            action = np.zeros(7)
        else:
            # Teleoperate the robot and get delta end-effector pose with gripper
            action, reset = PnPEnv.teleop_robot()
            if not record_flag and np.any(np.abs(action) > 1e-6):
                record_flag = True
                print("Start recording")
            if reset:
                # Reset the environment and clear the episode buffer
                # This can be done by pressing 'z' key
                PnPEnv.reset(seed=SEED)
                dataset.clear_episode_buffer()
                record_flag = False
                action = np.zeros(7)
            # Step the environment
            # Get the end-effector pose and images
            ee_pose = PnPEnv.get_ee_pose()
            agent_image, wrist_image = PnPEnv.grab_image()
            agent_image = np.array(Image.fromarray(agent_image).resize((256, 256)))
            wrist_image = np.array(Image.fromarray(wrist_image).resize((256, 256)))
            joint_q = PnPEnv.step(action)
            if record_flag:
                # Add the frame to the dataset
                frame = {
                    "observation.image": agent_image,
                    "observation.wrist_image": wrist_image,
                    "observation.state": ee_pose,
                    "action": joint_q,
                    "obj_init": PnPEnv.obj_init_pose,
                    "task": TASK_NAME,
                }
                dataset.add_frame(frame)
        PnPEnv.render(teleop=True)


dataset.finalize()
print("Dataset saved successfully.")

PnPEnv.env.close_viewer()

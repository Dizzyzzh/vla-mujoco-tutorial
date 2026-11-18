import numpy as np
from torch.utils.data import DataLoader
from lerobot.datasets.utils import serialize_dict
from lerobot.utils.io_utils import write_json
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from mujoco_env.y_env import SimpleEnv

# if youu want to use the example data provided, DATAPATH = './demo_data_example' instead!
DATAPATH = './data/demo_data_20261118_085434'  # Path to the dataset

# Select an episode index that you want to visualize
episode_index = 0

dataset = LeRobotDataset('omy_pnp', root=DATAPATH, episodes=[episode_index])
dataloader = DataLoader(dataset, batch_size=1, num_workers=0)

xml_path = './asset/example_scene_y.xml'
PnPEnv = SimpleEnv(xml_path, action_type='joint_angle')

step = 0
iter_dataloader = iter(dataloader)
PnPEnv.reset()

while PnPEnv.env.is_viewer_alive():
    PnPEnv.step_env()
    if PnPEnv.env.loop_every(HZ=20):
        # Get the action from dataset
        data = next(iter_dataloader)
        if step == 0:
            # Reset the object pose based on the dataset
            PnPEnv.set_obj_pose(
                data['obj_init'][0, :3].numpy(),
                data['obj_init'][0, 3:].numpy()
            )
        # Get the action from dataset
        action = data['action'].numpy()
        PnPEnv.step(action[0])

        # Visualize the image from dataset to rgb_overlay
        PnPEnv.rgb_agent = (data['observation.image'][0].numpy() * 255).astype(np.uint8)
        PnPEnv.rgb_ego = (data['observation.wrist_image'][0].numpy() * 255).astype(np.uint8)
        # 3 256 256 -> 256 256 3
        PnPEnv.rgb_agent = np.transpose(PnPEnv.rgb_agent, (1,2,0))
        PnPEnv.rgb_ego = np.transpose(PnPEnv.rgb_ego, (1,2,0))
        PnPEnv.rgb_side = np.zeros((480, 640, 3), dtype=np.uint8)
        PnPEnv.render()
        step += 1

        if step == len(dataset):
            # start from the beginning
            iter_dataloader = iter(dataloader)
            PnPEnv.reset()
            step = 0

PnPEnv.env.close_viewer()

stats = dataset.meta.stats
PATH = dataset.root / 'meta' / 'stats.json'
stats = serialize_dict(stats)

write_json(stats, PATH)
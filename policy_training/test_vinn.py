import os
from envs.kinova_onrobot import KinovaOnrobotEnv
from agent.vinn import VINNAgent, VINNDataset, compute_kinova_onrobot_distance
import h5py
from tqdm import tqdm
from utils import DATA_DIR


if __name__ == "__main__":
    # Load dataset
    with h5py.File(
        os.path.join(DATA_DIR, "reach_green_cup/reach_green_cup_processed.h5"), "r"
    ) as f:
        demo_dict = {}
        for k in f.keys():
            demo_dict[k] = f[k][()]
    dataset = VINNDataset(demo_dict, repr_keys=["kinova", "onrobot"])
    agent = VINNAgent(
        device="cpu",
        candidate_dataset=dataset,
        precomp_embeddings=True,
        encoder=None,
        num_neighbors=1,
        distance_metric=lambda x, y: compute_kinova_onrobot_distance(
            x, y, [10, 0.0, 0.002]
        ),
    )
    env = KinovaOnrobotEnv()
    base_state = env.base_state
    gripper_base_state = env.gripper_base_state
    state = base_state + gripper_base_state

    obs = env.reset()
    input("Move robot to desired location")
    # obs = env.step(demo_dict["kinova"][0].tolist() + env.gripper_base_state)[0]

    for _ in tqdm(range(1000)):
        state = obs["features"]
        action, nns = agent.act(state, 0)
        obs = env.step(action)[0]
    env.close()

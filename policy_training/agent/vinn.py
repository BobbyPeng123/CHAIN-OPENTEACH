import os
import numpy as np
from scipy.spatial.transform import Rotation as R
from torchvision.io import read_image

from utils import DATA_DIR


class VINNDataset:
    """
    A dataset class for VINN (Visual Imitation with Nearest Neighbors) data.

    Args:
        demo_dict (dict): A dictionary containing the demo data.
        repr_keys (str or list): The key(s) to access the representation data in the demo_dict.
        action_mode (str, optional): The mode for selecting actions. Defaults to "next".
        action_keys (list, optional): The key(s) to access the action data in the demo_dict. Defaults to ["kinova", "onrobot"].

    Attributes:
        demo_dict (dict): A dictionary containing the demo data.
        repr_keys (str or list): The key(s) to access the representation data in the demo_dict.
        transform (None or callable): A transformation function to apply to the image data.
        obs_arr (ndarray): The observation array containing the representation data.
        actions (ndarray): The action array.

    Methods:
        __getitem__(self, idx): Returns the observation and action at the given index.
        __len__(self): Returns the length of the dataset.

    Raises:
        ValueError: If an invalid repr_key is provided.

    """

    def __init__(
        self,
        demo_dict,
        repr_keys,
        action_mode="next",
        action_keys=["kinova", "onrobot"],
    ) -> None:
        self.demo_dict = demo_dict
        self.repr_keys = repr_keys

        self.transform = None
        action_data = []
        eids = demo_dict["episode_ids"]

        # Extract observation data
        if isinstance(self.repr_keys, list) and len(self.repr_keys) == 1:
            self.repr_keys = self.repr_keys[0]
        if isinstance(self.repr_keys, list):
            for k in self.repr_keys:
                assert "cam" not in k, "Image data not supported with other data"
                if len(demo_dict[k].shape) == 1:
                    demo_dict[k] = np.expand_dims(demo_dict[k], axis=-1)
            self.obs_arr = np.concatenate(
                [demo_dict[k] for k in self.repr_keys], axis=-1
            )
        elif isinstance(self.repr_keys, str):
            self.obs_arr = demo_dict[self.repr_keys]
        else:
            raise ValueError("Invalid repr_key")

        # Extract action data
        if action_mode == "next":
            for i in range(len(eids) - 1):
                actions = []
                for k in action_keys:
                    if len(demo_dict[k].shape) == 1:
                        demo_dict[k] = np.expand_dims(demo_dict[k], axis=-1)
                    actions.append(demo_dict[k][eids[i] + 1 : eids[i + 1]])
                action_data.append(np.concatenate(actions, axis=-1))
                action_data.append(action_data[-1][-1:])
            self.actions = np.concatenate(action_data, axis=0)

    def __getitem__(self, idx):
        if "cam" in self.repr_keys:
            img_path = os.path.join(DATA_DIR, self.obs_arr[idx].decode("ascii"))
            image = read_image(img_path)
            if self.transform:
                image = self.transform(image)
            obs = image
        else:
            obs = self.obs_arr[idx]
        return obs, self.actions[idx]

    def __len__(self):
        return len(self.obs_arr)


def compute_kinova_onrobot_distance(x, y, coeffs, print_err=False):
    trans_err = np.linalg.norm(x[:3] - y[:3])
    rot_err = (R.from_quat(x[3:7]) * R.from_quat(y[3:7]).inv()).magnitude()
    gripper_err = np.linalg.norm(x[7:] - y[7:])
    err_arr = [trans_err, rot_err, gripper_err]
    err = np.dot(coeffs, err_arr)
    if print_err:
        print(trans_err, gripper_err, coeffs * np.array(err_arr))
    return err


class VINNAgent:
    def __init__(
        self,
        device,
        candidate_dataset,
        precomp_embeddings,
        encoder,
        num_neighbors=5,
        distance_metric=lambda x, y: np.linalg.norm(x - y, axis=-1),
    ):
        self.device = device
        self.candidate_dataset = candidate_dataset
        self.precomp_embeddings = precomp_embeddings
        self.encoder = encoder
        self.num_neighbors = num_neighbors
        self.distance_metric = distance_metric

        if not precomp_embeddings:
            self.setup()

    def __repr__(self) -> str:
        return "vinn"

    def setup(self):
        # TODO: Compute embeddings for the candidate dataset if not already computed
        pass

    def act(self, obs, step):
        # Compute distances from images in the neighbor set
        # to the current image
        # Compute the embedding of the current image
        if self.encoder is not None:
            obs = self.encoder(obs)
        top_k = self.get_neighbor_ids(obs)
        # Compute the action from the top k neighbors
        action = self._compute_action_from_neighbors(self.candidate_dataset[top_k][1])
        return action, self.candidate_dataset[top_k][0]

    def get_neighbor_ids(self, embed):
        distances = []
        for neighbor, label in self.candidate_dataset:
            distances.append(self.distance_metric(neighbor, embed))

        # Find the closest N images in the neighbor set
        # to the current image
        top_k = np.argsort(distances)[: self.num_neighbors]
        y = self.candidate_dataset[top_k][0][0]
        trans_err = np.linalg.norm(embed[:3] - y[:3])
        gripper_err = np.linalg.norm(embed[7:] - y[7:])
        print(trans_err, gripper_err)
        print(10 * trans_err, 0.001 * gripper_err)
        return top_k

    def _compute_action_from_neighbors(self, top_k_actions):
        # Compute the action from the top k neighbors
        action = np.mean(top_k_actions, axis=0)
        return action

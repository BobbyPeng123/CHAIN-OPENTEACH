import h5py as h5
import numpy as np
from pandas import read_csv
import pickle as pkl
import cv2
from pathlib import Path
from scipy.spatial.transform import Rotation as R

from utils import DATA_DIR, SAVE_PKL_DIR

task_names = ["0812_pick_yoghurt_out_from_fridge_venky"]
PROCESSED_DATA_PATH = Path(DATA_DIR) / "processed_data/"

SAVE_DATA_PATH = Path(SAVE_PKL_DIR) / "processed_data_pkl/"

camera_indices = [1, 2, 51, 52]
img_size = (128, 128)
NUM_DEMOS = None
use_sensor = True

# if task_names is None, get all task names
if task_names is None:
    task_names = [x.name for x in PROCESSED_DATA_PATH.iterdir() if x.is_dir()]

# Create the save path
SAVE_DATA_PATH.mkdir(parents=True, exist_ok=True)

for TASK_NAME in task_names:
    DATASET_PATH = Path(f"{PROCESSED_DATA_PATH}/{TASK_NAME}")

    if (SAVE_DATA_PATH / f"{TASK_NAME}.pkl").exists():
        print(f"Data for {TASK_NAME} already exists. Appending to it...")
        input("Press Enter to continue...")
        data = pkl.load(open(SAVE_DATA_PATH / f"{TASK_NAME}.pkl", "rb"))
        observations = data["observations"]
        max_cartesian = data["max_cartesian"]
        min_cartesian = data["min_cartesian"]
        max_gripper = data["max_gripper"]
        min_gripper = data["min_gripper"]
        # task_emb = data["task_emb"]
    else:
        # Get task name sentence
        # label_path = DATASET_PATH / "label.txt"
        # task_name = label_path.read_text().strip()
        # print(f"Task name: {task_name}")
        # lang_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        # task_emb = lang_model.encode(task_name)

        # Init storing variables
        observations = []

        # Store max and min
        max_cartesian, min_cartesian = None, None
        max_sensor, min_sensor = None, None
        # max_rel_cartesian, min_rel_cartesian = None, None
        max_gripper, min_gripper = None, None

    # Load each data point and save in a list
    dirs = [x for x in DATASET_PATH.iterdir() if x.is_dir()]
    for i, data_point in enumerate(dirs):
        use_sensor = True
        print(f"Processing data point {i+1}/{len(dirs)}")

        if NUM_DEMOS is not None:
            if int(str(data_point).split("_")[-1]) >= NUM_DEMOS:
                print(f"Skipping data point {data_point}")
                continue

        observation = {}
        # images
        image_dir = data_point / "videos"
        if not image_dir.exists():
            print(f"Data point {data_point} is incomplete")
            continue
        for save_idx, idx in enumerate(camera_indices):
            # Read the frames in the video
            video_path = image_dir / f"camera{idx}.mp4"
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                print(f"Video {video_path} could not be opened")
                continue
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if idx == 52:
                    # crop the right side of the image for the gripper cam
                    shape = frame.shape
                    crop_percent = 0.2
                    frame = frame[:, : int(shape[1] * (1 - crop_percent))]
                frame = cv2.resize(frame, img_size)
                frames.append(frame)
            observation[f"pixels{idx}"] = np.array(frames)
        # read cartesian and gripper states from csv
        state_csv_path = data_point / "states.csv"
        sensor_csv_path = data_point / "sensor.csv"
        state = read_csv(state_csv_path)
        try:
            sensor_data = read_csv(sensor_csv_path)
            sensor_states = sensor_data["sensor_values"].values
            sensor_states = np.array(
                [
                    np.array([float(x.strip()) for x in sensor[1:-1].split(",")])
                    for sensor in sensor_states
                ],
                dtype=np.float32,
            )
        except FileNotFoundError:
            use_sensor = False
            print(f"Sensor data not found for {data_point}")

        # Read cartesian state where every element is a 6D pose
        # Separate the pose into values instead of string
        cartesian_states = state["pose_aa"].values
        cartesian_states = np.array(
            [
                np.array([float(x.strip()) for x in pose[1:-1].split(",")])
                for pose in cartesian_states
            ],
            dtype=np.float32,
        )

        # cartesian_pos = cartesian_states[:, :3]
        # cartesian_ori = cartesian_states[:, 3:]
        # # cartesian_ori = wrap_angle(cartesian_ori)
        # cartesian_ori = np.concatenate([np.sin(cartesian_ori), np.cos(cartesian_ori)], axis=1)
        # cartesian_states = np.concatenate([cartesian_pos, cartesian_ori], axis=1)
        # rest
        gripper_states = state["gripper_state"].values.astype(np.float32)
        observation["cartesian_states"] = cartesian_states.astype(np.float32)
        observation["gripper_states"] = gripper_states.astype(np.float32)
        if use_sensor:
            observation["sensor_states"] = sensor_states.astype(np.float32)
            if max_sensor is None:
                max_sensor = np.max(sensor_states)
                min_sensor = np.min(sensor_states)
            else:
                max_sensor = np.maximum(max_sensor, np.max(sensor_states))
                min_sensor = np.minimum(min_sensor, np.min(sensor_states))
            max_sensor = np.max(sensor_states, axis=0)
            min_sensor = np.min(sensor_states, axis=0)

        # update max and min
        if max_cartesian is None:
            max_cartesian = np.max(cartesian_states, axis=0)
            min_cartesian = np.min(cartesian_states, axis=0)
        else:
            max_cartesian = np.maximum(max_cartesian, np.max(cartesian_states, axis=0))
            min_cartesian = np.minimum(min_cartesian, np.min(cartesian_states, axis=0))
        if max_gripper is None:
            max_gripper = np.max(gripper_states)
            min_gripper = np.min(gripper_states)
        else:
            max_gripper = np.maximum(max_gripper, np.max(gripper_states))
            min_gripper = np.minimum(min_gripper, np.min(gripper_states))

        # append to observations
        observations.append(observation)

    # Save the data
    data = {
        "observations": observations,
        "max_cartesian": max_cartesian,
        "min_cartesian": min_cartesian,
        "max_gripper": max_gripper,
        "min_gripper": min_gripper,
        "max_sensor": max_sensor,
        "min_sensor": min_sensor,
        # "task_emb": task_emb,
    }
    pkl.dump(data, open(SAVE_DATA_PATH / f"{TASK_NAME}.pkl", "wb"))

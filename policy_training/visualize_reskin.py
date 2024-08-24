import cv2
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
import pandas as pd

if __name__ == "__main__":
    expt_name = "0729_erase_circle_tactile"
    demo_id = 30
    camera_ids = [1, 51]
    fps = 50
    num_frame_limit = None

    num_frames = num_frame_limit
    data_dir = Path(
        f"/mnt/robotlab/siddhant/tactile_openteach/Open-Teach/data/processed_data/{expt_name}/demonstration_{demo_id}"
    )
    video_paths = [data_dir / f"videos/camera{cid}.mp4" for cid in camera_ids]
    sensor_path = data_dir / "sensor.csv"

    frames_dict = defaultdict(list)
    for cid, vpath in zip(camera_ids, video_paths):
        cap = cv2.VideoCapture(str(vpath))
        if not cap.isOpened():
            print(f"Video {vpath} could not be opened")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames_dict[cid].append(frame_rgb)

    if num_frames is None:
        num_frames = len(frames_dict[camera_ids[0]])

    sensor_data = pd.read_csv(sensor_path)
    sensor_states = sensor_data["sensor_values"].values
    sensor_states = np.array(
        [
            np.array([float(x.strip()) for x in sensor[1:-1].split(",")])
            for sensor in sensor_states
        ],
        dtype=np.float32,
    )
    sensor_states = sensor_states - sensor_states[0]
    fig, ax = plt.subplots(max(2, len(camera_ids)), 2)

    im_plots = {}
    for pid, cid in enumerate(camera_ids):
        row, col = int(pid // 2), pid % 2
        im_plots[cid] = ax[row, col].imshow(frames_dict[cid][0])
        ax[row, col].get_xaxis().set_visible(False)
        ax[row, col].get_yaxis().set_visible(False)

    sensor_ims = [
        ax[-1, t].plot(sensor_states[:, t * 15 : (t + 1) * 15]) for t in range(2)
    ]

    def update(frame_t):
        for cid in camera_ids:
            im_plots[cid].set_array(frames_dict[cid][frame_t])
        for t, (line1, line2) in enumerate(zip(*sensor_ims)):
            line1.set_data(range(frame_t), sensor_states[:frame_t, t])
            line2.set_data(range(frame_t), sensor_states[:frame_t, 15 + t])

    ani = animation.FuncAnimation(
        fig=fig,
        func=update,
        frames=len(frames_dict[camera_ids[0]]),
        interval=int(1000 / fps),
    )
    output_path = f"visualizations/{expt_name}_{demo_id}.mp4"
    ani.save(output_path, writer="ffmpeg", fps=fps)
    print(f"Saved visualization to {output_path}")
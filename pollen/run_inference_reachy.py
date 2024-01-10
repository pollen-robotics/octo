import logging
import os
import time

import cv2
import jax
import numpy as np
from reachy_sdk import ReachySDK
from reachy_sdk.trajectory import goto
from reachy_sdk.trajectory.interpolation import InterpolationMode

from octo.model.octo_model import OctoModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# reachy = ReachySDK("localhost")
# reachy = ReachySDK("192.168.0.106")
reachy = ReachySDK("192.168.1.252")
logging.basicConfig(level=logging.INFO)

model = OctoModel.load_pretrained("/data1/apirrone/octo/new_training1_ws2_ph4_5hz_no_proprio_nofreeze/")
print("===")
print(model.get_pretty_spec())
print("===")
task = model.create_tasks(texts=["Grab the can"])
SAMPLING_FREQ = 5


def get_state():
    present_positions = []
    l_arm_present_pos = []
    l_arm_present_pos.append(reachy.l_arm.l_shoulder_pitch.present_position)
    l_arm_present_pos.append(reachy.l_arm.l_shoulder_roll.present_position)
    l_arm_present_pos.append(reachy.l_arm.l_arm_yaw.present_position)
    l_arm_present_pos.append(reachy.l_arm.l_elbow_pitch.present_position)
    l_arm_present_pos.append(reachy.l_arm.l_forearm_yaw.present_position)
    l_arm_present_pos.append(reachy.l_arm.l_wrist_pitch.present_position)
    l_arm_present_pos.append(reachy.l_arm.l_wrist_roll.present_position)
    l_arm_present_pos.append(reachy.l_arm.l_gripper.present_position)

    r_arm_present_pos = []
    r_arm_present_pos.append(reachy.r_arm.r_shoulder_pitch.present_position)
    r_arm_present_pos.append(reachy.r_arm.r_shoulder_roll.present_position)
    r_arm_present_pos.append(reachy.r_arm.r_arm_yaw.present_position)
    r_arm_present_pos.append(reachy.r_arm.r_elbow_pitch.present_position)
    r_arm_present_pos.append(reachy.r_arm.r_forearm_yaw.present_position)
    r_arm_present_pos.append(reachy.r_arm.r_wrist_pitch.present_position)
    r_arm_present_pos.append(reachy.r_arm.r_wrist_roll.present_position)
    r_arm_present_pos.append(reachy.r_arm.r_gripper.present_position)

    neck_present_pos = []
    neck_present_pos.append(reachy.head.neck_roll.present_position)
    neck_present_pos.append(reachy.head.neck_pitch.present_position)
    neck_present_pos.append(reachy.head.neck_yaw.present_position)

    present_positions.extend(l_arm_present_pos)
    present_positions.extend(r_arm_present_pos)
    present_positions.extend(neck_present_pos)

    # normalize
    present_positions = (
        present_positions - model.dataset_statistics["proprio"]["mean"]
    ) / model.dataset_statistics["proprio"]["std"]

    # return present_positions
    return np.array(present_positions, dtype=np.float32)


# def get_vel(prev_state, dt):
#     current_state = get_state()
#     vel = (current_state - prev_state) / dt
#     ret = np.array(vel, dtype=np.float32)
#     ret = np.expand_dims(ret, axis=0)
#     ret = np.expand_dims(ret, axis=0)
#     return np.array(ret, dtype=np.float32)
#     # return vel


def get_image():
    im = cv2.resize(reachy.right_camera.last_frame, (256, 256))
    return np.array(im, dtype=np.uint8)


def exec_goto(action):
    left_action = action[:8]
    right_action = action[8:16]
    neck_action = action[16:]

    pos = {
        reachy.l_arm.l_shoulder_pitch: left_action[0],
        reachy.l_arm.l_shoulder_roll: left_action[1],
        reachy.l_arm.l_arm_yaw: left_action[2],
        reachy.l_arm.l_elbow_pitch: left_action[3],
        reachy.l_arm.l_forearm_yaw: left_action[4],
        reachy.l_arm.l_wrist_pitch: left_action[5],
        reachy.l_arm.l_wrist_roll: left_action[6],
        reachy.l_arm.l_gripper: left_action[7],
        reachy.r_arm.r_shoulder_pitch: right_action[0],
        reachy.r_arm.r_shoulder_roll: right_action[1],
        reachy.r_arm.r_arm_yaw: right_action[2],
        reachy.r_arm.r_elbow_pitch: right_action[3],
        reachy.r_arm.r_forearm_yaw: right_action[4],
        reachy.r_arm.r_wrist_pitch: right_action[5],
        reachy.r_arm.r_wrist_roll: right_action[6],
        reachy.r_arm.r_gripper: right_action[7],
        # reachy.head.neck_roll: 0,
        # reachy.head.neck_pitch: 45,
        # reachy.head.neck_yaw: 0,
    }

    goto(
        goal_positions=pos,
        duration=1/SAMPLING_FREQ,
        interpolation_mode=InterpolationMode.MINIMUM_JERK
	)


def set_joints(action):
    left_action = action[:8]
    right_action = action[8:16]
    neck_action = action[16:]
    reachy.l_arm.l_shoulder_pitch.goal_position = left_action[0]
    reachy.l_arm.l_shoulder_roll.goal_position = left_action[1]
    reachy.l_arm.l_arm_yaw.goal_position = left_action[2]
    reachy.l_arm.l_elbow_pitch.goal_position = left_action[3]
    reachy.l_arm.l_forearm_yaw.goal_position = left_action[4]
    reachy.l_arm.l_wrist_pitch.goal_position = left_action[5]
    reachy.l_arm.l_wrist_roll.goal_position = left_action[6]
    reachy.l_arm.l_gripper.goal_position = left_action[7]

    reachy.r_arm.r_shoulder_pitch.goal_position = right_action[0]
    reachy.r_arm.r_shoulder_roll.goal_position = right_action[1]
    reachy.r_arm.r_arm_yaw.goal_position = right_action[2]
    reachy.r_arm.r_elbow_pitch.goal_position = right_action[3]
    reachy.r_arm.r_forearm_yaw.goal_position = right_action[4]
    reachy.r_arm.r_wrist_pitch.goal_position = right_action[5]
    reachy.r_arm.r_wrist_roll.goal_position = right_action[6]
    reachy.r_arm.r_gripper.goal_position = right_action[7]

    reachy.head.neck_roll.goal_position = 0
    reachy.head.neck_pitch.goal_position = 45
    reachy.head.neck_yaw.goal_position = 0

    # reachy.head.neck_roll.goal_position = neck_action[0]
    # reachy.head.neck_pitch.goal_position = neck_action[1]
    # reachy.head.neck_yaw.goal_position = neck_action[2]


history_size = 2
t0 = time.time()
im_history = [np.zeros((256, 256, 3), dtype=np.float32) for _ in range(history_size)]
# state_history = [np.zeros((19), dtype=np.float32) for _ in range(history_size)]
pm = [False for _ in range(history_size)]
prev_t = time.time()
while True:
    t = time.time() - t0
    im = get_image()
    # state = get_state()
    im_history.append(im)
    im_history = im_history[1:]

    # state_history.append(state)
    # state_history = state_history[1:]

    pm.append(True)
    pm = pm[1:]

    ims = np.expand_dims(np.stack(tuple(im_history)), axis=0)
    # states = np.expand_dims(np.stack(tuple(state_history)), axis=0)
    pad_mask = np.array([pm])

    # pad_mask = np.array([[False if prev_state is None else True, True]])
    # timestep = np.array([[prev_t, t]])

    observations = {
        "image_primary": ims,
        # "proprio": states,
        "pad_mask": pad_mask,
        # "timestep" : timestep
    }

    start = time.time()
    actions = model.sample_actions(observations, task, rng=jax.random.PRNGKey(0))[0]
    print("Sampling actions took ", time.time() - start, " seconds")

    prev_t = t
    
    # Unnormalize
    actions = (
        actions * model.dataset_statistics["action"]["std"]
        + model.dataset_statistics["action"]["mean"]
    )
    # set_joints(np.array(actions[-1]))
    # time.sleep(0.03)
    exec_goto(actions[0])
    # for i, step in enumerate(actions):
    #     exec_goto(step)
        # set_joints(np.array(step))
        # time.sleep(0.03)
        # time.sleep(1/len(actions[:5]))

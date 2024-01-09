import logging
import os
import time

import cv2
import jax
import numpy as np
from reachy_sdk import ReachySDK

from octo.model.octo_model import OctoModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# reachy = ReachySDK("localhost")
reachy = ReachySDK("192.168.1.252")
logging.basicConfig(level=logging.INFO)

model = OctoModel.load_pretrained("/data1/apirrone/octo/trainings6_nofreeze_new_dataset/")
print("===")
print(model.get_pretty_spec())
print("===")
task = model.create_tasks(texts=["Grab the can"])


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

    # reachy.head.neck_roll.goal_position = 0
    # reachy.head.neck_pitch.goal_position = 45
    # reachy.head.neck_yaw.goal_position = 0

    reachy.head.neck_roll.goal_position = neck_action[0]
    reachy.head.neck_pitch.goal_position = neck_action[1]
    reachy.head.neck_yaw.goal_position = neck_action[2]


t0 = time.time()
prev_im = None
prev_state = None
prev_t = time.time()
while True:
    t = time.time() - t0
    im = get_image()
    state = get_state()

    if prev_im is None:
        ims = np.expand_dims(np.stack((im, im)), axis=0)
        states = np.expand_dims(np.stack((state, state)), axis=0)
    else:
        ims = np.expand_dims(np.stack((prev_im, im)), axis=0)
        states = np.expand_dims(np.stack((prev_state, state)), axis=0)

    pad_mask = np.array([[False if prev_state is None else True, True]])
    # pad_mask = np.array([[False, True]])
    timestep = np.array([[prev_t, t]])

    observations = {
        "image_primary": ims,
        "proprio": states,
        "pad_mask": pad_mask,
        "timestep" : timestep
    }

    start = time.time()
    actions = model.sample_actions(observations, task, rng=jax.random.PRNGKey(0))[0]
    print("Sampling actions took ", time.time() - start, " seconds")

    prev_state = state
    prev_im = im
    prev_t = t
    
    # Unnormalize
    actions = (
        actions * model.dataset_statistics["action"]["std"]
        + model.dataset_statistics["action"]["mean"]
    )

    # set_joints(np.array(actions[0]))
    # time.sleep(0.01)

    for i, step in enumerate(actions):
        set_joints(np.array(step))
        time.sleep(0.01)

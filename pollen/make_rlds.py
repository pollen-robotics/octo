import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np


# step = {}
# # Required
# step["is_first"] = True
# step["is_last"] = False

# # Optional
# step["observation"] = None
# step["action"] = None
# # step["reward"] = 0
# # step["is_terminal"] = False # Terminal step of the dataset
# # step["discount"] = 0


# dataset = tf.data.Dataset

nb_episodes = 5
nb_steps_per_episode = 10

episodes = []
for episode_id in range(nb_episodes):
    episode = None

    for i in range(nb_steps_per_episode):
        step = {}

        # 'is_first': tfds.features.Scalar(
        #     dtype=np.bool_,
        #     doc='True on first step of the episode.'
        # ),
        step["is_first"] = i==0
        # step["is_first"] = tfds.features.Scalar(i==0, dtype=np.bool_)
        step["is_last"] = i==(nb_steps_per_episode-1)
        # step["is_last"] = tfds.features.Scalar(i==(nb_steps_per_episode-1), dtype=np.bool_)
        step["observation"] = {}
        step["observation"]["state"] = np.zeros(19, dtype=np.float32)
        step["observation"]["image1"] = np.zeros((640, 480, 3), dtype=np.float32)

        # step["observation"] = tfds.features.FeaturesDict({
        #     "state" : tfds.features.Tensor(np.zeros(19), dtype=np.float32),
        #     "image1" : tfds.features.Image(np.zeros((640, 480, 3)), dtype=np.uint8)
        # })

        step["action"] = np.zeros(19, dtype=np.float32)
        # step["action"] = tfds.features.Tensor(np.zeros(19), dtype=np.float32)
        tf.data.Dataset()
        episode.append(step)

    

    episodes.append(
        tfds.features.Dataset({
            "steps" : episode,
            "episode_metadata":{}
        })
    )
    # episodes.append({
    #     "steps" : episode,
    #     "episode_metadata":{}
    # })

# print(episodes[0])







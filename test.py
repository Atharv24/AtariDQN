from tensorflow import keras
import gym
import atari_wrappers
import numpy as np

TEST_GAMES = 5

ENV_NAME = 'PongNoFrameskip-v4'
env = atari_wrappers.wrap_deepmind(gym.make('PongNoFrameskip-v4'))
env = gym.wrappers.Monitor(env, "test", force=True, video_callable = lambda episode_id : True)

obs = env.reset()
model = keras.models.load_model("saved_models/model_games_done_50.h5", compile=False)

games_done = 0

while games_done < TEST_GAMES:
    state = np.array([obs])
    q_values = model(state, training=False)
    action = np.argmax(q_values, axis=-1)
    print(action)

    new_obs, reward, done, _ = env.step(action)
    obs = new_obs
    if done:
        games_done +=1
        print("Games Done: {}".format(games_done))
        obs = env.reset()
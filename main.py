from tensorflow import summary
import gym
import atari_wrappers
from agent import DQNAgent

SAVE_VIDEO = True
ENV_NAME = 'PongNoFrameskip-v4'

hyperparameters = {
    'buffer_start_size': 10001,
	'buffer_size': 15000,
	'epsilon': 1.0,
	'decay': 1-5e-6,
	'epsilon_final': 0.02,
	'learning_rate': 5e-5,
	'gamma': 0.99,
	'iter_update_target': 1000
}

MAX_GAMES = 3000
BATCH_SIZE = 32

env = atari_wrappers.wrap_deepmind(gym.make('PongNoFrameskip-v4'))
if SAVE_VIDEO:
    env = gym.wrappers.Monitor(env, "main-"+ENV_NAME, force=True, video_callable=lambda episode_id: episode_id%20==0)

obs = env.reset()

agent = DQNAgent(hyperparameters)

games_done = agent.games_done

writer = summary.create_file_writer("logs")

with writer.as_default():
    while games_done < MAX_GAMES:
        action = agent.act_eps_greedy(obs)
        new_obs, reward, done, _ = env.step(action)
        agent.store_memory(obs, action, reward, new_obs, done)
        agent.sample_and_optimize(BATCH_SIZE)
        obs = new_obs
        if done:
            games_done, reward, mean_reward, epsilon, loss = agent.get_info()

            summary.scalar("reward", reward, games_done)
            summary.scalar("mean_reward", mean_reward, games_done)
            summary.scalar("loss", loss, games_done)
            summary.scalar("epsilon", epsilon, games_done)
            writer.flush()

            agent.reset()
            obs = env.reset()
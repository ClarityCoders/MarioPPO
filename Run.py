import retro
import gym
from RandomAgent import TimeLimitWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv


model = PPO.load("tmp/best_model.zip")

def main():
    steps = 0
    #env = retro.make(game='MegaMan2-Nes')
    env = retro.make(game='SuperMarioBros-Nes')
    env = TimeLimitWrapper(env)
    env = MaxAndSkipEnv(env, 4)

    obs = env.reset()
    done = False

    while not done:
        action, state = model.predict(obs)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()
        steps += 1
        if steps % 1000 == 0:
            print(f"Total Steps: {steps}")
            print(info)

    print("Final Info")
    print(info)
    env.close()


if __name__ == "__main__":
    main()
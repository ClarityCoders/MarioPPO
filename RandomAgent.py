import retro
import gym
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv

class TimeLimitWrapper(gym.Wrapper):
  """
  :param env: (gym.Env) Gym environment that will be wrapped
  :param max_steps: (int) Max number of steps per episode
  """
  def __init__(self, env, max_steps=10000):
    # Call the parent constructor, so we can access self.env later
    super(TimeLimitWrapper, self).__init__(env)
    self.max_steps = max_steps
    # Counter of steps per episode
    self.current_step = 0
  
  def reset(self):
    """
    Reset the environment 
    """
    # Reset the counter
    self.current_step = 0
    return self.env.reset()

  def step(self, action):
    """
    :param action: ([float] or int) Action taken by the agent
    :return: (np.ndarray, float, bool, dict) observation, reward, is the episode over?, additional informations
    """
    self.current_step += 1
    obs, reward, done, info = self.env.step(action)
    # Overwrite the done signal when 
    if self.current_step >= self.max_steps:
      done = True
      # Update the info dict to signal that the limit was exceeded
      info['time_limit_reached'] = True
    info['Current_Step'] = self.current_step
    return obs, reward, done, info

def main():
    steps = 0
    #env = retro.make(game='MegaMan2-Nes')
    env = retro.make(game='SuperMarioBros-Nes')
    env = TimeLimitWrapper(env)
    env = MaxAndSkipEnv(env, 4)
    
    obs = env.reset()
    print(obs.shape)
    done = False
    while not done:
        obs, rew, done, info = env.step(env.action_space.sample())
        env.render()
        if done:
            obs = env.reset()
        steps += 1
        #print(rew)
        if steps % 1000 == 0:
            print(f"Total Steps: {steps}") 
            print(info)

    print("Final Info")
    print(info)
    env.close()


if __name__ == "__main__":
    main()
import gym, cv2, io
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict


class SinusoidLaneEnv(gym.Env):
    def __init__(self):
        self.__omega = .5
        self.__amplitude = 2.0
        self.__lb = 0
        self.__ub = 10
        self.__n = 150
        self.__lane, self.__x = self._make_lane()
        self.__left_bound = self.__lane - 1
        self.__right_bound = self.__lane + 1
        self.__lane_pad = 1
        self.__counter = 0
        self.__end_idx = -5
        self.__dx = (self.__ub - self.__lb)/(self.__n - 1)
        
        self.action_space = gym.spaces.Box(low=-0.5, high=0.5, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-(self.__amplitude+1), high=self.__amplitude+1,shape=(4,), dtype=np.float32
        )
        self._max_episode_steps = self.__n
        self.terminate = False
        self.truncate = False
        self.state = np.array([-1, 0, 1, self.__x[self.__counter]], dtype=np.float32)
        
    def _make_lane(self) -> Tuple[np.ndarray, np.ndarray]:
        x = np.linspace(self.__lb, self.__ub, self.__n)
        lane = self.__amplitude * np.sin(self.__omega * x)
        return lane, x
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, float]]:
        if isinstance(action, float):
            action = np.array([action])

        pos = self.state[1] + (action[0] * self.__dx)
        # pos = (
        #     self.state[1] + 
        #     (self.__amplitude * self.__omega * np.cos(self.__omega * self.state[3]) * self.__dx)
        # )
        self.state = np.array([
            pos - self.__lane_pad,
            pos,
            pos + self.__lane_pad,
            self.state[3] + self.__dx,
        ], dtype=np.float32)

        self.__counter += 1
        reward = 1 - np.abs(self.state[1] - self.__lane[self.__counter])

        if self.__counter >= self._max_episode_steps:
            self.truncate = True

        if self.state[3] >= self.__x[self.__end_idx]:
            self.terminate = True

        if self.state[1] < self.observation_space.low[1] or self.state[1] > self.observation_space.high[1]:
            self.terminate = True

        info =  {
            "left_pos": self.state[0],
            "center_pos": self.state[1],
            "right_pos": self.state[2],
            "x_pos": self.state[3],
        }
        return self.state, reward, self.truncate, self.terminate, info

    def reset(self) -> Tuple[np.ndarray, Dict[str, float]]:
        self.__counter = 0
        self.state = np.array([-1, 0, 1, self.__x[self.__counter]], dtype=np.float32)
        self.truncate = False 
        self.terminate =  False
        info =  {
            "left_pos": self.state[0],
            "center_pos": self.state[1],
            "right_pos": self.state[2],
        }
        return self.state, info

    def render(self):
        if not (hasattr(self, "state")):
            raise Exception("run 'step(...)' or reset(...) first")
        
        if self.terminate or self.truncate:
            return

        buffer = io.BytesIO()
        fig = plt.figure(figsize=(20, 7))
        plt.xlim(self.__lb-0.5, self.__ub+0.5)
        plt.ylim(-3.5, 3.5)
        plt.autoscale(False)
        plt.xticks(visible=False)
        #plt.yticks(visible=False)
        plt.plot(self.__x, self.__left_bound, c="orange", linewidth=3)
        plt.plot(self.__x, self.__right_bound, c="orange", linewidth=3)
        plt.plot(
            [self.state[3], self.state[3], self.state[3], self.state[3] + (7 * self.__dx), self.state[3]],
            [self.state[0], self.state[1], self.state[2], self.state[1], self.state[0]],
            c="black"
        )
        plt.scatter(
            [self.state[3], self.state[3], self.state[3], self.state[3] + (7 * self.__dx), self.state[3]],
            [self.state[0], self.state[1], self.state[2], self.state[1], self.state[0]], 
            c="orange"
        )
        plt.plot(
            [self.__x[self.__end_idx], self.__x[self.__end_idx]],
            [self.__left_bound[self.__end_idx], self.__right_bound[self.__end_idx]],
            c="green"
        )
        fig.savefig(buffer, format="png")
        plt.close(fig)
        img = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (1000, 300))
        cv2.imshow(self.__class__.__name__, img)
        cv2.waitKey(1)
        buffer.seek(0)

    def close(self):
        cv2.destroyAllWindows()
        

if __name__ == "__main__":
    env = SinusoidLaneEnv()
    env.reset()
    total_reward = 0
    for i in range(env._max_episode_steps):
        state, reward, truncate, terminate, info = env.step(env.action_space.sample())
        total_reward += reward
        env.render()
        if truncate or terminate:
            break
    print(f"Total Reward: {total_reward}")

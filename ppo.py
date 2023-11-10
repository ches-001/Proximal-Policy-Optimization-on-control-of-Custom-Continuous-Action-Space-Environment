import IPython
if IPython.get_ipython() is not None:
    from tqdm import tqdm
else:
    from tqdm.notebook import tqdm
import gym, torch, tqdm, os, copy
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal
from typing import Tuple, Iterable, Dict, Union, Optional, Any

os.environ["DEVICE"] = "cuda" if torch.cuda.is_available() else "cpu"

class PPOTrainer:
    def __init__(
            self, 
            policy_agent: Any, 
            value_agent: Any,
            policy_optimizer: torch.optim.Optimizer,
            value_optimzier: torch.optim.Optimizer,
            *,
            epsilon: float=0.2, 
            entropy_coef: float=0.01,
            n_policy_train_steps: int=100,
            n_value_train_steps: int=100,
            sparse_penalty_coef: Optional[float]=None,
            maximize_entropy: bool=True,
            clip_policy_grads: bool=False,
            clip_value_grads: bool=False,
        ):
        
        self.policy_agent = policy_agent
        self.policy_agent.to(os.environ["DEVICE"])
        self.value_agent = value_agent
        self.value_agent.to(os.environ["DEVICE"])
        self.policy_optimizer = policy_optimizer
        self.value_optimizer = value_optimzier
        self.epsilon = epsilon
        self.entropy_coef = entropy_coef
        self.n_policy_train_steps = n_policy_train_steps
        self.n_value_train_steps = n_value_train_steps
        self.sparse_penalty_coef = sparse_penalty_coef
        self.maximize_entropy = maximize_entropy
        self.clip_policy_grads = clip_policy_grads
        self.clip_value_grads = clip_value_grads


    def train_policy_net(
            self, 
            obs: torch.Tensor, 
            actions: torch.Tensor, 
            old_log_probs: torch.Tensor, 
            gaes: torch.Tensor):
        
        obs = obs.to(os.environ["DEVICE"] )
        actions = actions.to(os.environ["DEVICE"] )
        old_log_probs = old_log_probs.to(os.environ["DEVICE"] )
        gaes = gaes.to(os.environ["DEVICE"])

        for _ in range(self.n_policy_train_steps):
            self.policy_optimizer.zero_grad()

            mu, covar = self.policy_agent(obs)
            action_dist = MultivariateNormal(mu, covar)
            new_log_probs = action_dist.log_prob(actions)
            policy_ratio = torch.exp(new_log_probs - old_log_probs)
            clipped_policy_ratio = (
                torch
                .exp(new_log_probs - old_log_probs)
                .clamp(1-self.epsilon, 1+self.epsilon)
            )
            policy_loss = -torch.min(policy_ratio * gaes, clipped_policy_ratio * gaes)

            if self.maximize_entropy:
                policy_loss = policy_loss - (self.entropy_coef * action_dist.entropy())
            else:
                policy_loss = policy_loss + (self.entropy_coef * action_dist.entropy())
                
            policy_loss = policy_loss.mean()    
            policy_loss.backward()

            if self.clip_policy_grads:
                for param in self.policy_agent.parameters():
                    param.grad.data.clamp(-1.0, 1.0)

            self.policy_optimizer.step()


    def train_value_net(self, obs: torch.Tensor, returns: torch.Tensor):
        obs = obs.to(os.environ["DEVICE"])
        returns = returns.to(os.environ["DEVICE"])

        for _ in range(self.n_value_train_steps):
            self.value_optimizer.zero_grad()
            values = self.value_agent(obs)
            loss = (values - returns) ** 2
            loss = loss.mean()
            loss.backward()

            if self.clip_value_grads:
                for param in self.value_agent.parameters():
                    param.grad.data.clamp(-1.0, 1.0)
            self.value_optimizer.step()


    def save_policy(self, filename:str, dir:str="params"):
        if not os.path.isdir(dir): os.makedirs(dir, exist_ok=True)
        torch.save(self.policy_agent.state_dict(), os.path.join(dir, filename))


def discount_reward(rewards: np.ndarray, gamma: float=0.99) -> np.ndarray:
    # n_Σ_(k=t) (r_(t + k) * γ ^ (k))
    discounted_rewards = []
    discount_t = 0
    # Start from the last reward and work backward
    for r in reversed(rewards):
        discount_t = discount_t * gamma + r
        discounted_rewards.insert(0, discount_t)
    discounted_rewards = np.array(discounted_rewards)
    return discounted_rewards


def compute_gaes(
        rewards: np.ndarray, 
        values: np.ndarray, 
        gamma: float=0.99, 
        lambda_: float=0.97) -> np.ndarray:
    
    next_values = np.concatenate((values[1:], np.zeros(1)))
    deltas = rewards + (gamma * (next_values - values))
    gaes = []
    gae_t = 0
    # Start from the last reward and work backward
    for delta in reversed(deltas):
        gae_t = delta + (gamma * lambda_ * gae_t)
        gaes.insert(0, gae_t)
    gaes = np.array(gaes)
    return gaes


def rollout(
        env: gym.Env, 
        policy_agent: Any,
        value_agent: Any,
        max_rollout_steps: Optional[int]=None, 
    ) -> Tuple[Dict[str, np.ndarray], float]:

    experience = {
        "observations": [],
        "actions": [],
        "rewards": [],
        "values": [],
        "act_log_probs": [],
    }

    obs, _ = env.reset()
    eps_reward = 0
    max_rollout_steps = max_rollout_steps or env._max_episode_steps
    
    for _ in range(0, max_rollout_steps):
        _obs = torch.from_numpy(obs).unsqueeze(0).to(dtype=torch.float32, device=os.environ["DEVICE"])

        mu, covar = policy_agent(_obs)
        value = value_agent(_obs)

        action_dist = MultivariateNormal(mu, covar)
        action = action_dist.sample()
        action_log_prob = action_dist.log_prob(action).cpu().detach().numpy()
        action = action.cpu().detach().squeeze(0).numpy()
        value = value.cpu().item()

        new_obs, reward, terminated, truncated, _ = env.step(action)
        
        experience["observations"].append(obs)
        experience["actions"].append(action)
        experience["rewards"].append(reward)
        experience["values"].append(value)
        experience["act_log_probs"].append(action_log_prob)

        eps_reward += reward
        obs = new_obs

        if truncated or terminated: break

    experience = {k:np.asarray(v) for k, v in experience.items()}
    return experience, eps_reward


def format_policy_data(data: Dict[str, torch.Tensor]):
    data = copy.deepcopy(data)
    data["observations"] = torch.tensor(data["observations"], dtype=torch.float32, device=os.environ["DEVICE"])
    data["actions"] = torch.tensor(data["actions"], dtype=torch.int32, device=os.environ["DEVICE"])
    data["rewards"] = torch.tensor(data["rewards"], dtype=torch.float32, device=os.environ["DEVICE"])
    data["values"] = torch.tensor(data["values"], dtype=torch.float32, device=os.environ["DEVICE"])
    data["returns"] = torch.tensor(data["returns"], dtype=torch.float32, device=os.environ["DEVICE"])
    data["gaes"] = torch.tensor(data["gaes"], dtype=torch.float32, device=os.environ["DEVICE"])
    data["act_log_probs"] = torch.tensor(data["act_log_probs"], dtype=torch.float32, device=os.environ["DEVICE"])
    return data


def train(
        env: gym.Env, 
        ppo: PPOTrainer, 
        *,
        n_episodes: int=400, 
        gamma: float=0.99, 
        lambda_: float=0.97, 
        max_episode_reward: Union[int, float]=200,
        max_rollout_steps: Optional[int]=None,
        continue_after_max_reward: bool=False,
        policy_weights_filename: Optional[str]="CartPolev0_PPO_policy_agent.pth.tar",
        save_best_weights: bool=True,
        verbose: bool=False,
        close_env: bool=False,
        normalize_returns: bool=False,
        normalize_gaes: bool=False,
        scale_obs: bool=False,
        obs_low: Optional[Union[int, np.ndarray, torch.Tensor]]=None,
        obs_high: Optional[Union[int, np.ndarray, torch.Tensor]]=None,
    ) -> Tuple[Dict[str, Iterable], Dict[str, Iterable]]:

    train_performance = {
        'rewards':[],
    }

    best_rewards = {
        'episodes': [],
        'rewards': []
    }

    if scale_obs:
        obs_low = obs_low or env.observation_space.low
        obs_high = obs_high or env.observation_space.high

        obs_low = obs_low if not isinstance(obs_low, torch.Tensor) else obs_low.numpy()
        obs_high = obs_high if not isinstance(obs_high, torch.Tensor) else obs_high.numpy()
    else:
        obs_low = 0
        obs_high = 1

    best_reward = -np.inf
    for episode in tqdm.tqdm(range(0, n_episodes)):
        data, episode_reward = rollout(
            env, 
            ppo.policy_agent, 
            ppo.value_agent, 
            max_rollout_steps=max_rollout_steps
        )

        data["returns"] = discount_reward(data["rewards"], gamma)
        data["gaes"] = compute_gaes(data["rewards"], data["values"], gamma, lambda_)

        if normalize_returns:
            data["returns"] = (data["returns"] - data["returns"].mean()) / data["returns"].std()
        
        if normalize_gaes:
            data["gaes"] = (data["gaes"] - data["gaes"].mean()) / data["gaes"].std()

        data["observations"] = (data["observations"] - obs_low) / (obs_high - obs_low)

        data = format_policy_data(data)
        ppo.train_policy_net(data["observations"], data["actions"], data["act_log_probs"], data["gaes"])
        ppo.train_value_net(data["observations"], data["returns"])

        train_performance["rewards"].append(episode_reward)

        if episode_reward > best_reward:
            best_reward = episode_reward
            if verbose:
                print(f'best total reward for episode {episode+1}: {episode_reward}')
            best_rewards["episodes"].append(episode)
            best_rewards["rewards"].append(best_reward)

            if save_best_weights:
                ppo.save_policy(policy_weights_filename)

        if not continue_after_max_reward:
            if best_reward >= max_episode_reward: break

    if close_env:
        env.close()
    return train_performance, best_rewards


def evaluate(
        env: gym.Env, 
        policy_agent: Any,
        max_episode_steps: int=1000,
        render_env: bool=True,
        close_env: bool=False,
    ) -> Tuple[Union[int, float], float, float]:

    policy_agent.eval()
    obs, _ = env.reset()

    total_reward = 0
    avg_policy_entropy = 0
    for _ in range(max_episode_steps):
        _obs = torch.from_numpy(obs).unsqueeze(0).to(dtype=torch.float32, device=os.environ["DEVICE"])

        with torch.no_grad():
            mu, covar = policy_agent(_obs)

        action_dist = MultivariateNormal(mu, covar)
        action = mu.cpu().squeeze(0).numpy() #action_dist.sample().cpu().squeeze(0).numpy()
        try:
            obs, reward, terminate, truncated, _ = env.step(action)
        except OverflowError:
            break

        if render_env:
            env.render()
            
        policy_entropy = action_dist.entropy()
        avg_policy_entropy += policy_entropy.item()

        total_reward += reward
        if truncated or terminate: break

    avg_policy_entropy /= max_episode_steps
    if close_env:
        env.close()
    return total_reward, avg_policy_entropy
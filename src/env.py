from tqdm import tqdm
from agent import TrafficAgent
import yaml
import gymnasium as gym
import sumo_rl
import os
import cv2

class TrafficEnv(): 
    def __init__(self, config: str, net: str, route: str, weights: str = None):
        with open(config, 'r') as f: 
            self.config = yaml.safe_load(f)
        
        self.env = gym.make("sumo-rl-v0",
                            net_file=net,
                            route_file=route,
                            use_gui=False,
                            num_seconds=1000,
                            min_green=0,
                            max_depart_delay=0,
                           additional_sumo_cmd="--no-step-log")

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.net = net 
        self.route = route
        
        self.agent = TrafficAgent(
            obs_dim=self.observation_space.shape[0],
            ac_dim=self.action_space.n,
            model=self.config["model"], 
            lr=self.config["lr"], 
            gamma=self.config["gamma"], 
            max_memory=self.config["max_memory"], 
            max_gradient=self.config["max_grad"]
        )
        
        if weights: 
            try:
                self.agent.load_weights(weights)
            except Exception as e:
                print(f"[INFO] Weights not found, training from scratch.")
        
        self.epsilon = self.config["epsilon"]
        self.epsilon_min = self.config["epsilon_min"]
        self.epsilon_decay = self.config["epsilon_decay"]
        
        self.episodes = self.config["episodes"]
        self.batch_size = self.config["batch_size"]
        self.update_freq = self.config["update_freq"]
        
    def train(self, path: str):
        os.makedirs(path, exist_ok=True)
        pbar = tqdm(range(self.episodes), desc="Episode")
        
        for i in pbar:
            state, _ = self.env.reset()
            done = False
            total_reward = 0.0
            total_loss = 0.0
            step = 0
            
            epsilon = self.epsilon if i == 0 else max(self.epsilon * self.epsilon_decay, self.epsilon_min)
            self.epsilon = epsilon
            
            while not done:
                action = self.agent.select_action(state, epsilon)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                self.agent.push(state, action, reward, next_state, done)
                
                total_reward += reward
                state = next_state

                if len(self.agent.buffer) > self.batch_size:
                    total_loss += self.agent.update(self.batch_size)
                
                step += 1
                
                if step % self.update_freq == 0:
                    self.agent.update_target_network(False)
                
            avg_loss = total_loss / step if step > 0 else 0.0
            pbar.set_postfix(reward=total_reward, loss=avg_loss)
            
            # To clear 
            os.system('cls' if os.name == 'nt' else 'clear')
        
        self.agent.save_weights(os.path.join(path, "shared_policy.pth"))
            
    def test(self, path: str):
        os.makedirs(path, exist_ok=True)
        FPS = 20.0
        
        env_vis = gym.make(
            "sumo-rl-v0",
            net_file=self.net,
            route_file=self.route,
            use_gui=True,
            render_mode="rgb_array",
            num_seconds=1000,
            min_green=0,
            max_depart_delay=0,
            additional_sumo_cmd="--no-step-log"
        )
        env_vis.metadata['render_fps'] = FPS

        self.agent.load_weights(os.path.join(path, "shared_policy.pth"))

        state, _ = env_vis.reset()
        frame = env_vis.render()
        height, width, _ = frame.shape

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_path = os.path.join(path, "traffic_sim.mp4")
        video = cv2.VideoWriter(video_path, fourcc, FPS, (width, height))

        done = False
        total_reward = 0.0

        while not done:
            frame = env_vis.render()
            video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            action = self.agent.select_action(state, epsilon=0.0)
            next_state, reward, terminated, truncated, _ = env_vis.step(action)
            done = terminated or truncated

            state = next_state
            total_reward += reward

        video.release()
        env_vis.close()

        print(f"✔ Video saved to {video_path}")
        print(f"✔ Test completed with total reward: {total_reward:.2f}")
        return total_reward
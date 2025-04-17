from tqdm import tqdm
from agent import TrafficAgent
from replay import ReplayBuffer

import yaml
import gym
import sumo_rl
import os

class TrafficEnv(): 
    def __init__(self, config: str, net: str, route: str, weights: str):
        with open(config, 'r') as f: 
            self.config = yaml.safe_load(f)
        
        self.env = gym.make("sumo-rl-v0",
                            net_file=net,
                            route_file=route,
                            use_gui=False,
                            num_seconds=1000,
                            min_green=5,
                            max_depart_delay=0)
        
        self.observation_spaces = self.env.observation_spaces
        self.action_spaces = self.env.action_spaces
        
        self.agents = {} 
        for intersection_id in self.observation_spaces.keys():
            self.agents[intersection_id] = TrafficAgent(
                obs_dim=self.observation_spaces[intersection_id].n, 
                ac_dim=self.action_spaces[intersection_id].n,
                model=self.config["model"], 
                lr=self.config["lr"], 
                gamma=self.config["gamma"], 
                max_memory=self.config["max_memory"], 
                max_gradient=self.config["max_grad"]
            )
            
            if weights: 
                try:
                    self.agents[intersection_id].load_weights(os.path.join(weights, f"intersection_{intersection_id}.pth"))
                except Exception as e:
                    print(f"[INFO] Weights for model-{intersection_id} not found.")
        
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
            states = self.env.reset()
            done = {id: False for id in self.agents.keys()}
            total_reward = {id: 0.0 for id in self.agents.keys()}
            total_loss = {id: 0.0 for id in self.agents.keys()}
            
            step = 0
            
            if i == 0: 
                epsilon = self.epsilon
            else: 
                epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
                self.epsilon = epsilon
            
            while not all(done.values()):
                actions = {}
                
                for intersection_id, agent in self.agents.items(): 
                    if not done[intersection_id]:
                        actions[intersection_id] = agent.select_action(states[intersection_id], epsilon)
                
                next_states, rewards, next_done, _ = self.env.step(actions)
                
                for intersection_id, agent in self.agents.items(): 
                    if not next_done[intersection_id]: 
                        self.agents.push(
                            states[intersection_id], 
                            actions[intersection_id],
                            rewards[intersection_id],
                            next_states[intersection_id], 
                            next_done[intersection_id],
                        )
                        
                        total_reward[intersection_id] += rewards[intersection_id]
                        
                states = next_states
                done = next_done
            
                for intersection_id, agent in self.agents.items():
                    if len(agent.buffer) > self.batch_size: 
                        total_loss[intersection_id] += agent.update(self.batch_size)
                        
                step+=1
                
                if step % self.update_freq == 0:
                    for agent in self.agents.values():
                        agent.update_target_network(False)
                
            avg_reward = sum(total_reward.values()) / len(total_reward)
            avg_loss = sum(total_loss.values()) / len(total_loss)
            pbar.set_postfix(reward=avg_reward, loss=avg_loss)
        
        for intersection_id, agent in self.agents.items():
            agent.save_weights(os.path.join(path, f"intersection_{intersection_id}_model.pth"))
            
    def test(self, path: str):
        os.makedirs(path, exist_ok=True)
        pass
            
        

    
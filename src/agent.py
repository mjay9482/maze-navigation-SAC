import os 
import torch 
import torch.optim as optim 
from model import * 
import torch.nn.functional as F 
from torch.optim import Adam
from replay_buffer import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
import datetime
import json
import numpy as np

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def soft_upadte(target, source, tau):
    for target_param , param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

class Agent:
    def __init__(self, num_inputs, action_space, gamma, tau, alpha, target_update_interval,
                 hidden_size, learning_rate, exploration_scaling_factor):
        
        self.gamma = gamma
        self.tau = tau 
        self.alpha = alpha 
        self.target_update_interval = target_update_interval
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Running on {self.device}")
        
        self.critic = Critic(num_inputs, action_space.shape[0], hidden_size).to(device = self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr = learning_rate)
        
        self.critic_target = Critic(num_inputs, action_space.shape[0], hidden_size).to(device = self.device)
        hard_update(self.critic_target, self.critic)
        
        self.policy = Actor(num_inputs, action_space.shape[0], hidden_size, action_space).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr = learning_rate)
        
        #initialize the predictive model 
        
        self.predictive_model = PredictiveModel(num_inputs, action_space.shape[0], hidden_size).to(self.device)
        self.predictive_model_optim = Adam(self.policy.parameters(), lr=learning_rate)
        self.exploration_scaling_factor = Config['exploration_scaling_factor']
        
    def select_action(self, state, evaluate =False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        
        return action.detach().cpu().numpy()[0]
    
    def update_parameters(self, memory, batch_size, updates): 
        
        state_batch, action_batch , reward_batch, next_state_batch, mask_batch = memory.sample_buffer(batch_size = batch_size)
        
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)
        
        #predictive model 
        predicted_next_state = self.predictive_model(state_batch, action_batch)
        prediction_error = F.mse_loss(predicted_next_state, next_state_batch)
        prediction_error_no_reduction = F.mse_loss(predicted_next_state, next_state_batch, reduce=False)
        
        scaled_intrinsic_reward = prediction_error_no_reduction.mean(dim=1)
        scaled_intrinsic_reward = self.exploration_scaling_factor * torch.reshape(scaled_intrinsic_reward, (batch_size, 1))
        
        reward_batch = reward_batch + scaled_intrinsic_reward

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy_sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf1_next_target = torch.min(qf1_next_target, qf1_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * min_qf1_next_target
        
        qf1, qf2 = self.critic(state_batch, action_batch)
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss
        
        # Update critic network
        self.critic_optim.zero_grad() 
        qf_loss.backward()
        self.critic_optim.step()
        
        # Update predictive network
        self.predictive_model_optim.zero_grad()
        prediction_error.backward()
        self.predictive_model_optim.step()
        
        pi, log_pi, _ = self.policy.sample(state_batch)
        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)   
        
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()  
        
        # Update policy network
        self.policy_optim.zero_grad() 
        policy_loss.backward()
        self.policy_optim.step()
        
        alpha_loss = torch.tensor(0.).to(self.device)      
        alpha_logs = torch.tensor(self.aplha)
        
        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)
        
        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), prediction_error.item(),  alpha_logs_loss.item()
    
    def train(self, env, env_name, memory, episodes=1000, batch_size=64, updates_per_step=1, summary_writer_name="", max_episode_steps=100):
    
        warmup = Config["warmup"]
        
        # Create logs directory if it doesn't exist
        log_dir = 'logs'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        # Create a unique run ID
        run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # Setup TensorBoard
        summary_writer_name = f'runs/{run_id}_{summary_writer_name}'
        writer = SummaryWriter(summary_writer_name)
        
        # Initialize metrics storage
        metrics = {
            'episode_rewards': [],
            'episode_steps': [],
            'critic_losses': [],
            'policy_losses': [],
            'entropy_losses': [],
            'prediction_losses': [],
            'alpha_values': [],
            'success_rate': []
        }
        
        #training loop
        total_numsteps = 0
        updates = 0 
        successful_episodes = 0
        
        for episode in range(episodes):
            episode_reward = 0
            episode_steps = 0
            done = False 
            state,_ = env.reset() 
            
            while not done and episode_steps < max_episode_steps:
                if warmup > episode:
                    action = env.action_space.sample()
                else:
                    action = self.select_action(state)
                
                if memory.can_sample(batch_size = batch_size):
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, prediction_loss, alpha = self.update_parameters(memory, batch_size, updates)
                    
                    # Store losses
                    metrics['critic_losses'].append((critic_1_loss + critic_2_loss) / 2)
                    metrics['policy_losses'].append(policy_loss)
                    metrics['entropy_losses'].append(ent_loss)
                    metrics['prediction_losses'].append(prediction_loss)
                    metrics['alpha_values'].append(alpha)
                    
                    # Log to TensorBoard
                    writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                    writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                    writer.add_scalar('loss/policy', policy_loss, updates)
                    writer.add_scalar('loss/entropy', ent_loss, updates)
                    writer.add_scalar('loss/prediction_loss', prediction_loss, updates)
                    writer.add_scalar('parameters/alpha', alpha, updates)
                    updates += 1

                next_state, reward, done, _, info = env.step(action)
                
                episode_steps += 1
                total_numsteps += 1
                episode_reward += reward 
                
                mask = 1 if episode_steps == max_episode_steps else float(not done)
                state = next_state
            
            # Store episode metrics
            metrics['episode_rewards'].append(episode_reward)
            metrics['episode_steps'].append(episode_steps)
            
            # Check if episode was successful (assuming reward of 1.0 indicates success)
            if episode_reward >= 1.0:
                successful_episodes += 1
            metrics['success_rate'].append(successful_episodes / (episode + 1))
            
            # Log episode metrics to TensorBoard
            writer.add_scalar('reward/train', episode_reward, episode)
            writer.add_scalar('steps/episode', episode_steps, episode)
            writer.add_scalar('success_rate', metrics['success_rate'][-1], episode)
            
            # Save metrics to file every 10 episodes
            if episode % 10 == 0:
                metrics_file = os.path.join(log_dir, f'metrics_{run_id}.json')
                with open(metrics_file, 'w') as f:
                    json.dump(metrics, f)
                
                # Save model checkpoint
                self.save_checkpoint()
                
                # Print progress
                print(f"Episode: {episode}")
                print(f"Total steps: {total_numsteps}")
                print(f"Episode steps: {episode_steps}")
                print(f"Episode reward: {round(episode_reward, 2)}")
                print(f"Success rate: {round(metrics['success_rate'][-1] * 100, 2)}%")
                print(f"Average reward (last 10): {round(np.mean(metrics['episode_rewards'][-10:]), 2)}")
                print("----------------------------------------")
        
        # Save final metrics
        metrics_file = os.path.join(log_dir, f'metrics_{run_id}_final.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f)
            
        writer.close()
        
        return metrics
    
    def test(self, env, episodes=10, max_episode_steps=500):
        
        for episode in range(episodes):
            episode_reward=0
            episode_steps=0
            done=False 
            state,_  = env.reset() 
            
            while not done and episode_steps < max_episode_steps:
                action = self.select_action(state)
                
                next_state, reward , done, _, _ = env.step(action)
                
                episode_steps += 1
                
                if reward == 1:
                    done = True 
                
                episode_reward += reward 
                state = next_state
            
            print(f"Episode: {episode}, episode_steps: {episode_steps}, reward :{round(episode_reward,2)}")

                
    def save_checkpoint(self, evaluate=False):
        if not os.path.exists('checkpoints/'):
            os.makedirs('checkpoints/')
        
        print("Saving Models")
        self.policy.save_checkpoint()
        self.critic_target.save_checkpoint()
        self.critic.save_checkpoint()
    
    def load_checkpoint(self, evaluate=False):
        try:
            print("Loading Models")
            self.policy.load_checkpoint()
            self.critic_target.load_checkpoint()
            self.critic.load_checkpoint()
        except:
            if (evaluate):
                raise Exception("Unable to evaluate models without a loaded checkpoint")
            else:
                print("Unable to load models, starting from again")
        
        if evaluate:
            self.policy.eval()
            self.critic.eval()
            self.critic_target.eval()
        else:
            self.policy.train() 
            self.critic.train() 
            
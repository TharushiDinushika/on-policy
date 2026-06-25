    
import time
import wandb
import os
import numpy as np
from itertools import chain
import torch

from onpolicy.utils.util import update_linear_schedule
from onpolicy.runner.separated.base_runner import Runner
import imageio

def _t2n(x):
    return x.detach().cpu().numpy()

def compute_coverage_rate(obs_batch, num_agents, num_landmarks, coverage_threshold=0.1):
    lm_start = 4
    n_threads = obs_batch.shape[0]
    per_thread_coverage = []

    for t in range(n_threads):
        covered = 0
        for lm in range(num_landmarks):
            lm_x_idx = lm_start + lm * 2
            lm_y_idx = lm_start + lm * 2 + 1
            for agent_id in range(num_agents):
                obs = obs_batch[t, agent_id]
                if lm_y_idx >= len(obs):
                    break
                rel_x = obs[lm_x_idx]
                rel_y = obs[lm_y_idx]
                dist  = np.sqrt(rel_x ** 2 + rel_y ** 2)
                if dist < coverage_threshold:
                    covered += 1
                    break
        per_thread_coverage.append(covered / num_landmarks)

    return float(np.mean(per_thread_coverage))

def compute_min_landmark_distance(obs_batch, num_agents, num_landmarks):
    lm_start = 4
    n_threads = obs_batch.shape[0]
    per_thread_min_dist = []

    for t in range(n_threads):
        lm_min_dists = []
        for lm in range(num_landmarks):
            lm_x_idx = lm_start + lm * 2
            lm_y_idx = lm_start + lm * 2 + 1
            min_dist = float('inf')
            for agent_id in range(num_agents):
                obs = obs_batch[t, agent_id]
                if lm_y_idx >= len(obs):
                    break
                rel_x = obs[lm_x_idx]
                rel_y = obs[lm_y_idx]
                dist = np.sqrt(rel_x ** 2 + rel_y ** 2)
                if dist < min_dist:
                    min_dist = dist
            if min_dist != float('inf'):
                lm_min_dists.append(min_dist)
        
        if lm_min_dists:
            per_thread_min_dist.append(np.mean(lm_min_dists))
        else:
            per_thread_min_dist.append(0.0)

    return float(np.mean(per_thread_min_dist))

def compute_inter_agent_distance(obs_batch, num_agents, num_landmarks):
    if num_agents <= 1:
        return 0.0
    other_agent_start = 4 + 2 * num_landmarks
    n_threads = obs_batch.shape[0]
    per_thread_dist = []

    for t in range(n_threads):
        total_dist = 0
        pairs = 0
        for agent_id in range(num_agents):
            obs = obs_batch[t, agent_id]
            for other_idx in range(num_agents - 1):
                x_idx = other_agent_start + other_idx * 2
                y_idx = other_agent_start + other_idx * 2 + 1
                if y_idx >= len(obs):
                    break
                rel_x = obs[x_idx]
                rel_y = obs[y_idx]
                dist = np.sqrt(rel_x ** 2 + rel_y ** 2)
                total_dist += dist
                pairs += 1
        
        if pairs > 0:
            per_thread_dist.append(total_dist / pairs)
        else:
            per_thread_dist.append(0.0)

    return float(np.mean(per_thread_dist))

def compute_reward_metrics(reward_history):
    arr = np.array(reward_history)
    return {
        "mean":  float(np.mean(arr)),
        "std":   float(np.std(arr)),
        "min":   float(np.min(arr)),
        "max":   float(np.max(arr)),
    }

def steps_to_convergence(reward_log, threshold=0.9, smooth_window=10):
    if len(reward_log) < smooth_window:
        return len(reward_log)
    arr     = np.array(reward_log)
    kernel  = np.ones(smooth_window) / smooth_window
    smooth  = np.convolve(arr, kernel, mode='valid')
    peak    = smooth.max()
    target  = threshold * peak
    hits    = np.where(smooth >= target)[0]
    return int(hits[0]) if len(hits) > 0 else len(reward_log)

class MPERunner(Runner):
    def __init__(self, config):
        super(MPERunner, self).__init__(config)
        self._episode_reward_history   = []
        self._episode_coverage_history = []
       
    def run(self):

        if self.use_render:
            self.render()
            return

        self.warmup()   

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                for agent_id in range(self.num_agents):
                    self.trainer[agent_id].policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(step)
                    
                # Obser reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions_env)

                data = obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic 
                
                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute()
            train_infos = self.train()
            
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            
            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                        .format(self.all_args.scenario_name,
                                self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                total_num_steps,
                                self.num_env_steps,
                                int(total_num_steps / (end - start))))

                if self.env_name == "MPE":
                    for agent_id in range(self.num_agents):
                        idv_rews = []
                        for count, info in enumerate(infos):
                            if 'individual_reward' in infos[count][agent_id].keys():
                                idv_rews.append(infos[count][agent_id].get('individual_reward', 0))
                        train_infos[agent_id].update({'individual_rewards': np.mean(idv_rews)})
                        train_infos[agent_id].update({"average_episode_rewards": np.mean(self.buffer[agent_id].rewards) * self.episode_length})
                    
                    # compute average coverage rate over the entire episode
                    obs_batch = np.stack(
                        [self.buffer[aid].obs[:-1] for aid in range(self.num_agents)],
                        axis=2)
                    obs_batch = obs_batch.reshape(-1, self.num_agents, obs_batch.shape[-1])
                    num_landmarks = getattr(self.all_args, "num_landmarks", self.num_agents)
                    coverage = compute_coverage_rate(
                        obs_batch,
                        num_agents=self.num_agents,
                        num_landmarks=num_landmarks,
                    )
                    
                    final_obs_batch = np.stack(
                        [self.buffer[aid].obs[-2] for aid in range(self.num_agents)],
                        axis=1)
                    final_coverage = compute_coverage_rate(
                        final_obs_batch,
                        num_agents=self.num_agents,
                        num_landmarks=num_landmarks,
                    )
                    final_landmarks = final_coverage * num_landmarks

                    min_lm_dist = compute_min_landmark_distance(
                        obs_batch,
                        num_agents=self.num_agents,
                        num_landmarks=num_landmarks,
                    )
                    inter_agent_dist = compute_inter_agent_distance(
                        obs_batch,
                        num_agents=self.num_agents,
                        num_landmarks=num_landmarks,
                    )

                    for agent_id in range(self.num_agents):
                        train_infos[agent_id].update({
                            "landmark_coverage_rate": coverage,
                            "final_covered_landmarks": final_landmarks,
                            "min_landmark_distance": min_lm_dist,
                            "inter_agent_distance": inter_agent_dist,
                        })
                    
                    mean_ep_rew = np.mean([
                        train_infos[aid]["average_episode_rewards"]
                        for aid in range(self.num_agents)])
                    self._episode_reward_history.append(float(mean_ep_rew))
                    self._episode_coverage_history.append(float(coverage))

                    print("  [eval] coverage rate: {:.3f}  final landmarks: {:.2f}  min LM dist: {:.3f}  inter-agent dist: {:.3f}  mean ep reward: {:.3f}".format(coverage, final_landmarks, min_lm_dist, inter_agent_dist, mean_ep_rew))

                self.log_train(train_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

        self._print_convergence_summary()

    def _print_convergence_summary(self):
        if not self._episode_reward_history:
            return

        conv_ep = steps_to_convergence(self._episode_reward_history)
        stats   = compute_reward_metrics(self._episode_reward_history)
        jump    = float(np.mean(self._episode_reward_history[:10])) \
                  if len(self._episode_reward_history) >= 10 \
                  else float(np.mean(self._episode_reward_history))
        peak_cov = float(np.max(self._episode_coverage_history)) \
                   if self._episode_coverage_history else 0.0
        mean_cov = float(np.mean(self._episode_coverage_history)) \
                   if self._episode_coverage_history else 0.0

        print("\n" + "=" * 60)
        print("  TRAINING SUMMARY")
        print("=" * 60)
        print(f"  Peak reward          : {stats['max']:.3f}")
        print(f"  Mean reward (all)    : {stats['mean']:.3f} ± {stats['std']:.3f}")
        print(f"  Jump-start reward    : {jump:.3f}  (first 10 episodes)")
        print(f"  Steps to convergence : episode {conv_ep}  (90% of peak)")
        print(f"  Peak coverage rate   : {peak_cov:.3f}")
        print(f"  Mean coverage rate   : {mean_cov:.3f}")
        print("=" * 60 + "\n")

    def warmup(self):
        # reset env
        obs = self.envs.reset()

        share_obs = []
        for o in obs:
            share_obs.append(list(chain(*o)))
        share_obs = np.array(share_obs)

        for agent_id in range(self.num_agents):
            if not self.use_centralized_V:
                share_obs = np.array([obs[env_i][agent_id] for env_i in range(self.n_rollout_threads)])
            self.buffer[agent_id].share_obs[0] = share_obs.copy()
            self.buffer[agent_id].obs[0] = np.array([obs[env_i][agent_id] for env_i in range(self.n_rollout_threads)]).copy()

    @torch.no_grad()
    def collect(self, step):
        values = []
        actions = []
        temp_actions_env = []
        action_log_probs = []
        rnn_states = []
        rnn_states_critic = []

        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            value, action, action_log_prob, rnn_state, rnn_state_critic \
                = self.trainer[agent_id].policy.get_actions(self.buffer[agent_id].share_obs[step],
                                                            self.buffer[agent_id].obs[step],
                                                            self.buffer[agent_id].rnn_states[step],
                                                            self.buffer[agent_id].rnn_states_critic[step],
                                                            self.buffer[agent_id].masks[step])
            # [agents, envs, dim]
            values.append(_t2n(value))
            action = _t2n(action)
            # rearrange action
            if self.envs.action_space[agent_id].__class__.__name__ == 'MultiDiscrete':
                for i in range(self.envs.action_space[agent_id].shape):
                    uc_action_env = np.eye(self.envs.action_space[agent_id].high[i]+1)[action[:, i]]
                    if i == 0:
                        action_env = uc_action_env
                    else:
                        action_env = np.concatenate((action_env, uc_action_env), axis=1)
            elif self.envs.action_space[agent_id].__class__.__name__ == 'Discrete':
                action_env = np.squeeze(np.eye(self.envs.action_space[agent_id].n)[action], 1)
            else:
                raise NotImplementedError

            actions.append(action)
            temp_actions_env.append(action_env)
            action_log_probs.append(_t2n(action_log_prob))
            rnn_states.append(_t2n(rnn_state))
            rnn_states_critic.append( _t2n(rnn_state_critic))

        # [envs, agents, dim]
        actions_env = []
        for i in range(self.n_rollout_threads):
            one_hot_action_env = []
            for temp_action_env in temp_actions_env:
                one_hot_action_env.append(temp_action_env[i])
            actions_env.append(one_hot_action_env)

        values = np.array(values).transpose(1, 0, 2)
        actions = np.array(actions).transpose(1, 0, 2)
        action_log_probs = np.array(action_log_probs).transpose(1, 0, 2)
        rnn_states = np.array(rnn_states).transpose(1, 0, 2, 3)
        rnn_states_critic = np.array(rnn_states_critic).transpose(1, 0, 2, 3)

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env

    def insert(self, data):
        obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        share_obs = []
        for o in obs:
            share_obs.append(list(chain(*o)))
        share_obs = np.array(share_obs)

        for agent_id in range(self.num_agents):
            if not self.use_centralized_V:
                share_obs = np.array([obs[env_i][agent_id] for env_i in range(self.n_rollout_threads)])

            self.buffer[agent_id].insert(share_obs,
                                        np.array([obs[env_i][agent_id] for env_i in range(self.n_rollout_threads)]),
                                        rnn_states[:, agent_id],
                                        rnn_states_critic[:, agent_id],
                                        actions[:, agent_id],
                                        action_log_probs[:, agent_id],
                                        values[:, agent_id],
                                        rewards[:, agent_id],
                                        masks[:, agent_id])

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode_rewards = []
        eval_coverage_rates = []
        eval_min_lm_dists = []
        eval_inter_agent_dists = []
        eval_obs = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        for eval_step in range(self.episode_length):
            eval_temp_actions_env = []
            for agent_id in range(self.num_agents):
                self.trainer[agent_id].prep_rollout()
                eval_action, eval_rnn_state = self.trainer[agent_id].policy.act(np.array(list(eval_obs[:, agent_id])),
                                                                                eval_rnn_states[:, agent_id],
                                                                                eval_masks[:, agent_id],
                                                                                deterministic=True)

                eval_action = eval_action.detach().cpu().numpy()
                # rearrange action
                if self.eval_envs.action_space[agent_id].__class__.__name__ == 'MultiDiscrete':
                    for i in range(self.eval_envs.action_space[agent_id].shape):
                        eval_uc_action_env = np.eye(self.eval_envs.action_space[agent_id].high[i]+1)[eval_action[:, i]]
                        if i == 0:
                            eval_action_env = eval_uc_action_env
                        else:
                            eval_action_env = np.concatenate((eval_action_env, eval_uc_action_env), axis=1)
                elif self.eval_envs.action_space[agent_id].__class__.__name__ == 'Discrete':
                    eval_action_env = np.squeeze(np.eye(self.eval_envs.action_space[agent_id].n)[eval_action], 1)
                else:
                    raise NotImplementedError

                eval_temp_actions_env.append(eval_action_env)
                eval_rnn_states[:, agent_id] = _t2n(eval_rnn_state)
                
            # [envs, agents, dim]
            eval_actions_env = []
            for i in range(self.n_eval_rollout_threads):
                eval_one_hot_action_env = []
                for eval_temp_action_env in eval_temp_actions_env:
                    eval_one_hot_action_env.append(eval_temp_action_env[i])
                eval_actions_env.append(eval_one_hot_action_env)

            # Obser reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
            eval_episode_rewards.append(eval_rewards)

            obs_snap = np.stack(
                [np.array(list(eval_obs[:, aid]))
                for aid in range(self.num_agents)],
                axis=1)
            num_landmarks = getattr(self.all_args, "num_landmarks", self.num_agents)
            cov = compute_coverage_rate(
                obs_snap,
                num_agents=self.num_agents,
                num_landmarks=num_landmarks,
            )
            eval_coverage_rates.append(cov)

            min_lm_dist = compute_min_landmark_distance(
                obs_snap,
                num_agents=self.num_agents,
                num_landmarks=num_landmarks,
            )
            eval_min_lm_dists.append(min_lm_dist)
            
            inter_agent_dist = compute_inter_agent_distance(
                obs_snap,
                num_agents=self.num_agents,
                num_landmarks=num_landmarks,
            )
            eval_inter_agent_dists.append(inter_agent_dist)

            eval_rnn_states[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

        eval_episode_rewards = np.array(eval_episode_rewards)
        mean_coverage = float(np.mean(eval_coverage_rates))
        num_landmarks = getattr(self.all_args, "num_landmarks", self.num_agents)
        final_landmarks = float(eval_coverage_rates[-2] * num_landmarks) if len(eval_coverage_rates) >= 2 else 0.0
        mean_min_lm_dist = float(np.mean(eval_min_lm_dists))
        mean_inter_agent_dist = float(np.mean(eval_inter_agent_dists))
        
        eval_train_infos = []
        for agent_id in range(self.num_agents):
            eval_average_episode_rewards = np.mean(np.sum(eval_episode_rewards[:, :, agent_id], axis=0))
            eval_train_infos.append({
                'eval_average_episode_rewards': eval_average_episode_rewards,
                'eval_landmark_coverage_rate': mean_coverage,
                'eval_final_covered_landmarks': final_landmarks,
                'eval_min_landmark_distance': mean_min_lm_dist,
                'eval_inter_agent_distance': mean_inter_agent_dist,
            })
            print("eval average episode rewards of agent%i: " % agent_id + str(eval_average_episode_rewards) + f"  coverage: {mean_coverage:.3f}  final landmarks: {final_landmarks:.2f}  min LM dist: {mean_min_lm_dist:.3f}  inter-agent dist: {mean_inter_agent_dist:.3f}")

        self.log_train(eval_train_infos, total_num_steps)  

    @torch.no_grad()
    def render(self):        
        all_frames = []
        render_coverage = []
        for episode in range(self.all_args.render_episodes):
            episode_rewards = []
            episode_coverage = []
            obs = self.envs.reset()
            if self.all_args.save_gifs:
                image = self.envs.render('rgb_array')[0][0]
                all_frames.append(image)

            rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

            for step in range(self.episode_length):
                calc_start = time.time()
                
                temp_actions_env = []
                for agent_id in range(self.num_agents):
                    if not self.use_centralized_V:
                        share_obs = np.array([obs[env_i][agent_id] for env_i in range(self.n_rollout_threads)])
                    self.trainer[agent_id].prep_rollout()
                    action, rnn_state = self.trainer[agent_id].policy.act(np.array([obs[env_i][agent_id] for env_i in range(self.n_rollout_threads)]),
                                                                        rnn_states[:, agent_id],
                                                                        masks[:, agent_id],
                                                                        deterministic=True)

                    action = action.detach().cpu().numpy()
                    # rearrange action
                    if self.envs.action_space[agent_id].__class__.__name__ == 'MultiDiscrete':
                        for i in range(self.envs.action_space[agent_id].shape):
                            uc_action_env = np.eye(self.envs.action_space[agent_id].high[i]+1)[action[:, i]]
                            if i == 0:
                                action_env = uc_action_env
                            else:
                                action_env = np.concatenate((action_env, uc_action_env), axis=1)
                    elif self.envs.action_space[agent_id].__class__.__name__ == 'Discrete':
                        action_env = np.squeeze(np.eye(self.envs.action_space[agent_id].n)[action], 1)
                    else:
                        raise NotImplementedError

                    temp_actions_env.append(action_env)
                    rnn_states[:, agent_id] = _t2n(rnn_state)
                   
                # [envs, agents, dim]
                actions_env = []
                for i in range(self.n_rollout_threads):
                    one_hot_action_env = []
                    for temp_action_env in temp_actions_env:
                        one_hot_action_env.append(temp_action_env[i])
                    actions_env.append(one_hot_action_env)

                # Obser reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions_env)
                episode_rewards.append(rewards)

                obs_snap = np.stack(
                    [np.array(list(obs[:, aid]))
                    for aid in range(self.num_agents)],
                    axis=1)
                cov = compute_coverage_rate(
                    obs_snap,
                    num_agents=self.num_agents,
                    num_landmarks=getattr(self.all_args, "num_landmarks", self.num_agents),
                )
                episode_coverage.append(cov)

                rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
                masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

                if self.all_args.save_gifs:
                    image = self.envs.render('rgb_array')[0][0]
                    all_frames.append(image)
                    calc_end = time.time()
                    elapsed = calc_end - calc_start
                    if elapsed < self.all_args.ifi:
                        time.sleep(self.all_args.ifi - elapsed)

            episode_rewards = np.array(episode_rewards)
            mean_ep_cov = float(np.mean(episode_coverage))
            render_coverage.append(mean_ep_cov)
            num_landmarks = getattr(self.all_args, "num_landmarks", self.num_agents)
            final_cov = float(episode_coverage[-2] * num_landmarks) if len(episode_coverage) >= 2 else 0.0
            
            for agent_id in range(self.num_agents):
                average_episode_rewards = np.mean(np.sum(episode_rewards[:, :, agent_id], axis=0))
                print("eval average episode rewards of agent%i: " % agent_id + str(average_episode_rewards) + f"  coverage: {mean_ep_cov:.3f}  final landmarks: {final_cov:.2f}")
        
        print(f"  Mean coverage : {np.mean(render_coverage):.3f} ± {np.std(render_coverage):.3f}")
        
        if self.all_args.save_gifs:
            imageio.mimsave(str(self.gif_dir) + '/render.gif', all_frames, duration=self.all_args.ifi)

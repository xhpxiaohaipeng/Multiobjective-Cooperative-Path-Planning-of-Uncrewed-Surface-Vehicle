#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gym
import argparse
import numpy as np
from parl.utils import logger, summary, ReplayMemory
#from parl.env import ActionMappingWrapper, CompatWrapper
from env.one_usv_path_plan_energy_env import usv_path_plan_env
from agent.agent import PlanAgent
from agent.model import MLPModel,MHA_att_Model,LSTM_att_Model,CNN_Model,MHA_Model
#from parl.algorithms import TD3,SAC
from sac import SAC_self as SAC
import paddle
import pandas as pd
import matplotlib.pyplot as plt
import glob

models = {'MLP':MLPModel,'CNN':CNN_Model,'MHA':MHA_Model}

WARMUP_STEPS = 5e2
MEMORY_SIZE = int(1e7)
BATCH_SIZE =128
GAMMA = 0.99
TAU = 0.005
ACTOR_LR = 1e-3
CRITIC_LR = 1e-3
EXPL_NOISE = 0.1

def write_data(data,name,method):
    test = pd.DataFrame(data=data,columns=name)
    test.to_csv('./result_models/{}_{}.csv'.format(name[0],method))

# Run episode for training
def run_train_episode(agent, env, rpm):
    action_dim = 2
    obs = env.reset()
    done = False
    episode_reward = 0
    episode_steps = 0
    T = []
    while not done:
        episode_steps += 1
        # Select action randomly or according to policy
        if rpm.size() < WARMUP_STEPS:
            action = np.random.uniform(-1, 1, size=action_dim)
        else:
            action = agent.sample(obs)

        # Perform action
        next_obs, reward, done, goal = env.step(action)
        terminal = done #if episode_steps < env._max_episode_steps else 0

        # Store data in replay memory
        #T.append([obs, action, reward, next_obs, terminal])
        rpm.append(obs, action, reward, next_obs, terminal)


        obs = next_obs
        episode_reward += reward

        if done:
            if goal:
                paddle.save(agent.alg.model.actor_model.state_dict(), 'result_models/plan_goal_actor.pdparams')
                paddle.save(agent.alg.model.critic_model.state_dict(), 'result_models/plan_goal_mean_critic.pdparams')
            # final_reward = T[-1][2]
            # for obs, action, reward, next_obs, terminal in T:
            #     rpm.append(obs, action, final_reward, next_obs, terminal)

            # Train agent after collecting sufficient data
            # if rpm.size() >= WARMUP_STEPS:
            #     batch_obs, batch_action, batch_reward, batch_next_obs, batch_terminal = rpm.sample_batch(
            #             BATCH_SIZE)
            #     agent.learn(batch_obs, batch_action, batch_reward, batch_next_obs,
            #                     batch_terminal)

        # # Train agent after collecting sufficient data
        if rpm.size() >= WARMUP_STEPS:
            batch_obs, batch_action, batch_reward, batch_next_obs, batch_terminal = rpm.sample_batch(BATCH_SIZE)
            agent.learn(batch_obs, batch_action, batch_reward, batch_next_obs,
                        batch_terminal)

    return episode_reward/episode_steps, episode_steps


# Runs policy for 5 episodes by default and returns average reward
# A fixed seed is used for the eval environment
def run_evaluate_episodes(agent, env, eval_episodes):
    avg_reward = 0.
    episode_steps = 0
    for id in range(eval_episodes):
        obs = env.reset()
        done = False
        while not done:
            episode_steps +=1
            action = agent.predict(obs)
            obs, reward, done, _ = env.step(action)
            #env.render()
            avg_reward += reward
    avg_reward /= eval_episodes
    return avg_reward/episode_steps



def main(method):
    logger.info("------------------- SAC ---------------------")
    logger.info('Env: {}, Seed: {}'.format(args.env, args.seed))
    logger.info("---------------------------------------------")
    logger.set_dir('./{}_{}'.format(args.env, args.seed))

    parm_path = "env/benchmark/8x8_obst12/map_8by8_obst12_agents1_ex68.yaml"

    obs_dim = 22
    action_dim = 2

    model = models[method]
    model = model(obs_dim,action_dim)

    scheduler_actor = paddle.optimizer.lr.StepDecay(learning_rate=ACTOR_LR, step_size=5000, gamma=0.1, verbose=False)
    scheduler_crictor = paddle.optimizer.lr.StepDecay(learning_rate=CRITIC_LR, step_size=5000, gamma=0.1, verbose=False)
    restore = False
    if restore:
        model.actor_model.set_state_dict(paddle.load('result_models/plan_best_mean_actor_{}.pdparams'.format(method)))
        model.critic_model.set_state_dict(paddle.load('result_models/plan_best_mean_critic_{}.pdparams'.format(method)))

    algorithm = SAC(
        model,
        gamma=GAMMA,
        tau=TAU,
        alpha=0.8,
        actor_lr=scheduler_actor.get_lr(),
        critic_lr=scheduler_crictor.get_lr())
    agent = PlanAgent(algorithm)

    rpm = ReplayMemory(
        max_size=MEMORY_SIZE, obs_dim=obs_dim, act_dim=action_dim)
    #agent.restore()
    train_rewards = []
    valid_rewards = []
    max_reward = -np.inf
    max_mean_reward = -np.inf
    for total_steps in range(args.train_total_steps):
        # Train episodene
        #id = np.random.choice(len(parm_train_paths),1)
        #parm_path = parm_train_paths[id[0]]
        env = usv_path_plan_env(parm_path)
        episode_reward, episode_steps = run_train_episode(agent, env, rpm)
        #print(episode_reward)
        #total_steps += episode_steps
        summary.add_scalar('train/episode_reward', episode_reward, total_steps)
        #logger.info('Total Steps: {} Reward: {}'.format(
          #  total_steps, episode_reward))

        # Evaluate episode
        if (total_steps) % args.test_every_steps == 0 and total_steps>0:
            env = usv_path_plan_env(parm_path,train_is=False)
            avg_reward = run_evaluate_episodes(agent, env, EVAL_EPISODES)
            train_rewards.append(episode_reward)
            valid_rewards.append(avg_reward)
            paddle.save(agent.alg.model.actor_model.state_dict(),'result_models/plan_actor_{}.pdparams'.format(method))
            paddle.save(agent.alg.model.critic_model.state_dict(), 'result_models/plan_critic_{}.pdparams'.format(method))
            if np.mean(valid_rewards)> max_mean_reward:
                print("Mean Reward rised!save model!best reward:{}".format(np.mean(valid_rewards)))
                max_mean_reward = np.mean(valid_rewards)
                paddle.save(agent.alg.model.actor_model.state_dict(), 'result_models/plan_best_mean_actor_{}.pdparams'.format(method))
                paddle.save(agent.alg.model.critic_model.state_dict(), 'result_models/plan_best_mean_critic_{}.pdparams'.format(method))


            if avg_reward  > max_reward:
                print("Best Reward rised!save model!best reward:{}".format(avg_reward))
                max_reward = avg_reward
                paddle.save(agent.alg.model.actor_model.state_dict(), 'result_models/plan_best_actor_{}.pdparams'.format(method))
                paddle.save(agent.alg.model.critic_model.state_dict(), 'result_models/plan_best_critic_{}.pdparams'.format(method))
            #agent.save('result_models/plan.ckpt')
            summary.add_scalar('eval/episode_reward', avg_reward, total_steps)
            logger.info('Steps: {}/{}, Train Reward: {},Valid Reward:{}'.format(
                total_steps,args.train_total_steps, np.mean(train_rewards),np.mean(valid_rewards)))
        scheduler_actor.step()
        scheduler_crictor.step()
    write_data(train_rewards,['train_reward'],method)
    write_data(valid_rewards, ['test_reward'],method)
    plot_values = [(valid_rewards,train_rewards)]
    for i, plot_value in enumerate(plot_values):
        if i < 2:
            for j in range(1):
                plt.plot(plot_value[0], label='Test_reward' if i == 0 else 'Max_test_reward')
                plt.plot(plot_value[1], label='Train_reward' if i == 0 else 'Max_train_reward')
                plt.title("Reward Trend ")
                plt.xlabel('Episode')
                plt.ylabel('Reward')
                plt.legend(loc=0)
                plt.savefig('./result_models/reward_{}.png'.format(method), dpi=300,
                            bbox_inches='tight')
                plt.show()
        # else:
        #     plt.plot(plot_value[0], label='Mean_test_reward')
        #     plt.plot(plot_value[1], label='Mean_train_reward')
        #     plt.title("Mean Reward of {}".format(model_name))
        #     plt.xlabel('Episode{:.1f}min'.format(time_train / 60))
        #     plt.ylabel('Reward')
        #     plt.legend(loc=0)
        #     plt.savefig("./result_models/Images/reward_mean_{}.png".format(model_name), dpi=300, bbox_inches='tight')
        #     plt.show()

EVAL_EPISODES = 1
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env", default="multi-agent", help='environment name')
    parser.add_argument("--seed", default=0, type=int, help='Sets Gym seed')
    parser.add_argument(
        "--train_total_steps",
        default=int(6e4),
        type=int,
        help='Max time steps to run environment')
    parser.add_argument(
        '--test_every_steps',
        type=int,
        default=int(10),
        help='The step interval between two consecutive evaluations')
    parser.add_argument(
        '--policy_freq',
        type=int,
        default=2,
        help='Frequency of delayed policy updates')
    args = parser.parse_args()
    method = 'CNN'
    main(method)

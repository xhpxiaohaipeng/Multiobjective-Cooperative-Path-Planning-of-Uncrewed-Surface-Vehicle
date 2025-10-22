import gym
import argparse
import numpy as np
from parl.utils import logger, summary, ReplayMemory
#from parl.env import ActionMappingWrapper, CompatWrapper
#from env.Test_multi_usv_path_plan_env import Test_multi_usv_path_plan_env
from env.one_usv_path_plan_energy_env import usv_path_plan_env
from agent.agent import PlanAgent
from agent.model import MLPModel,MHA_att_Model,LSTM_att_Model,CNN_Model
#from parl.algorithms import TD3,SAC
from sac import SAC_self as SAC
import paddle
import pandas as pd
import matplotlib.pyplot as plt
import glob
import time
from plot_result import plot_result
WARMUP_STEPS = 5e2
EVAL_EPISODES = 1
MEMORY_SIZE = int(1e6)
BATCH_SIZE = 256
GAMMA = 0.99
TAU = 0.005
ACTOR_LR = 1e-3
CRITIC_LR = 1e-3
EXPL_NOISE = 0.1
models = {'MLP':MLPModel,'CNN':CNN_Model}

def test(agent, parm_path,method,model_name,eval_episodes=1,energy=True,render=True,):
    avg_reward = 0.
    episode_steps = 0
    ES = []
    DS = []
    TS = []
    V_s = []
    ARC_s  = []
    goal_safes = []
    times = 0
    P_all = []
    for r in range(eval_episodes):
        env = usv_path_plan_env(parm_path, train_is=False, Test=True,Energy=energy)
        obs = env.reset()
        done = False
        indexs = []
        step = 0
        while not done:
            if step == 0:
                indexs.append('R{}'.format(r))
            else:
                indexs.append(step)
            episode_steps += 1
            step += 1

            st = time.time()
            action = agent.predict(obs)
            times += (time.time()-st)
            obs, reward, done, _ = env.step(action)
            # env.render()
            avg_reward += reward

        indexs[len(indexs)-1] = 'R{}_end'.format(r)
        #print('infer time:',times)
        if render:
            E,dis,Time,x,y,m_v,m_ARC,goal_safe,P = env.render('./result_models/CSV/trajectory_{}_{}'.format(method,model_name),method,model_name,avg_reward/eval_episodes/episode_steps)
            ES.append(E)
            DS.append(dis)
            TS.append(Time)
            V_s.append(m_v)
            ARC_s.append(m_ARC)
            goal_safes.append(goal_safe)
        P_index = np.concatenate([np.array(indexs)[None, :], np.array(P)[None, :]], axis=0)
        if isinstance(P_all,list):
            P_all = P_index
        else:
            P_all = np.concatenate([P_all,P_index],axis=1)
    #print (P_all)
    P_all = P_all.T
    name = ['time', 'load']
    load_record = pd.DataFrame(columns=name, data=P_all)
    load_record.to_csv('./result_models/load_USV_test_{}_{}.csv'.format(method, model_name))

    print('Env:{},Model:{},Mean E:{},dis:{},Time:{},mean speed:{},mean ARC:{},safe goal num:{}'.format(method,model_name,np.mean(ES),np.mean(DS),np.mean(TS),np.mean(V_s),np.mean(ARC_s),np.sum(goal_safes)))
    avg_reward /= eval_episodes
    return avg_reward/episode_steps,x,y

def main(method,model_name):

    #parm_path = "env/benchmark/8x8_obst12/map_8by8_obst12_agents8_ex74.yaml"
    parm_path = "env/benchmark/8x8_obst12/map_8by8_obst12_agents1_ex68.yaml"
    #parm_paths = glob.glob('env/data/test/*')
    #print(parm_paths)

    #id = np.random.choice(len(parm_paths), 1)
    #parm_path = parm_paths[id[0]]
    #print(id)
    if method == 'plan':
        obs_dim = 15
    else:
        obs_dim = 22
    action_dim = 2
    # Initialize model, algorithm, agent, replay_memory
    #model = PlanModel(obs_dim, action_dim)
    # algorithm = TD3(
    #     model,
    #     gamma=GAMMA,
    #     tau=TAU,
    #     actor_lr=ACTOR_LR,
    #     critic_lr=CRITIC_LR,
    #     policy_freq=args.policy_freq)
    scheduler_actor = paddle.optimizer.lr.StepDecay(learning_rate=ACTOR_LR, step_size=3000, gamma=0.1, verbose=False)
    scheduler_crictor = paddle.optimizer.lr.StepDecay(learning_rate=CRITIC_LR, step_size=3000, gamma=0.1, verbose=False)
    if method == 'plan':
        model = models[model_name]
        model = model(obs_dim, action_dim)
        #model = MLPModel(obs_dim, action_dim)
        restore = True
        if restore:
            model.actor_model.set_state_dict(paddle.load('result_plan/plan_best_actor.pdparams'))
            model.critic_model.set_state_dict(paddle.load('result_plan/plan_best_mean_critic.pdparams'))
    if method == 'plan_energy':
        model = models[model_name]
        model = model(obs_dim, action_dim)
        #model = LSTM_att_Model(obs_dim, action_dim)
        restore = True
        if restore:
            model.actor_model.set_state_dict(paddle.load('result_models/plan_actor_{}.pdparams'.format(model_name)))
            model.critic_model.set_state_dict(paddle.load('result_models/plan_best_mean_critic_{}.pdparams'.format(model_name)))
    algorithm = SAC(
        model,
        gamma=GAMMA,
        tau=TAU,
        alpha=0.2,
        actor_lr=scheduler_actor.get_lr(),
        critic_lr=scheduler_crictor.get_lr())

    agent = PlanAgent(algorithm)
    rpm = ReplayMemory(
        max_size=MEMORY_SIZE, obs_dim=obs_dim, act_dim=action_dim)


    #env = usv_path_plan_env(parm_path)
    #print(method,":")
    if method == 'plan':
        avg_reward,x,y = test(agent, parm_path, method, model_name,2,False)
    else:
        avg_reward,x,y = test(agent, parm_path, method, model_name, 1)
    #print(avg_reward)

    return x,y

if __name__ == "__main__":
    methods = ['plan_energy']# 'plan',
    #model_name = 'MLP'
    for method in methods:
        if method == 'plan':
            x,y = main(method,'MLP')
            #plot_result(method, 'MLP',x,y)
        else:
            for name in ['CNN']: # 'MLP',
                x,y = main(method, name)
                plot_result(method,name,x,y)

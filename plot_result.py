import yaml
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.patches import Circle, Rectangle, Arrow
import glob
import numpy as np
import matplotlib.patches as mpatches
import random
import pandas as pd
from _collections import deque


def plot_reward():

    fig, ax = plt.subplots(1, 1)
    data1 = pd.read_csv('./result_models/test_reward_CNN.csv')['test_reward'].values[:2000]
    data2 = pd.read_csv('./result_models/test_reward_MLP.csv')['test_reward'].values[:2000]

    Att_rpm = deque(maxlen=10)
    MLP_rpm = deque(maxlen=4)


    mean_Att = []
    mean_MLP = []


    mean_Att1 = []
    mean_MLP1 = []

    max_Att = []
    max_MLP = []


    Att = []
    MLP = []


    for i in range(len(data1)):
        Att_rpm.append(data1[i])
        MLP_rpm.append(data2[i])


        Att.append(data1[i])
        MLP.append(data2[i])


        max_Att.append(max(Att))
        max_MLP.append(max(MLP))


        mean_Att.append(np.mean(Att))
        mean_MLP.append(np.mean(MLP))
        mean_Att1.append(np.mean(Att_rpm))
        mean_MLP1.append(np.mean(MLP_rpm))

    data1 = max_Att
    data2 = mean_MLP


    #ax.plot(data1, label='test reward')
    #ax.plot(data2, label="train reward")
    ax.plot(range(len(Att)),Att,label='Reward of FRCF',linestyle='--')
    ax.plot(mean_Att,label='Mean Reward of FRCF',linewidth=2)
    ax.plot(max_Att,label='Max Reward of FRCF',linewidth=2)

    ax.plot(range(len(Att)),MLP,label='Reward of MLP',linestyle='--')
    ax.plot(mean_MLP,label='Mean Reward of MLP',linewidth=2)
    ax.plot(max_MLP,label='Max Reward of MLP',linewidth=2)



    # ax.legend()

    ax.legend()
    plt.savefig('result_models/image/reward1.png',dpi=300,bbox_inches='tight')
    plt.show()


def find_area(x,y,sea_state):
    for name,pos in sea_state.items():
        #print(pos)
        x_min = pos['x_min']
        x_max = pos['x_max']
        y_min = pos['y_min']
        y_max = pos['y_max']
        ARC = pos['ARC']
        if  x_min<= x <=x_max and y_min<=y<=y_max:
            break

    return ARC

def generate_sea_state(length=0.4,sea_state_file='env/sea_state1.yaml'):
    """
    生成海洋环境信息
    """
    map_file = "env/benchmark/8x8_obst12/map_8by8_obst12_agents1_ex68.yaml"
    with open(map_file) as map_file:
        map = yaml.load(map_file, Loader=yaml.FullLoader)
    fig = plt.figure(figsize=(6, 6))
    fs_list = [-0.1, -0.2, 0, 0.1, 0.2, 0.3, 0.4]
    sea_state_areas_dict = {}
    for x in np.arange(0,map['map']['dimensions'][0],length):
        for y in np.arange(0,map['map']['dimensions'][1],length):
            sea_state_areas_dict["area:({},{},{},{})".format(float(x),float(y),float(x+length),float(y+length))] =\
                {'x_min':float(x),'y_min':float(y),'x_max':float(x+length),'y_max':float(y+length),
                 'ARC':float(np.random.choice(fs_list,size=1,p=[0.1,0.1,0.4,0.1,0.1,0.1,0.1])[0])}
    scale = 100
    colors = {-0.1: 'r', -0.2: 'b', 0: 'c', 0.1: 'yellow', 0.2: 'm', 0.3: 'g', 0.4: 'k'}
    ARC_EXIT = set()
    for name, pos in sea_state_areas_dict.items():
        # print(pos)
        x_min = pos['x_min']
        x_max = pos['x_max']
        y_min = pos['y_min']
        y_max = pos['y_max']
        ARC = pos['ARC']
        if ARC not in ARC_EXIT:
            # print('legend')
            rect = mpatches.Rectangle((x_min * scale, y_min * scale), (x_max - x_min) * scale,
                                      (y_max - y_min) * scale,
                                      # fill=False,
                                      alpha=0.2,
                                      facecolor=colors[ARC], label=ARC)
        else:
            rect = mpatches.Rectangle((x_min * scale, y_min * scale), (x_max - x_min) * scale,
                                      (y_max - y_min) * scale,
                                      # fill=False,
                                      alpha=0.2,
                                      facecolor=colors[ARC])
        if len(ARC_EXIT) < len(colors):
            ARC_EXIT.add(ARC)
        plt.gca().add_patch(rect)
        plt.axis('scaled')
    with open(sea_state_file, 'w') as output_yaml:
        yaml.safe_dump(sea_state_areas_dict, output_yaml)
    fig.legend(ncol=4, bbox_to_anchor=(0.92, 0.961))
    plt.show()



def plot_result(method,model_name,x_s,y_s):
    #fig1, ax = plt.subplots(figsize=(10, 10), facecolor='w')
    fig = plt.figure(figsize=(6, 6))
    map_file = "env/benchmark/8x8_obst12/map_8by8_obst12_agents1_ex68.yaml"

    # parm_paths = glob.glob('env/data/test/*')
    # print(parm_paths)

    # id = np.random.choice(len(parm_paths), 1)
    # map_file = parm_paths[0]
    # print(id)
    schedule_file = 'result_models/CSV/trajectory_{}_{}_{}_{}.yaml'.format(method,model_name,x_s,y_s)
    with open(map_file) as map_file:
        map = yaml.load(map_file, Loader=yaml.FullLoader)

    with open(schedule_file) as states_file:
        schedule = yaml.load(states_file, Loader=yaml.FullLoader)

    solution_x = defaultdict(list)
    solution_y = defaultdict(list)

    scale = 100
    # print(map,schedule)
    obstacles = map["map"]["obstacles"]

    # print(obstacles)
    # obstacles_x_list = []
    # obstacles_y_list = []
    for i, o in enumerate(obstacles):
        x, y = o[0] * scale, o[1] * scale
        # Rectangle((x - 0.5, y - 0.5), 0.1, 0.1, facecolor='red', edgecolor='red')
        if i == 0:
            circle = plt.Circle((x, y), 20, color='r',label='obstacle')

        else:
            circle = plt.Circle((x, y), 20, color='r')
        # plt.legend()
        plt.gca().add_patch(circle)
        plt.axis('scaled')
        # obstacles_x_list.append(o[0])
        # obstacles_y_list.append(o[1])

    """
    生成海洋环境信息
    """
    # for x in np.arange(0,map['map']['dimensions'][0],length):
    #     for y in np.arange(0,map['map']['dimensions'][1],length):
    #         #x_min, y_min,x_max,y_max = env_state[0] * scale, env_state[1] * scale,env_state[2]*scale,env_state[3]*scale
    #         #left, bottom, width, height = (0, 0, 200, 200)
    #         #print(i,j)
    #         sea_state_areas_dict["area:({},{},{},{})".format(float(x),float(y),float(x+length),float(y+length))] =\
    #             {'x_min':float(x),'y_min':float(y),'x_max':float(x+length),'y_max':float(y+length),
    #              'ARC':float(np.random.choice(fs_list,size=1,p=[0.1,0.1,0.4,0.1,0.1,0.1,0.1])[0])}
    #                                                     #round(random.uniform(-0.2,0.4),1)
    #         #sea_state_areas_list.append((x, y, x + length, y + length)) = 0.1
    #         rect = mpatches.Rectangle((x*scale, y*scale), (length)*scale, (length)*scale,
    #                               #fill=False,
    #                               alpha=0.3,
    #                               facecolor=colors[sea_state_areas_dict["area:({},{},{},{})".
    #                                   format(float(x),float(y),float(x+length),float(y+length))]['ARC']])
    #         plt.gca().add_patch(rect)
    #         #plt.text((x_min+x_max)/2, (y_min+y_max)/2, 'f=0.1', fontsize=8, color="black", weight="bold")
    #         #plt.legend()
    #         plt.axis('scaled')
    # #print(sea_state_areas_dict)
    # with open('env/sea_state.yaml', 'w') as output_yaml:
    #     yaml.safe_dump(sea_state_areas_dict, output_yaml)
    #print(len(sea_state_areas))

    #展示海洋环境信息
    #colors = {-0.1:'lightcyan',-0.2:'thistle',0:'mistyrose',0.1:'yellowgreen',0.2:'c',0.3:'plum',0.4:'orchid'}
    colors = {-0.1: 'r', -0.2: 'b', 0: 'c', 0.1: 'yellow', 0.2: 'm', 0.3: 'g', 0.4: 'k'}
    with open('env/sea_state.yaml', 'r') as sea_file:
        try:
            sea_states = yaml.load(sea_file, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            print(exc)
    ARC_EXIT = set()
    for name, pos in sea_states.items():
        # print(pos)
        x_min = pos['x_min']
        x_max = pos['x_max']
        y_min = pos['y_min']
        y_max = pos['y_max']
        ARC = pos['ARC']
        if ARC not in ARC_EXIT:
            # print('legend')
            rect = mpatches.Rectangle((x_min * scale, y_min * scale), (x_max - x_min) * scale,
                                          (y_max - y_min) * scale,
                                          # fill=False,
                                          alpha=0.3,
                                          facecolor=colors[ARC], label=ARC)
        else:
            rect = mpatches.Rectangle((x_min * scale, y_min * scale), (x_max - x_min) * scale,
                                          (y_max - y_min) * scale,
                                          # fill=False,
                                          alpha=0.3,
                                          facecolor=colors[ARC])
        if len(ARC_EXIT) < len(colors):
            ARC_EXIT.add(ARC)
        plt.gca().add_patch(rect)
        # plt.text((x_min+x_max)/2, (y_min+y_max)/2, 'f=0.1', fontsize=8, color="black", weight="bold")
        # plt.legend()
        plt.axis('scaled')
        # fig.legend(ncol=4, bbox_to_anchor=(1.09, 1.13))
        # print(name,pos)
        # print(schedul)

    for d, i in zip(map["agents"], range(0, len(map["agents"]))):
        #print(d)
        name = d['name']
        # goal = d['goal']
        # start= (3.051320876143258 ,1.475290378292728)#d['start']
        state = np.load('result_models/CSV/start_{}_{}.npy'.format(x_s,y_s))
        #print(state)
        start = state[:2]
        goal = state[2:]
        circle1 = plt.Circle((goal[0] * scale, goal[1] * scale), 10, color='b', label='goal')
        circle2 = plt.Circle((start[0] * scale, start[1] * scale), 10, color='g', label='start')
        # plt.legend()
        plt.gca().add_patch(circle1)
        plt.gca().add_patch(circle2)
        plt.axis('scaled')
    # plt.scatter(obstacles_x_list,obstacles_y_list)

    for name, positions in schedule["schedule"].items():
        for pos in positions:
            solution_x[name].append(pos['x'] * scale)
            solution_y[name].append(pos['y'] * scale)

    for name, positions in schedule["schedule"].items():
        plt.plot(solution_x[name], solution_y[name],linewidth=2.0)
    fig.legend(ncol=5, bbox_to_anchor=(0.92, 0.961))
    plt.savefig('result_models/trajectory_{}_{}.png'.format(method,model_name), dpi=300,bbox_inches='tight')
    plt.show()
    # for i,j in map.items():
    #     print(i,j)

#plot_result()
#generate_sea_state()
# with open('env/sea_state.yaml') as sea_file:
#     sea_state = yaml.load(sea_file, Loader=yaml.FullLoader)
# x = 6.165
# y = 5.2
# ARC = find_area(x,y,sea_state)
# print(ARC)
#plot_reward()

def plot_map():
    #fig1, ax = plt.subplots(figsize=(10, 10), facecolor='w')
    fig = plt.figure(figsize=(6, 6))
    map_file = "env/benchmark/8x8_obst12/map_8by8_obst12_agents1_ex68.yaml"

    # parm_paths = glob.glob('env/data/test/*')
    # print(parm_paths)

    # id = np.random.choice(len(parm_paths), 1)
    # map_file = parm_paths[0]
    # print(id)
    #schedule_file = 'result_models/output_{}_{}.yaml'.format(method,model_name)
    with open(map_file) as map_file:
        map = yaml.load(map_file, Loader=yaml.FullLoader)

    # with open(schedule_file) as states_file:
    #     schedule = yaml.load(states_file, Loader=yaml.FullLoader)

    solution_x = defaultdict(list)
    solution_y = defaultdict(list)

    scale = 100
    # print(map,schedule)
    obstacles = map["map"]["obstacles"]

    # print(obstacles)
    # obstacles_x_list = []
    # obstacles_y_list = []
    for i, o in enumerate(obstacles):
        x, y = o[0] * scale, o[1] * scale
        # Rectangle((x - 0.5, y - 0.5), 0.1, 0.1, facecolor='red', edgecolor='red')
        if i == 0:
            circle = plt.Circle((x, y), 20, color='r',label='obstacle')

        else:
            circle = plt.Circle((x, y), 20, color='r')
        # plt.legend()
        plt.gca().add_patch(circle)
        plt.axis('scaled')
        # obstacles_x_list.append(o[0])
        # obstacles_y_list.append(o[1])

    """
    生成海洋环境信息
    """
    # for x in np.arange(0,map['map']['dimensions'][0],length):
    #     for y in np.arange(0,map['map']['dimensions'][1],length):
    #         #x_min, y_min,x_max,y_max = env_state[0] * scale, env_state[1] * scale,env_state[2]*scale,env_state[3]*scale
    #         #left, bottom, width, height = (0, 0, 200, 200)
    #         #print(i,j)
    #         sea_state_areas_dict["area:({},{},{},{})".format(float(x),float(y),float(x+length),float(y+length))] =\
    #             {'x_min':float(x),'y_min':float(y),'x_max':float(x+length),'y_max':float(y+length),
    #              'ARC':float(np.random.choice(fs_list,size=1,p=[0.1,0.1,0.4,0.1,0.1,0.1,0.1])[0])}
    #                                                     #round(random.uniform(-0.2,0.4),1)
    #         #sea_state_areas_list.append((x, y, x + length, y + length)) = 0.1
    #         rect = mpatches.Rectangle((x*scale, y*scale), (length)*scale, (length)*scale,
    #                               #fill=False,
    #                               alpha=0.3,
    #                               facecolor=colors[sea_state_areas_dict["area:({},{},{},{})".
    #                                   format(float(x),float(y),float(x+length),float(y+length))]['ARC']])
    #         plt.gca().add_patch(rect)
    #         #plt.text((x_min+x_max)/2, (y_min+y_max)/2, 'f=0.1', fontsize=8, color="black", weight="bold")
    #         #plt.legend()
    #         plt.axis('scaled')
    # #print(sea_state_areas_dict)
    # with open('env/sea_state.yaml', 'w') as output_yaml:
    #     yaml.safe_dump(sea_state_areas_dict, output_yaml)
    #print(len(sea_state_areas))

    #展示海洋环境信息
    #colors = {-0.1:'lightcyan',-0.2:'thistle',0:'mistyrose',0.1:'yellowgreen',0.2:'c',0.3:'plum',0.4:'orchid'}
    colors = {-0.1: 'r', -0.2: 'b', 0: 'c', 0.1: 'yellow', 0.2: 'm', 0.3: 'g', 0.4: 'k'}
    with open('env/sea_state.yaml', 'r') as sea_file:
        try:
            sea_states = yaml.load(sea_file, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            print(exc)
    ARC_EXIT = set()
    for name, pos in sea_states.items():
        # print(pos)
        x_min = pos['x_min']
        x_max = pos['x_max']
        y_min = pos['y_min']
        y_max = pos['y_max']
        ARC = pos['ARC']
        if ARC not in ARC_EXIT:
            # print('legend')
            rect = mpatches.Rectangle((x_min * scale, y_min * scale), (x_max - x_min) * scale,
                                          (y_max - y_min) * scale,
                                          # fill=False,
                                          alpha=0.3,
                                          facecolor=colors[ARC], label=ARC)
        else:
            rect = mpatches.Rectangle((x_min * scale, y_min * scale), (x_max - x_min) * scale,
                                          (y_max - y_min) * scale,
                                          # fill=False,
                                          alpha=0.3,
                                          facecolor=colors[ARC])
        if len(ARC_EXIT) < len(colors):
            ARC_EXIT.add(ARC)
        plt.gca().add_patch(rect)
        # plt.text((x_min+x_max)/2, (y_min+y_max)/2, 'f=0.1', fontsize=8, color="black", weight="bold")
        # plt.legend()
        plt.axis('scaled')
        # fig.legend(ncol=4, bbox_to_anchor=(1.09, 1.13))
        # print(name,pos)
        # print(schedul)

    for d, i in zip(map["agents"], range(0, len(map["agents"]))):
        #print(d)
        name = d['name']
        # goal = d['goal']
        # start= (3.051320876143258 ,1.475290378292728)#d['start']
        state = np.load('result_models/state.npy')
        #print(state)
        start = state[:2]
        goal = state[2:]
        circle1 = plt.Circle((goal[0] * scale, goal[1] * scale), 10, color='b', label='goal')
        circle2 = plt.Circle((start[0] * scale, start[1] * scale), 10, color='g', label='start')
        # plt.legend()
        plt.gca().add_patch(circle1)
        plt.gca().add_patch(circle2)
        plt.axis('scaled')
    # plt.scatter(obstacles_x_list,obstacles_y_list)

    # for name, positions in schedule["schedule"].items():
    #     for pos in positions:
    #         solution_x[name].append(pos['x'] * scale)
    #         solution_y[name].append(pos['y'] * scale)

    #for name, positions in schedule["schedule"].items():
        #plt.plot(solution_x[name], solution_y[name],linewidth=2.0)
    fig.legend(ncol=5, bbox_to_anchor=(0.92, 0.961))
    plt.savefig('result_models/image/sea_erea.png', dpi=300, bbox_inches='tight')
    #plt.savefig('result_models/trajectory_{}_{}.png'.format(method,model_name), dpi=300,bbox_inches='tight')
    plt.show()

def plot_all_result():
    #fig1, ax = plt.subplots(figsize=(10, 10), facecolor='w')
    fig = plt.figure(figsize=(6, 6))
    map_file = "env/benchmark/8x8_obst12/map_8by8_obst12_agents1_ex68.yaml"

    # parm_paths = glob.glob('env/data/test/*')
    # print(parm_paths)

    # id = np.random.choice(len(parm_paths), 1)
    # map_file = parm_paths[0]
    # print(id)
    #'plan', 'plan_energy'
    with open(map_file) as map_file:
        map = yaml.load(map_file, Loader=yaml.FullLoader)

    scale = 100
    # print(map,schedule)
    obstacles = map["map"]["obstacles"]

    # print(obstacles)
    # obstacles_x_list = []
    # obstacles_y_list = []
    for i, o in enumerate(obstacles):
        x, y = o[0] * scale, o[1] * scale
        # Rectangle((x - 0.5, y - 0.5), 0.1, 0.1, facecolor='red', edgecolor='red')
        if i == 0:
            circle = plt.Circle((x, y), 20, color='r', label='obstacle')

        else:
            circle = plt.Circle((x, y), 20, color='r')
        # plt.legend()
        plt.gca().add_patch(circle)
        plt.axis('scaled')
        # obstacles_x_list.append(o[0])
        # obstacles_y_list.append(o[1])

    # 展示海洋环境信息
    # colors = {-0.1:'lightcyan',-0.2:'thistle',0:'mistyrose',0.1:'yellowgreen',0.2:'c',0.3:'plum',0.4:'orchid'}
    colors = {-0.1: 'r', -0.2: 'b', 0: 'c', 0.1: 'yellow', 0.2: 'm', 0.3: 'g', 0.4: 'k'}
    with open('env/sea_state.yaml', 'r') as sea_file:
        try:
            sea_states = yaml.load(sea_file, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            print(exc)
    ARC_EXIT = set()
    for name, pos in sea_states.items():
        # print(pos)
        x_min = pos['x_min']
        x_max = pos['x_max']
        y_min = pos['y_min']
        y_max = pos['y_max']
        ARC = pos['ARC']
        if ARC not in ARC_EXIT:
            # print('legend')
            rect = mpatches.Rectangle((x_min * scale, y_min * scale), (x_max - x_min) * scale,
                                      (y_max - y_min) * scale,
                                      # fill=False,
                                      alpha=0.3,
                                      facecolor=colors[ARC], label=ARC)
        else:
            rect = mpatches.Rectangle((x_min * scale, y_min * scale), (x_max - x_min) * scale,
                                      (y_max - y_min) * scale,
                                      # fill=False,
                                      alpha=0.3,
                                      facecolor=colors[ARC])
        if len(ARC_EXIT) < len(colors):
            ARC_EXIT.add(ARC)
        plt.gca().add_patch(rect)
        # plt.text((x_min+x_max)/2, (y_min+y_max)/2, 'f=0.1', fontsize=8, color="black", weight="bold")
        # plt.legend()
        plt.axis('scaled')
        # fig.legend(ncol=4, bbox_to_anchor=(1.09, 1.13))
        # print(name,pos)
        # print(schedul)


    for k,pos in enumerate([(0.0,0.0),(4.0,0.0),(8.0,0.0),(2.0,2.65)]):
        x1 = pos[0]
        y1 = pos[1]
        schedule_file1 = 'result_models/CSV/trajectory_plan_MLP_{}_{}.yaml'.format(x1, y1)
        schedule_file2 = 'result_models/CSV/trajectory_plan_energy_MLP_{}_{}.yaml'.format(x1, y1)
        schedule_file3 = 'result_models/CSV/trajectory_plan_energy_CNN_{}_{}.yaml'.format(x1, y1)

        with open(schedule_file1) as states_file:
            schedule1 = yaml.load(states_file, Loader=yaml.FullLoader)

        with open(schedule_file2) as states_file:
            schedule2 = yaml.load(states_file, Loader=yaml.FullLoader)

        with open(schedule_file3) as states_file:
            schedule3 = yaml.load(states_file, Loader=yaml.FullLoader)

        solution_1x = defaultdict(list)
        solution_1y = defaultdict(list)

        solution_2x = defaultdict(list)
        solution_2y = defaultdict(list)

        solution_3x = defaultdict(list)
        solution_3y = defaultdict(list)

        for d, i in zip(map["agents"], range(0, len(map["agents"]))):
            # print(d)
            name = d['name']
            # goal = d['goal']
            # start= (3.051320876143258 ,1.475290378292728)#d['start']
            state = np.load('result_models/CSV/start_{}_{}.npy'.format(x1, y1))
            # print(state)
            start = state[:2]
            goal = state[2:]
            if k == 0:
                circle1 = plt.Circle((goal[0] * scale, goal[1] * scale), 5, color='b', label='goal')
                circle2 = plt.Circle((start[0] * scale, start[1] * scale), 2, color='g', label='start')
            else:
                circle1 = plt.Circle((goal[0] * scale, goal[1] * scale), 5, color='b')
                circle2 = plt.Circle((start[0] * scale, start[1] * scale), 2, color='g')
            # plt.legend()
            plt.gca().add_patch(circle1)
            plt.gca().add_patch(circle2)
            plt.axis('scaled')
        # plt.scatter(obstacles_x_list,obstacles_y_list)

        for name, positions in schedule1["schedule"].items():
            for pos in positions:
                solution_1x[name].append(pos['x'] * scale)
                solution_1y[name].append(pos['y'] * scale)

        for name, positions in schedule1["schedule"].items():
            if k == 0:
                plt.plot(solution_1x[name], solution_1y[name], linewidth=1.5, color= 'peru', label='NEBM')
            else:
                plt.plot(solution_1x[name], solution_1y[name], linewidth=1.5, color='peru')

        for name, positions in schedule2["schedule"].items():
            for pos in positions:
                solution_2x[name].append(pos['x'] * scale)
                solution_2y[name].append(pos['y'] * scale)

        for name, positions in schedule2["schedule"].items():
            if k == 0:
                plt.plot(solution_2x[name], solution_2y[name], linewidth=1.5,color = 'brown', label='CEBM')
            else:
                plt.plot(solution_2x[name], solution_2y[name], linewidth=1.5, color='brown')

        for name, positions in schedule3["schedule"].items():
            for pos in positions:
                solution_3x[name].append(pos['x'] * scale)
                solution_3y[name].append(pos['y'] * scale)

        for name, positions in schedule3["schedule"].items():
            if k == 0:
                plt.plot(solution_3x[name], solution_3y[name], linewidth=1.5, color='g',label='CEBF')
            else:
                plt.plot(solution_3x[name], solution_3y[name], linewidth=1.5, color='g')
    fig.legend(ncol=5, bbox_to_anchor=(0.93, 0.993))
    plt.savefig('result_models/image/trajectorys_all.png', dpi=300, bbox_inches='tight')
    #plt.savefig('result_models/trajectory_{}_{}.png'.format(method,model_name), dpi=300,bbox_inches='tight')
    plt.show()

def plot_speed_and_power(method,model_name):
    './result_models/actor_USV_{}_{}.csv'
    actor = pd.read_csv('./result_models/actor_USV_{}_{}.csv'.format(method,model_name))
    #print(actor)
    P_load = actor['p_load'].values
    E = actor['P_E'].values
    V = actor['V'].values
    heading = actor['heading'].values
    ARC = actor['ARC'].values
    plot_values = [E,(ARC,V)]
    for i, plot_value in enumerate(plot_values):
        for j in range(1):
            # if i == 0:
            #     plt.title("PL")
            #     plt.ylabel('kw')
            #     plt.plot(plot_value, linestyle='--', marker='*', color='r', label="PL")
            #     plt.legend(loc=0)
            #     plt.savefig("./result_models/PL_USV_plan_energy_CNN.png", dpi=300,
            #                 bbox_inches='tight')
            if i == 0:
                plt.title("E")
                plt.ylabel('kwh')
                plt.plot(plot_value, linestyle='--', marker='*', color='r', label="E")
                plt.tick_params(labelsize=15)
                plt.legend(loc=0)
                plt.savefig("./result_models/image/E_USV_{}_{}.png".format(method,model_name), dpi=300,
                            bbox_inches='tight')
            # if i == 2:
            #     ax = plt.subplot()
            #     ax2 = ax.twinx()
            #     plt.title("V")
            #     plt.ylabel('kn/h')
            #     plt.plot(plot_value, linestyle='--', marker='*', color='r', label="Speed")
            #     plt.legend(loc=0)
            #     plt.savefig("./result_models/Speed_USV_plan_energy_CNN.png", dpi=300,
            #                 bbox_inches='tight')
            # if i == 3:
            #     plt.title("heading")
            #     #plt.ylabel('’')
            #     plt.plot(plot_value, linestyle='--', marker='*', color='r', label="heading")
            #     plt.legend(loc=0)
            #     plt.savefig("./result_models/heading_USV_plan_energy_CNN.png", dpi=300,
            #                 bbox_inches='tight')
            if i == 1:
                fig = plt.figure()
                print(method,model_name,' mean ARC:',np.mean(ARC))
                ax = plt.subplot()
                ax2 = ax.twinx()
                ax.set_ylabel("AEC", color='black',fontsize=14)
                ax.set_xlabel("Time Step/20min", color='black',fontsize=14)
                ax2.set_ylabel("Speed/kn",fontsize=14)
                plt.title("ARC and speed",fontsize=15)
                ax.tick_params(labelsize=12)
                ax2.tick_params(labelsize=12)
                #plt.ylabel('’')
                ax.plot(plot_value[0], linestyle='--', marker='*', color='r', label="ARC")
                ax2.plot(plot_value[1],linestyle='--', marker='*', color='b', label="speed")
                # ax.legend(inses, labs,ncol=5,bbox_to_anchor=(1.0, 1.22))
                fig.legend(ncol=2, bbox_transform=ax.transAxes, bbox_to_anchor=(0.48, 0.1),fontsize=12)
                #plt.legend(loc=0)
                plt.savefig("./result_models/image/ARC_speed_USV_{}_{}.png".format(method,model_name), dpi=300,
                            bbox_inches='tight')
            plt.show()

def plot_all_speed_and_power():
    './result_models/actor_USV_{}_{}.csv'
    x1 = 2.0
    y1 = 2.65
    ld = 0.5
    markersize = 1.0
    actor1 = pd.read_csv('./result_models/CSV/actor_USV_plan_MLP_{}_{}.csv'.format(x1,y1))
    actor2 = pd.read_csv('./result_models/CSV/actor_USV_plan_energy_MLP_{}_{}.csv'.format(x1, y1))
    actor3 = pd.read_csv('./result_models/CSV/actor_USV_plan_energy_CNN_{}_{}.csv'.format(x1, y1))
    #print(actor)
    P_load1 = actor1['p_load'].values
    E1 = actor1['P_E'].values
    V1 = actor1['V'].values
    heading1 = actor1['heading'].values
    ARC1= actor1['ARC'].values


    P_load2 = actor2['p_load'].values
    E2 = actor2['P_E'].values
    V2 = actor2['V'].values
    heading2 = actor2['heading'].values
    ARC2= actor2['ARC'].values


    P_load3 = actor3['p_load'].values
    E3 = actor3['P_E'].values
    V3 = actor3['V'].values
    heading3 = actor3['heading'].values
    ARC3= actor3['ARC'].values

    print(' mean ARC:', 'NEBM:',np.mean(ARC1),'CEBM:',np.mean(ARC2),'CEBF:',np.mean(ARC3))
    print(' mean V:', 'NEBM:', np.mean(V1), 'CEBM:', np.mean(V2), 'CEBF:', np.mean(V3))


    # 创建一个包含6个子图的画布
    fig, axs = plt.subplots(2, 3, figsize=(9*0.8, 3.3*0.8))
    fig.subplots_adjust(hspace=0.78,wspace=0.795)  # 调整子图之间的垂直间距
    #figs = plt.figure()

    # 绘制子图1
    ax1 = axs[0, 0]
    ax2 = ax1.twinx()

    ax1.set_ylabel("ARC", color='black', fontsize=9)
    ax1.set_xlabel("Step/20min", color='black', fontsize=9)
    ax2.set_ylabel("Speed/kn", fontsize=9)
    ax1.set_title("ARC and speed of NEBM", fontsize=9)
    ax1.set_ylim(-0.2,0.4)
    ax2.set_ylim(0,40)
    ax1.tick_params(labelsize=8.95)
    ax2.tick_params(labelsize=8.95)
    # plt.ylabel('’')
    ax1.plot(ARC1, linestyle='--', marker='.', color='r',linewidth= ld,markersize=markersize, label="ARC")
    ax2.plot(V1, linestyle='--', marker='.', color='b',linewidth= ld, markersize=markersize,label="Speed")
    # ax.legend(inses, labs,ncol=5,bbox_to_anchor=(1.0, 1.22))
    #fig.legend(ncol=2, bbox_transform=ax1.transAxes, bbox_to_anchor=(0.48, 0.1), fontsize=12)

    # 绘制子图2
    ax1 = axs[0, 1]
    ax2 = ax1.twinx()

    ax1.set_ylabel("ARC", color='black', fontsize=9)
    ax1.set_xlabel("Step/20min", color='black', fontsize=9)
    ax2.set_ylabel("Speed/kn", fontsize=9)
    ax1.set_title("ARC and speed of CEBM", fontsize=9)
    ax1.set_ylim(-0.2,0.4)
    ax2.set_ylim(0,40)
    ax1.tick_params(labelsize=8.95)
    ax2.tick_params(labelsize=8.95)
    # plt.ylabel('’')
    ax1.plot(ARC2, linestyle='--', marker='.',markersize=markersize,color='r',linewidth= ld)
    ax2.plot(V2, linestyle='--',marker='.',markersize=markersize, color='b',linewidth= ld)

    # 绘制子图3
    ax1 = axs[0, 2]
    ax2 = ax1.twinx()

    ax1.set_ylabel("ARC", color='black', fontsize=9)
    ax1.set_xlabel("Step/20min", color='black', fontsize=9)
    ax2.set_ylabel("Speed/kn", fontsize=9)
    ax1.set_title("ARC and speed of CEBF", fontsize=9)
    ax1.set_ylim(-0.2,0.4)
    ax2.set_ylim(0,40)
    ax1.tick_params(labelsize=8.95)
    ax2.tick_params(labelsize=8.95)
    # plt.ylabel('’')
    ax1.plot(ARC3, linestyle='--', marker='.', color='r',linewidth= ld,markersize=markersize)
    ax2.plot(V3, linestyle='--', marker='.', color='b',linewidth= ld,markersize=markersize)

    # 绘制子图4
    ax1 = axs[1, 0]
    ax2 = ax1.twinx()
    ax1.set_ylabel("E/Kwh", color='black', fontsize=9)
    ax1.set_xlabel("Step/20min", color='black', fontsize=9)
    ax2.set_ylabel("power/kw", fontsize=9)
    ax1.set_title("E and pl of NEBM",fontsize=9)
    ax1.set_ylim(0, 21)
    ax2.set_ylim(0, 60)
    ax1.tick_params(labelsize=8.95)
    ax2.tick_params(labelsize=8.95)
    # plt.ylabel('’')
    ax1.plot(E1, linestyle='--', marker='.', color='c',linewidth= ld,markersize=markersize,label='E')
    ax2.plot(P_load1, linestyle='--', marker='.', color='m',label='propulsive load',linewidth= ld,markersize=markersize)

    # 绘制子图5
    ax1 = axs[1, 1]
    ax2 = ax1.twinx()
    ax1.set_ylabel("E/Kwh", color='black', fontsize=9)
    ax1.set_xlabel("Step/20min", color='black', fontsize=9)
    ax2.set_ylabel("power/kw", fontsize=9)
    ax1.set_title("E and pl of CEBM", fontsize=9)
    ax1.set_ylim(0, 21)
    ax2.set_ylim(0, 60)
    ax1.tick_params(labelsize=8.95)
    ax2.tick_params(labelsize=8.95)
    # plt.ylabel('’')
    ax1.plot(E2, linestyle='--', marker='.', color='c',linewidth= ld,markersize=markersize)
    ax2.plot(P_load2, linestyle='--', marker='.', color='m',linewidth= ld,markersize=markersize)

    # 绘制子图6
    ax1 = axs[1, 2]
    ax2 = ax1.twinx()
    ax1.set_ylabel("E/Kwh", color='black', fontsize=9)
    ax1.set_xlabel("Step/20min", color='black', fontsize=9)
    ax2.set_ylabel("power/kw", fontsize=9)
    ax1.set_title("E and pl of CEBF", fontsize=9)
    ax1.set_ylim(0, 21)
    ax2.set_ylim(0, 60)
    #ax2.set_yticks([0,30,60])
    ax1.tick_params(labelsize=8.95)
    ax2.tick_params(labelsize=8.95)
    # plt.ylabel('’')
    ax1.plot(E3, linestyle='--', marker='.', color='c',linewidth= ld,markersize=markersize)
    ax2.plot(P_load3, linestyle='--', marker='.', color='m',linewidth= ld,markersize=markersize)
    # 显示图形
    fig.legend(ncol=4, bbox_transform=ax1.transAxes, bbox_to_anchor=(0.25, 3.43), fontsize=8.5)
    plt.savefig("./result_models/image/ARC_speed_USV_{}_{}.eps".format(x1,y1), dpi=600,
                           bbox_inches='tight')
    plt.show()

#plot_reward()
# plot_map()
#plot_all_result()
#plot_all_speed_and_power()
# methods = ['plan','plan_energy']
# for method in methods:
#     if method == 'plan':
#         plot_speed_and_power(method, 'MLP')
#     else:
#         for name in ['MLP', 'CNN']:
#             plot_speed_and_power(method, name)

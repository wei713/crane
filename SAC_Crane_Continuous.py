import gym
import matplotlib.pyplot as plt
import torch
import random
import numpy as np
import math
import craneenv

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 初始化游戏环境。
env = craneenv.CraneEnv()
env.reset()

# 定义action网络模型
class ModelAction(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_state = torch.nn.Sequential(
            torch.nn.Linear(4,128),
            torch.nn.ReLU(),
        )
        self.fc_miu = torch.nn.Linear(128,1)
        self.fc_std = torch.nn.Sequential(
            torch.nn.Linear(128,1),
            torch.nn.Softplus(),
        )
    
    def forward(self,state):
        state = self.fc_state(state)
        # 创建正态分布，从正态分布当中进行一个采样。
        miu = self.fc_miu(state)

        std = self.fc_std(state)

        dist = torch.distributions.Normal(miu,std)

        # 采样b个样本，这里采用的是rsample，表示重采样，从一个标准正态分布中采样，然后乘以标准差，加上均值。
        sample = dist.rsample()
        # 样本压缩到[-1,1]之间，求动作。
        action = sample.tanh()
        # 求概率对数。
        prob = dist.log_prob(sample).exp()
        # 动作的熵。
        entropy = prob/(1 - action.tanh()**2 + 1e-7)
        entropy = -entropy.log()

        action = action*2

        return action,entropy

# 定义两对value模型。
class ModelValue(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sequential = torch.nn.Sequential(
            torch.nn.Linear(5,256),
            torch.nn.ReLU(),
            torch.nn.Linear(256,256),
            torch.nn.ReLU(),
            torch.nn.Linear(256,1),
        )

    def forward(self,state,action):
        # torch.cat 拼接函数。
        state = torch.cat([state,action],dim=1)

        return self.sequential(state)
    
def save_net(model_action,model_value1,model_value2,model_value_next1,model_value_next2):
    torch.save(model_action.state_dict(),'net_save/model_action.pkl')
    torch.save(model_value1.state_dict(),'net_save/model_value1.pkl')
    torch.save(model_value2.state_dict(),'net_save/model_value2.pkl')
    torch.save(model_value_next1.state_dict(),'net_save/model_value_next1.pkl')
    torch.save(model_value_next2.state_dict(),'net_save/model_value_next2.pkl')

def load_net():
    model_action = ModelAction().to(device)
    model_value1 = ModelValue().to(device)
    model_value2 = ModelValue().to(device)
    model_value_next1 = ModelValue().to(device)
    model_value_next2 = ModelValue().to(device)

    model_value_next1.load_state_dict(model_value1.state_dict()) # state_dict 存放训练过程中需要学习的权重和偏执系数。
    model_value_next2.load_state_dict(model_value2.state_dict()) # load_state_dict 函数就是用于将预训练的参数权重加载到新的模型之中。

    try:
        model_action.load_state_dict(torch.load('net_save/model_action.pkl'))
        model_value1.load_state_dict(torch.load('net_save/model_value1.pkl'))
        model_value2.load_state_dict(torch.load('net_save/model_value2.pkl'))
        model_value_next1.load_state_dict(torch.load('net_save/model_value_next1.pkl'))
        model_value_next2.load_state_dict(torch.load('net_save/model_value_next2.pkl'))
    except:
        pass
    return model_action,model_value1,model_value2,model_value_next1,model_value_next2

model_action,model_value1,model_value2,model_value_next1,model_value_next2 = load_net()

# 获取动作函数。
def get_action(state):
    state = torch.FloatTensor(state).reshape(1,4).to(device)
    action,_ = model_action(state)
    return action.item()

get_action([1,2,3,4])

# 定义样本池。
datas = []
# 更新样本池函数，准备离线学习。
# 向样本池中添加N条数据，删除最古老的数据。
def update_data():
    # 初始化游戏。
    state,_ = env.reset()

    over = False
    while not over:
        # 根据当前状态得到一个动作。
        action = get_action(state)
        # 执行动作，获得反馈。
        next_state,reward,over,_ = env.step(action)
        # 记录数据样本。
        datas.append((state,action,reward,next_state,over))
        # 更新游戏状态，开始下一个动作。
        state = next_state
 
    # 删除最古老的数据。       
    while len(datas) > 20000:
        datas.pop(0)

update_data()
# len(datas),datas[0]


# 采样函数。
def get_sample():
    samples = random.sample(datas,128)
    state = torch.FloatTensor([i[0] for i in samples]).reshape(-1,4).to(device)
    action = torch.FloatTensor([i[1] for i in samples]).reshape(-1,1).to(device)
    reward = torch.FloatTensor([i[2] for i in samples]).reshape(-1,1).to(device)
    next_state = torch.FloatTensor([i[3] for i in samples]).reshape(-1,4).to(device)
    over = torch.LongTensor([i[4] for i in samples]).reshape(-1,1).to(device)

    return state,action,reward,next_state,over


REWARD = []

# 测试函数。
def test():
    state,_ = env.reset()

    reward_sum = 0
    X = []
    THETA = []
    over = False
    ACTION = []
    while not over:
        action = get_action(state)
        state,reward,over,_ = env.step(action)
        X.append(state[0])
        THETA.append(state[2])
        ACTION.append(action)
        reward_sum += reward
    REWARD.append(reward_sum)

    return reward_sum,X,THETA,ACTION

#test(play=False)

# 软更新函数。
def soft_update(model,model_next):
    for param,param_next in zip(model.parameters(),model_next.parameters()):
        # 以一个小的比例更新。
        value = param_next.data * 0.995 + param.data * 0.005
        param_next.data.copy_(value)
# soft_update(torch.nn.Linear(4,64),torch.nn.Linear(4,64))

# alpha 一个可学习的参数。
alpha = torch.tensor(math.log(0.1)).to(device)
alpha.requires_grad = True

# 计算target，同时要考虑熵。
def get_target(reward,next_state,over):
    # 首先使用model_action计算动作和动作的熵。
    action,entropy = model_action(next_state)
    # 评估next_state的价值。 [b,4],[b,1]->[b,1]
    target1 = model_value_next1(next_state,action)
    target2 = model_value_next2(next_state,action)
    # 出于稳定性考虑，取值小的。[b,1]
    target = torch.min(target1,target2)
    # exp 和 log 互为反操作，把alpha还原了。target加上了动作的熵，alpha作为权重系数。
    # [b,1] - [b,1] -> [b,1]。
    target += alpha.exp() * entropy
    # [b,1]
    target *= 0.99
    target *= (1-over)
    target += reward

    return target
# get_target(reward,next_state,over).shape

# 计算action模型的loss，要求最大化熵，增加动作的随机性。
def get_loss_action(state):
    # 计算action的熵。[b,3] -> [b,1],[b,1]。
    action,entropy = model_action(state)

    # 使用两个value网络评估action的价值。[b,3],[b,1] -> [b,1]。
    value1 = model_value1(state,action)
    value2 = model_value1(state,action)

    # 出于稳定性考虑，取价值小的。[b,1]。
    value = torch.min(value1,value2)

    # alpha 还原后乘以熵，期望值越大越好，由于是计算loss，符号取反。
    # [b] - [b,1] -> [b,1]
    loss_action = -alpha.exp() * entropy
    # 减去value。
    loss_action -= value
    
    return loss_action.mean(),entropy

# get_loss_action(state)

def read_score():
    f = open('net_save/reward_best.txt','r')
    a = float(f.read())
    f.close()
    return a

def write_score(SCORE_BEST):
    f = open('net_save/reward_best.txt','w')
    f.write(str(SCORE_BEST))
    f.close()

def write_txt(addr,LOSS):
    f = open(addr,'w')
    for i in LOSS:
        f.write(str(float(i)) + " ")
    f.close()

def train():
    

    optimzer_action = torch.optim.Adam(model_action.parameters(),lr=3e-3)
    optimzer_value1 = torch.optim.Adam(model_value1.parameters(),lr=3e-3)
    optimzer_value2 = torch.optim.Adam(model_value2.parameters(),lr=3e-3)
    # alpha 也是要更新的参数，所以要定义优化器。
    optimzer_alpha = torch.optim.Adam([alpha],lr=3e-3)

    loss_fn = torch.nn.MSELoss()

    LOSS = []
    ALPHA = []
    X_best = []
    THETA_best = []
    ACTION_best = []

    try:
        SCORE_BEST = read_score()
    except:
        SCORE_BEST = -np.inf

    # 训练N次。
    for epoch in range(501):
        # 更新N条数据。
        update_data()
        # 每次更新过数据后，学习N次。
        for _ in range(200):
            # 采样一批数据。
            state,action,reward,next_state,over = get_sample()

            # 计算target的熵，这个target里已经考虑了动作的熵。[b,1]
            target = get_target(reward,next_state,over)
            target = target.detach()
            # 计算两个value。
            value1 = model_value1(state,action)
            value2 = model_value2(state,action)
            # 计算两个loss，两个value的目标都是要贴近target。
            loss_value1 = loss_fn(value1,target)
            loss_value2 = loss_fn(value2,target)
            # 更新参数。
            optimzer_value1.zero_grad()
            loss_value1.backward()
            optimzer_value1.step()

            optimzer_value2.zero_grad()
            loss_value2.backward()
            optimzer_value2.step()
            # 使用model_value计算model_action的loss。
            loss_action,entropy = get_loss_action(state)

            LOSS.append(loss_action)
            optimzer_action.zero_grad()
            loss_action.backward()
            optimzer_action.step()

            # 熵乘以alpha就是alpha的loss。[b,1]->[1]。
            loss_alpha = (entropy+1).detach()*alpha.exp()
            loss_alpha = loss_alpha.mean()
            # 更新alpha的值。
            optimzer_alpha.zero_grad()
            loss_alpha.backward()
            optimzer_alpha.step()
            
            # 增量更新next模型。
            soft_update(model_value1,model_value_next1)
            soft_update(model_value2,model_value_next2)
            
            
        if epoch % 10 == 0:

            test_result, X, THETA, ACTION = test()

            if test_result > SCORE_BEST:
                save_net(model_action,model_value1,model_value2,model_value_next1,model_value2)
                print("______saving done______")
                SCORE_BEST = test_result
                write_score(SCORE_BEST)
                ACTION_best = ACTION
                X_best = X
                THETA_best = THETA
                plt.figure()
                plt.subplot(2,1,1)
                plt.plot(np.linspace(0,20,2000),X,color="r")
                plt.title(str(epoch)+"-X"),plt.grid()
                plt.subplot(2,1,2)
                plt.plot(np.linspace(0,20,2000),THETA,color="r")
                plt.title(str(epoch)+"-THETA"),plt.grid(),plt.ylim(-0.5,0.5)
            print("eposide:{},alpha:{},score:{}".format(epoch,alpha.exp().item(),test_result))

    return X_best,THETA_best,SCORE_BEST,ALPHA,LOSS,ACTION_best

while 1:
    X_best,THETA_best,SCORE_BEST,ALPHA,LOSS,ACTION = train()
    print(X_best)
    print(THETA_best)
    print(ALPHA)
    plt.subplot()
    plt.plot(ACTION)
    addr = "1.txt"
    write_txt(addr,LOSS)

test_result, X, THETA, ACTION = test()
print(test_result)
plt.figure(dpi=320)
plt.plot(X),plt.grid()
plt.figure()
plt.plot(THETA),plt.grid()
plt.figure()
plt.plot(ACTION),plt.grid()

# def testf():
#     state,_ = env.reset()

#     reward_sum = 0
#     X = []
#     THETA = []
#     over = False
#     ACTION = []
#     t = 0
#     while not over:
#         t = t+1
#         action = get_action(state)
#         if abs(t-1100) <= 100:
#             f = 1
#         else:
#             f = 0

#         state,reward,over,_ = env.step(action+f,1)
#         X.append(state[0])
#         THETA.append(state[2])
#         ACTION.append(action)
#         reward_sum += reward

#     return reward_sum,X,THETA,ACTION

# test_result, X, THETA, ACTION = testf()
# print(test_result)
# plt.figure(dpi=320)
# plt.plot(X),plt.grid()
# plt.figure()
# plt.plot(THETA),plt.grid()
# plt.figure()
# plt.plot(ACTION),plt.grid()

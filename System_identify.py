from FBLS import FBLS
import numpy as np
import argparse
import os
from time import time
import matplotlib.pyplot as plt
def get_test_data(data_num=200):
    y0, y1 = 1, 1
    y = [y0, y1]
    u = lambda n: np.sin(2 * np.pi * n / 25)
    u_data = [u(n) for n in range(1, data_num - 1)]

    for n in range(2, data_num):
        yn = (y[n - 1] * y[n - 2] * (y[n - 1] + 2.5)) / (1 + y[n - 1]**2 + y[n - 2]**2) + u(n - 1)
        y.append(yn)

    y = np.array(y).reshape(-1, 1)
    u_data = np.array(u_data).reshape(-1, 1)
    X = np.hstack((y[:data_num - 2], y[1:data_num - 1], u_data))
    Y = y[2:data_num]
    return X, Y
def get_train_noisy_data(data_num=5000,noise = 0.2):
    y0, y1 = 1, 1
    y = [y0, y1]
    #数据分分布选择非常重要
    #u = np.linspace(-2,2,data_num)
    u = np.random.uniform(-2,2,size=(data_num,))
    

    for n in range(2, data_num):
        yn = (y[n - 1] * y[n - 2] * (y[n - 1] + 2.5)) / (1 + y[n - 1]**2 + y[n - 2]**2) + u[n-1]+np.random.uniform(-noise,noise)
        y.append(yn)

    y = np.array(y).reshape(-1, 1)
    u = np.array(u).reshape(-1, 1)
    X = np.hstack((y[:data_num - 2], y[1:data_num - 1], u[1:data_num - 1]))
    Y = y[2:data_num]
    return X, Y
def get_train_data(data_num=1000):
    y0, y1 = 1, 1
    y = [y0, y1]
    #u = np.linspace(-2,2,data_num)
    u = np.random.uniform(-2,2,size=(data_num,))
    for n in range(2, data_num):
        yn = (y[n - 1] * y[n - 2] * (y[n - 1] + 2.5)) / (1 + y[n - 1]**2 + y[n - 2]**2) + u[n - 1]
        y.append(yn)

    y = np.array(y).reshape(-1, 1)
    u = np.array(u).reshape(-1, 1)
    X = np.hstack((y[:data_num - 2], y[1:data_num - 1], u[1:data_num - 1]))
    Y = y[2:data_num]
    return X, Y


# 创建 ArgumentParser 对象
parser = argparse.ArgumentParser(description="FBLS 参数设置")
# 添加命令行参数
parser.add_argument('--fuzz_sys', type=int, required=True, help='模糊系统数数目')
parser.add_argument('--fuzz_rule', type=int, required=True, help='模糊规则数目')
parser.add_argument('--enhance_node', type=int, required=True, help='增强节点数目')
# 解析命令行参数
args = parser.parse_args()

X_train,Y_train = get_train_data(5000)
X_test,Y_test = get_test_data()
net = FBLS(args.fuzz_sys,args.fuzz_rule,args.enhance_node)
start = time()
net.train(X_train,Y_train)
train_time = time()  - start
Y_pred = net.predict(X_test)
test_time = time() - start - train_time

error = Y_test - Y_pred
rmse = np.sqrt(np.mean(error**2,axis = 0))
plt.plot(Y_test,label='true')
plt.plot(Y_pred,label='pred')
plt.legend()
images_dir = os.path.join(os.path.dirname(__file__), 'Images')
os.makedirs(images_dir, exist_ok=True)  # 确保目录存在
save_path = os.path.join(images_dir, './fbls.png')
plt.savefig(save_path)
print('rmse:',rmse)
print('训练时间:',train_time)
print('测试时间:',test_time)
print(save_path)
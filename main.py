# -*- coding: utf-8 -*-
# @Time    : 2020-05-28 18:10
# @Author  : zxl
# @FileName: main.py

import numpy as np
from sklearn.metrics import accuracy_score

def define_parameter(S,M):
    """
    自定义参数
    :param S: 状态空间数目 0 - S-1
    :param M: 观测空间数目 0 - M-1
    :return: 初始化向量pi,状态转移矩阵A,输出概率矩阵B
    """
    # pi = np.array([0.25,0.25,0.25,0.25])
    # A = np.array([[0,1,0,0],[0.4,0,0.6,0],
    #               [0,0.4,0,0.6],[0,0,0.5,0.5]])
    # B = np.array([[0.5,0.5],[0.3,0.7],[0.6,0.4],[0.8,0.2]])
    pi = np.random.random(size = (S,))
    pi = pi / sum(pi)
    A = np.random.random(size = (S,S))
    for i in range(S):
        A[i] = A[i] / sum(A[i])
    B = np.random.random(size = (S,M))
    for i in range(S):
        B[i] = B[i] / sum(B[i])

    return pi,A,B

def initia_parameter(S,M):
    return define_parameter(S,M)
    # pi = np.array([0.2,0.4,0.4])
    # A = np.array([[0.5,0.2,0.3],[0.3,0.5,0.2],[0.2,0.3,0.5]])
    # B = np.array([[0.5,0.5],[0.4,0.6],[0.7,0.3]])
    # return pi, A, B

def my_random(arr):
    """
    按照arr的分布，选取一个
    :param arr:
    :return:
    """
    r = np.random.rand()
    c = 0
    for i in range(len(arr)):
        c += arr[i]
        if c >= r:
            return i


def generate_data(pi,A,B,n,l):
    """

    :param pi: 初始化概率
    :param A: 状态转移矩阵
    :param B: 输出概率矩阵
    :param n: 生成样本数目
    :param l: 生成样本序列长度
    :return:
    """
    samples = []
    state = []
    for i in range(n):
        cur_samples = []
        cur_state = []
        hidden_state = pi
        while len(cur_samples) < l:
            h = my_random(hidden_state)
            o = my_random(B[h])
            cur_samples.append(o)
            cur_state.append(h)
            hidden_state = A[h]
        samples.append(cur_samples)
        state.append(cur_state)
    return np.array(samples), np.array(state)


def cal_alpha(pi,A,B,x_arr):
    """
    前向算法
    :param pi:
    :param A:
    :param B:
    :param o_arr:观测到的x序列
    :return: alpha_matri l * S
    """
    alpha_matri = []
    S = len(pi) #状态个数
    l = len(x_arr) #观测到序列长度
    cur_alpha = [pi[i]*B[i,x_arr[0]] for i in range(S)]
    alpha_matri.append(cur_alpha)
    while len(alpha_matri) < l:
        cur_idx = len(alpha_matri)
        o = x_arr[cur_idx]
        cur_alpha = []
        for cur_state in range(S):
            p = 0
            for last_state in range(S):
                p += alpha_matri[-1][last_state] * A[last_state,cur_state]
            cur_alpha.append(p * B[cur_state,o])
        alpha_matri.append(cur_alpha)
    return np.array(alpha_matri)


def cal_beta(pi,A,B,x_arr):
    """
    后向算法
    :param pi:
    :param A:
    :param B:
    :param x_arr:
    :return: l * S
    """
    beta_matri = []
    S = len(pi)
    l = len(x_arr)
    cur_beta = [1 for i in range(S)]
    beta_matri.insert(0,cur_beta)
    while len(beta_matri) < l:
        cur_idx = l - len(beta_matri)
        next_o = x_arr[cur_idx]
        cur_beta = []
        for cur_state in range(S):
            p = 0
            for next_state in range(S):
                p += A[cur_state,next_state] * B[next_state,next_o] * beta_matri[0][next_state]
            cur_beta.append(p)
        beta_matri.insert(0,cur_beta)
    return np.array(beta_matri)

def cal_gama(alpha_matri,beta_matri):
    """

    :param alpha_matri:
    :param beta_matri:
    :return: gama_matri l * S
    """
    l,S = alpha_matri.shape
    gama_matri = np.zeros(shape=(l,S))
    for t in range(l):
        for s in range(S):
            gama_matri[t,s] = alpha_matri[t,s] * beta_matri[t,s]/\
            (np.dot(alpha_matri[t,:],beta_matri[t,:]))
    return gama_matri

def cal_zeta(A,B,alpha_matri,beta_matri,x_arr):
    """
    :param A: 状态转移矩阵
    :param B: 输出概率矩阵
    :param alpha_matri: 前向矩阵
    :param beta_matri: 后向矩阵
    :param x_arr: 观测到的向量
    :return: gama_tensor l * S * S
    """

    zeta_tensor = []
    l = len(x_arr)
    N = len(alpha_matri[0])

    for t in range(l-1):
        zeta_matri = np.zeros(shape = (N,N))
        for i in range(N):
            for j in range(N):
                if t == l-1:
                    b = 1
                    cur_beta = 1
                else:
                    b = B[j,x_arr[t+1]]
                    cur_beta = beta_matri[t+1,j]
                r = alpha_matri[t,i] * A[i,j] * b * cur_beta
                zeta_matri[i,j] = r
                deno = sum(sum(
                    alpha_matri[t,i1] * A[i1,j1] * B[j,x_arr[t+1]] *beta_matri[t+1,j1]
                    for j1 in range(N))
                      for i1 in range(N)
                )
                zeta_matri[i,j] /= deno
        zeta_tensor.append(zeta_matri)
    return np.array(zeta_tensor)

def BaumWelch(O,S,M,epoch):
    """
    使用BaumWelch算法学习模型
    :param O:
    :param S:
    :param M:
    :param epoch:
    :return:
    """
    pi_tensor = []
    A_tensor = []
    B_tensor = []
    for x_arr in O:
        pi, A, B = initia_parameter(S, M)
        for i in range(epoch):
            l = len(x_arr) #序列长度
            alpha_matri = cal_alpha(pi, A, B,x_arr)
            beta_matri = cal_beta(pi,A,B,x_arr)
            gama_matri = cal_gama(alpha_matri,beta_matri)
            zeta_tensor = cal_zeta(A,B,alpha_matri,beta_matri,x_arr)
            new_A = np.zeros(shape = A.shape)
            new_B = np.zeros(shape = B.shape)
            new_pi = np.zeros(shape = pi.shape)
            for i in range(len(new_A)):
                for j in range(len(new_A[0])):
                    new_A[i,j] = sum(zeta_tensor[:l-1,i,j])/\
                                 sum(gama_matri[:l-1,i])
            for i in range(len(pi)):
                new_pi[i] = gama_matri[0,i]

            for j in range(len(B)):
                for k in range(len(B[0])):
                    vk = k #第k个输出值
                    idx = np.argwhere(x_arr == vk)
                    r1 = sum(gama_matri[idx,j])
                    r2 = sum(gama_matri[:,j])
                    new_B[j,k] = r1/r2
            A = new_A
            B = new_B
            pi = new_pi
        pi_tensor.append(pi)
        A_tensor.append(A)
        B_tensor.append(B)
    pi = np.mean(np.array(pi_tensor),axis = 0)
    A = np.mean(np.array(A_tensor),axis = 0)
    B = np.mean(np.array(B_tensor),axis = 0)
    return pi, A, B

def inference_state (x_arr,pi,A,B):
    """
    Viterbi Algorithm: 推理观测对应隐层状态
    :param x_arr:
    :param pi:
    :param A:
    :param B:
    :return:
    """
    state_lst = []
    S = len(pi)  # 状态个数
    l = len(x_arr)  # 观测到序列长度

    X = []

    for i in range(S):
        r = pi[i] * B[i, x_arr[0]]
        X.append(r)

    best_state_matri = []
    for n in np.arange(1,l,1):
        new_X = []
        o = x_arr[n]
        cur_state_lst = []
        for cur_state in range(S):
            max_val = -1
            max_state = 0
            for last_state in range(S): #找一个最好的上一个state
                r = A[last_state,cur_state] * X[last_state]
                if r > max_val:
                    max_val = r
                    max_state = last_state
            new_X.append(max_val * B[cur_state,o])
            cur_state_lst.append(max_state)
        X = new_X
        best_state_matri.append(cur_state_lst)
    state_lst.insert(0,np.argwhere(X==max(X))[0][0] )
    for i in np.arange(len(best_state_matri)-1,-1,-1):
        best_state = best_state_matri[i][state_lst[0]]
        state_lst.insert(0,best_state)
    return state_lst


def predict_next_output(x_arr,pi,A,B):
    S = len(pi)#状态数目
    M = len(B[0]) # 输出数目
    alpha_matri = cal_alpha(pi,A,B,x_arr)
    max_val = -1
    best_o = 0
    for o in range(M):
        p = 0.0
        for next_state in range(S):
            ps = 0.0
            for cur_state in range(S):
                p += alpha_matri[-1,cur_state] * A[cur_state,next_state]
            ps *= B[next_state,o]
            p += ps
        if p > max_val:
            max_val = p
            best_o = o
    return best_o



if __name__ == "__main__":

    S = 4
    M = 2
    n_samples = 100
    l = 10

    pi, A, B = define_parameter(S,M)
    print(pi)
    dataset,state = generate_data(pi,A,B,n_samples,l)
    n_train =int(len(dataset) * 0.7)
    # n_train = 1
    trainset = dataset[:n_train]
    testset = dataset[n_train:]


    pred_pi, pred_A, pred_B = BaumWelch(trainset,S,M,20)
    print(pred_pi)
    print(A)
    print(pred_A)
    print(B)
    print(pred_B)

    pred_lst = []
    y_labels = []
    for arr in testset :
        y_labels.append(arr[-1])
        pred = predict_next_output(arr[:-1],pred_pi,A,B)
        pred_lst.append(pred)
    acc =accuracy_score(y_labels,pred_lst)
    print("预测下一个观测值, acc: %.3f"%acc)

    true_state_matri = state[n_train:]
    pred_state_matri = []
    for arr in testset:
        state_lst =inference_state(arr,pi,A,B)
        pred_state_matri.append(state_lst)
    acc = accuracy_score(true_state_matri.flatten(),np.array(pred_state_matri).flatten())
    print("预测隐状态序列, acc: %.3f"%acc)








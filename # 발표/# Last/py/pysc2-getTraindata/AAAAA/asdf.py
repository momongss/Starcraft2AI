"""
1 [[117, 102]]
2 [[0], [54, 48]]
3 [[0], [55, 48], [87, 80]]
7 [[0]]
13 [[0], [13, 14]]
42 [[0], [28, 58]]
91 [[0], [78, 24]]
451 [[1], [89, 69]]
477 [[0]]
490 [[0]]
"""

import numpy as np

transform_list = [1, 2, 3, 7, 13, 42, 91, 451, 477, 490]


def LtoS(L_action):
    if L_action == 1:
        S_action = 0
    if L_action == 2:
        S_action = 1
    if L_action == 3:
        S_action = 2
    if L_action == 7:
        S_action = 3
    if L_action == 13:
        S_action = 4
    if L_action == 42:
        S_action = 5
    if L_action == 91:
        S_action = 6
    if L_action == 451:
        S_action = 7
    if L_action == 477:
        S_action = 8
    if L_action == 490:
        S_action = 9

    net_action = [0 for _ in range(10)]
    net_action[S_action] = 1

    return net_action


def StoL(net_action):
    S_action = np.argmax(net_action)
    L_action = transform_list(S_action)

    return L_action
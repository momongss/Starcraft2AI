def model_predict(self, availabel_action):
    # model.py로 옮겨서 사용할 model_predict

    # 모든 네트워크 출력
    Action_Type = []
    Queue = []  # size 2
    Selected_Units = []  # size 500
    Target_Unit = []  # size 19
    Target_Point = []  # size 592

    actionlist = [Action_Type[i] for i in availabel_action]

    action = availabel_action[np.argmax(actionlist)]
    actionID = self.reverse_list[action]

    # Queue (2,)
    # Selected_Units (500,)
    # Target_Unit (4,), (5,), (10,)
    # Target_Point (128,128), (84,84), (84,84)

    if actionID == 0:  # [(4,), (500,)]
        arg1 = np.argmax(Target_Unit[:4])
        arg2 = np.argmax(Selected_Units)
        arg = [(arg1,), (arg2,)]
        return action, arg
    if actionID == 1:  # [(128, 128)]
        arg1 = np.argmax(Target_Point[:128])
        arg2 = np.argmax(Target_Point[128:256])
        arg = [(arg1, arg2)]
        return action, arg
    if actionID == 2:  # [(2,), (84, 84)]
        arg1 = np.argmax(Queue)
        arg2 = np.argmax(Target_Point[256:340])
        arg3 = np.argmax(Target_Point[340:424])
        arg = [(arg1,), (arg2, arg3)]
        return action, arg
    if actionID == 3:  # [(2,), (128, 128)]
        arg1 = np.argmax(Queue)
        arg2 = np.argmax(Target_Point[:128])
        arg3 = np.argmax(Target_Point[128:256])
        arg = [(arg1,), (arg2, arg3)]
        return action, arg
    if actionID == 4:  # [(4,)]
        arg1 = np.argmax(Target_Unit[:4])
        arg = [(arg1,)]
        return action, arg
    if actionID == 5:  # [(2,)]
        arg1 = np.argmax(Queue)
        arg = [(arg1,)]
        return action, arg
    if actionID == 6:  # [(4,), (84, 84)]
        arg1 = np.argmax(Target_Unit[:4])
        arg2 = np.argmax(Target_Point[256:340])
        arg3 = np.argmax(Target_Point[340:424])
        arg = [(arg1,), (arg2, arg3)]
        return action, arg
    if actionID == 7:  # [(500,)]
        arg1 = np.argmax(Selected_Units)
        arg = [(arg1,)]
        return action, arg
    if actionID == 8:  # []
        arg = []
        return action, arg
    if actionID == 9:  # [(5,), (10,)]
        arg1 = np.argmax(Target_Unit[4:9])
        arg2 = np.argmax(Target_Unit[9:19])
        arg = [(arg1,), (arg2,)]
        return action, arg
    if actionID == 10:  # [(2,), (84, 84), (84, 84)]
        arg1 = np.argmax(Queue)
        arg2 = np.argmax(Target_Point[256:340])
        arg3 = np.argmax(Target_Point[340:424])
        arg4 = np.argmax(Target_Point[424:508])
        arg5 = np.argmax(Target_Point[508:592])
        arg = [(arg1,), (arg2, arg3), (arg4, arg5)]
        return action, arg
    if actionID == 11:  # [(10,)]
        arg1 = np.argmax(Target_Unit[9:19])
        arg = [(arg1,)]
        return action, arg

    return
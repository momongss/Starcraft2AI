def arguments_processing(self, func_id, args_Qs):
    data_2 = np.argmax(args_Qs[0][0:2])
    data_4 = np.argmax(args_Qs[0][2:6])
    data_5 = np.argmax(args_Qs[0][6:11])
    data_10 = np.argmax(args_Qs[0][11:21])
    data_500 = np.argmax(args_Qs[0][21:521])
    data_84_1_0 = np.argmax(args_Qs[0][521:605])
    data_84_1_1 = np.argmax(args_Qs[0][605:689])
    data_84_2_0 = np.argmax(args_Qs[0][689:773])
    data_84_2_1 = np.argmax(args_Qs[0][773:857])
    data_64_0 = np.argmax(args_Qs[0][857:921])
    data_64_1 = np.argmax(args_Qs[0][921:985])

    data_dic = {
        (2,): data_2,
        (4,): data_4,
        (5,): data_5,
        (10,): data_10,
        (500,): data_500,
        (84, 84): [[data_84_1_0, data_84_1_1], [data_84_2_0, data_84_2_1]],
        (64, 64): [data_64_0, data_64_1]
    }

    flag_84 = False
    arguments = []
    for arg in self.action_spec.functions[func_id].args:
        if arg.sizes == (84, 84):
            if not flag_84:
                arguments.append(data_dic[arg.sizes][0])
                flag_84 = True
            else:
                arguments.append(data_dic[arg.sizes][1])
        else:
            if arg.sizes == (64, 64):
                arguments.append(data_dic[arg.sizes])
            else:
                arguments.append([data_dic[arg.sizes]])

    return arguments

def train_network(self, input_data, available_actions):
    func_input_data = np.array(input_data).reshape(1, len(input_data))
    full_Qs = self.full_network.predict(func_input_data)
    # Qs = self.function_network.predict(func_input_data)
    func_Qs = full_Qs[0][:573]
    functions = func_Qs
    for i in range(len(functions)):
        if i not in available_actions:
            functions[i] = 0
    args_Qs = [full_Qs[0][573:]]
    func_id = np.argmax(func_Qs)
    arguments = self.arguments_processing(func_id, args_Qs)

    return func_id, arguments
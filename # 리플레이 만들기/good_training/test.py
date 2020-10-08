import pickle

with open('good_replay.txt', 'rb') as fk:
    replay_data = pickle.load(fk)

ID = 0
ARGS = 1
OBS = 2

count = {}
action = set()

for game_loop in sorted(replay_data.keys()):
    print(game_loop, replay_data[game_loop][ID], replay_data[game_loop][ARGS])
    id = replay_data[game_loop][ID]
    args = replay_data[game_loop][ARGS]
    if len(replay_data[game_loop][ARGS][0]) == 1:
        if id not in count:
            count[id] = set([replay_data[game_loop][ARGS][0][0]])
        else:
            count[id].add(replay_data[game_loop][ARGS][0][0])
    action.add(replay_data[game_loop][ID])

print(action)
print(len(action))

# 2 : arg1 : 0,2,3 screen
# 451 : arg1 : 1 screen
# 3 : arg1 : 0, 1 screen rect
# 7 : arg : [[0]]
# 490 : arg : [[0]]
# 42 : arg1 : [[0]], arg2 : screen
# 13 : arg1 : [[0]], arg2 : minimap
# 91 : arg1 : [[0]], arg2 : screen
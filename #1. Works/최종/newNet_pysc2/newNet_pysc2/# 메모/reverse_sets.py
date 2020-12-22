s = set()

reverse_list = [0 for _ in range(573)]
reverse_dict = {"[(4,),(500,)]": 0,
"[(64,64)]": 1,
"[(2,),(84,84)]": 2,
"[(2,),(64,64)]":3,
"[(4,)]":4,
"[(2,)]":5,
"[(4,),(84,84)]":6,
"[(500,)]":7,
"[]":8,
"[(5,),(10,)]":9,
"[(2,),(84,84),(84,84)]":10,
"[(10,)]":11}

while True:
    inp = input().strip().split()
    inp2 = inp.copy()
    if not inp:
        break
    if len(inp) > 1:
        inp = inp[1:]
    elif len(inp) == 1:
        inp = inp[0]
    string = ''
    for p in inp:
        string += p
    reverse_list[int(inp2[0])] = reverse_dict[string]

print(reverse_list)
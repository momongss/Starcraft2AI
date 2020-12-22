import shutil
import os

store_dir = 'C:\\Users\\rkrp1\\Desktop\\종합설계\\#5. 리플레이'
count_dir = store_dir + '\\' + 'count.txt'

def search(dirname):
    filenames = os.listdir(dirname)
    for name in filenames:
        if name.endswith('.SC2Replay'):
            with open(count_dir, "r") as f:
                count = int(f.read().strip())
            with open(count_dir, "w") as f:
                f.write(str(int(count)+1))
            shutil.copy(dirname+'\\'+name, store_dir+'\\'+str(count)+'.SC2Replay')
        elif os.path.isdir(dirname+'\\'+name):
            search(dirname+'\\'+name)

search("C:\\Users\\rkrp1\\Desktop\\종합설계\\#1. Works")
# shutil.copy()
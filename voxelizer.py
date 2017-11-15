import glob
import os
import subprocess
import sys

voxsize = int(sys.argv[1])
paths = glob.glob("{}/*.obj".format(sys.argv[2]))
print("number of data:", len(paths))

with open(os.devnull, 'w') as devnull:
    for i, path in enumerate(paths):
        cmd = "./binvox -d {0} -cb -e {1}".format(voxsize, path)
        ret = subprocess.check_call(cmd.split(' '), stdout=devnull, stderr=devnull)
        if ret != 0:
            print("error", i, path)
        else:
            print(i+1, path)

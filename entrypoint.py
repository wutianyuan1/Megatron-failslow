import os
import sys
import subprocess


os.chdir("/workspace/Megatron-failslow")
MASTER_ADDR = '172.31.16.182'
cmd = "MASTER_ADDR={} WORLD_SIZE={} RANK={} python run_training.py --tp 1 --pp 1 --probe 0 --hsize 1024"
cmd = cmd.format(MASTER_ADDR, 4, int(sys.argv[1]))
train_proc = subprocess.Popen(cmd, shell=True)

os.chdir("/workspace/ncclprobe/injection")
cmd2 = "MASTER_ADDR={} WORLD_SIZE={} RANK={} MASTER_PORT=12345 python injection.py"
cmd2 = cmd2.format(MASTER_ADDR, 4, int(sys.argv[1]))
inject_proc = subprocess.Popen(cmd2, shell=True)

while True:
    try:
        pass
    except KeyboardInterrupt:
        print("!!!! stop")
        train_proc.terminate()
        inject_proc.terminate()

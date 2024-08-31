import os
import sys
import subprocess
import time


MASTER_ADDR = '10.22.4.147'
nnodes = 8

os.chdir("/workspace/Megatron-failslow")
cmd = "MASTER_ADDR={} WORLD_SIZE={} RANK={} python run_training.py --tp 1 --pp 4 --probe 0 --hsize 3072"
cmd = cmd.format(MASTER_ADDR, nnodes, int(sys.argv[1]))
train_proc = subprocess.Popen(cmd, shell=True)

os.chdir("/workspace/ncclprobe/injection")
cmd2 = "MASTER_ADDR={} WORLD_SIZE={} RANK={} MASTER_PORT=12345 python injection.py"
cmd2 = cmd2.format(MASTER_ADDR, nnodes, int(sys.argv[1]))
inject_proc = subprocess.Popen(cmd2, shell=True)

print('='*80)
while True:
    try:
        time.sleep(1)
    except KeyboardInterrupt:
        print("!!!! stop")
        train_proc.terminate()
        inject_proc.terminate()
        time.sleep(3)
        train_proc.kill()
        inject_proc.kill()
        break

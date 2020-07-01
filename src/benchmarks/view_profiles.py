import os
import subprocess
import multiprocessing as mp


folder = "profiles"
date = "200630"
# time = "12.24.32"   # pre optimization
time = "23.23.54"   # post optimization
files = os.listdir(f'profiles/{date}/{time}')
command = f"snakeviz {folder}/{date}/{time}/" + "{}"
commands = list(map(command.format, files))


if __name__ == '__main__':
    pool = mp.Pool(processes=mp.cpu_count())
    pool.map(subprocess.run, commands)
    pool.close()
    pool.join()
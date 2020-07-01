import os
import subprocess
import multiprocessing as mp


folder = "profiles"

date = "200630"   # pre optimization
time = "12.24.32"   # pre optimization

# date = "200701"     # post optimization
# time = "10.52.41"   # post optimization

files = os.listdir(f'profiles/{date}/{time}')
command = f"snakeviz {folder}/{date}/{time}/" + "{}"
commands = list(map(command.format, files))


if __name__ == '__main__':
    pool = mp.Pool(processes=mp.cpu_count())
    pool.map(subprocess.run, commands)
    pool.close()
    pool.join()

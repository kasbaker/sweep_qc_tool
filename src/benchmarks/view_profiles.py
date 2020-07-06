import os
import subprocess
import multiprocessing as mp


folder = "profiles"

# date = "200630"   # pre optimization
# time = "12.24.32"   # pre optimization

# date = "200701"     # post sweep_data optimization
# time = "10.52.41"   # post sweep_data optimization

date = "200701"     # post single_plotter optimization
time = "18.27.04"   # post single_plotter optimization

files = os.listdir(f'profiles/{date}/{time}')
command = f"snakeviz {folder}/{date}/{time}/" + "{}"
commands = list(map(command.format, files))


if __name__ == '__main__':
    pool = mp.Pool(processes=mp.cpu_count())
    pool.map(subprocess.run, commands)
    pool.close()
    pool.join()

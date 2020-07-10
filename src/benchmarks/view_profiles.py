import os
import subprocess
import multiprocessing as mp


folder = "profiles"

date = "200630"   # pre optimization
time = "12.24.32"   # pre optimization

# date = "200701"     # post sweep_data optimization
# time = "10.52.41"   # post sweep_data optimization

# date = "200701"     # post single_plotter optimization
# time = "18.27.04"   # post single_plotter optimization

# date = "200706"     # post single_plotter optimization w/ new C2
# time = "12.32.40"   # post single_plotter optimization w/ new C2

# date = "plotter/200706"     # multi plotter benchmarks
# time = "14.42.57"   # multi plotter bencharks

# date = "plotter/200706"     # multi plotter benchmarks
# time = "18.25.36"         # multi plotter bencharks

# date = "plotter/200710"     # multi plotter benchmarks
# time = "10.19.44"         # multi plotter bencharks

# date = "plotter/200710"     # multi plotter benchmarks
# time = "10.39.30"         # multi plotter bencharks

files = os.listdir(f'profiles/{date}/{time}')
command = f"snakeviz {folder}/{date}/{time}/" + "{}"
commands = list(map(command.format, files))


if __name__ == '__main__':
    pool = mp.Pool(processes=mp.cpu_count())
    pool.map(subprocess.run, commands)
    pool.close()
    pool.join()

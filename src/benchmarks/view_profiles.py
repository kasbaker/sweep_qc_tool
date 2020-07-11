import webbrowser
from pathlib import Path
import datetime as dt
import subprocess
import multiprocessing as mp
from tkinter import filedialog
from tkinter import *


if __name__ == '__main__':
    root = Tk()
    root.withdraw()

    base_dir = Path(__file__).parent

    today = dt.datetime.now().strftime('%y%m%d')
    folder_selected = filedialog.askdirectory(initialdir=base_dir.joinpath('profiles', today))

    files = list(Path(folder_selected).glob("*.prof"))

    command = "snakeviz {}"
    commands = list(map(command.format, files))

    pool = mp.Pool(processes=len(commands))
    pool.map(subprocess.run, commands)
    pool.close()
    pool.join()

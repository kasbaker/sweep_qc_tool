from pathlib import Path
import datetime as dt
import subprocess
from multiprocessing import Pool

from tkinter import filedialog
from tkinter import *


if __name__ == '__main__':
    root = Tk()
    root.withdraw()

    base_dir = Path(__file__).parent

    today = dt.datetime.now().strftime('%y%m%d')
    folder_selected = filedialog.askdirectory(initialdir=base_dir.joinpath('profiles', today))

    files = list(Path(folder_selected).glob("*.prof"))

    commands = list(map("snakeviz {}".format, files))

    with Pool(processes=len(commands)) as pool:
        pool.map(subprocess.run, commands)
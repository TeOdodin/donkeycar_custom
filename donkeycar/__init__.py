import sys
from pyfiglet import Figlet

__version__ = '4.1.0'
f = Figlet(font='speed')

print(f.renderText('Donkey Car'))
print(f'using donkey v{__version__} ...')

if sys.version_info.major < 3 or sys.version_info.minor < 6:
    msg = f'Donkey Requires Python 3.6 or greater. You are using {sys.version}'
    raise ValueError(msg)

# The default recursion limits in CPython are too small.
sys.setrecursionlimit(10**5)

from .vehicle import Vehicle
from .memory import Memory
from . import utils
from . import config
from . import contrib
from .config import load_config
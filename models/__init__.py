from os import listdir
from os.path import isfile, join

# # Linter does not recognise the imports well dynamically
# path = "models"
# __all__ = [f[:-3] for f in listdir(path) if isfile(join(path, f)) 
#             and f != "__init__.py" and f[-3:] == ".py"]

# # Please add all model module names
# __all__ = ["testModel", "model2"]

from .base import *
from .baseOld import *
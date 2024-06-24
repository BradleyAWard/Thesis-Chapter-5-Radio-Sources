# --------------------------------------------------
# Imports
# --------------------------------------------------

import os
from pandas import read_csv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --------------------------------------------------
# Full Data Loader
# --------------------------------------------------

def full_loader(file_name, root=ROOT):
    """
    Loads a full database from file
    
    :param file_name: The name of the file
    :param root: The root directory
    :return: Data frame
    """

    file_path = root + '/data/' + str(file_name)
    data = read_csv(file_path)
    return data

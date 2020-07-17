""" Misc util functions. """

import os

def clean_hcp_mac_files():
    """ Deletes all the useless hidden files in the data folder. """
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    for root, dirs, files in os.walk(data_dir, topdown=False):
        for name in files:
            if name.startswith('.'):
                os.remove(os.path.join(root, name))

clean_hcp_mac_files()
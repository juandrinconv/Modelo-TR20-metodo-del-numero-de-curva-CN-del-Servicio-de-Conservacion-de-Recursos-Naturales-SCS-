"""Functions for executing system commands for Hydra

:copyright: 2016, See AUTHORS for more details
:license: GNU General Public License, See LICENSE for more details.

"""

import sys
import os


def open_file_with_default(filename):
    """Open a file, externally, given a valid filename.

    Args:
        filename (str): The name of the file to open.

    Raises:
        EnvironmentError: if the current OS is not valid.
        IOError: if the specified filename does not have an associated file.

    """
    if os.path.isfile(filename):
        if sys.platform == 'win32':                                # Windows
            os.system('start ' + filename)
        elif sys.platform == 'linux' or sys.platform == 'linux2':  # Linux
            os.system('xdg-open ' + filename)
        elif sys.platform == 'darwin':                             # OS X
            os.system('open ' + filename)
        else:
            raise EnvironmentError
    else:
        raise IOError

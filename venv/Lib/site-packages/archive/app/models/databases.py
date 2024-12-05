"""Functions for communicating with a valid sqlite database file.

:copyright: 2016, See AUTHORS for more details
:license: GNU General Public License, See LICENSE for more details.

"""

import peewee as pw


database = pw.SqliteDatabase(None)


def open_file(filename):
    """Open a valid sqlite database and initializes it.

    Args:
        filename (str): The name of the file to open

    """
    database.init(filename)


def connect():
    """Connect to the initialized sqlite database.

    Raises:
        Exception: If valid sqlite database is not initialized.

    """

    try:
        database.connect()
    except Exception:
        raise Exception('Failure to connect to database file')

import unittest

import peewee as pw
from playhouse.test_utils import test_database

from archive.app import models
from archive.app.models import databases as db
from pyflo import build

test_db = pw.SqliteDatabase(':memory:')


class DatabaseTest(unittest.TestCase):

    def test_no_connection(self):
        self.assertRaises(Exception, db.connect)


class ModelTest(unittest.TestCase):

    def test_pipes_ordered_up_from_node(self):
        with test_database(test_db, (models.Node, models.Section, models.Pipe)):
            s102 = models.Node.create(name='S-102', elevation=10.0)
            s101 = models.Node.create(name='S-101', elevation=9.00)
            o1_1 = models.Node.create(name='OUT-1.1', elevation=8.0)
            rc18 = models.Section.create(name='RCP 18"', span=18.0, rise=10.0, mannings=0.012, shape=models.CIRCLE)
            p101 = models.Pipe.create(node_1=s101, node_2=s102, invert_1=8.0, invert_2=7.0, length=300.0, section=rc18)
            p102 = models.Pipe.create(node_1=s102, node_2=o1_1, invert_1=7.0, invert_2=6.0, length=300.0, section=rc18)
            pipes = models.Pipe.select()
            pipes = build.pipes_ordered_up_from_node(o1_1, pipes)
            self.assertEqual(pipes, [p102, p101])

    def test_pipes_ordered_down_to_node(self):
        with test_database(test_db, (models.Node, models.Section, models.Pipe)):
            s102 = models.Node.create(name='S-102', elevation=10.0)
            s101 = models.Node.create(name='S-101', elevation=9.00)
            o1_1 = models.Node.create(name='OUT-1.1', elevation=8.0)
            rc18 = models.Section.create(name='RCP 18"', span=18.0, rise=10.0, mannings=0.012, shape=models.CIRCLE)
            p101 = models.Pipe.create(node_1=s101, node_2=s102, invert_1=8.0, invert_2=7.0, length=300.0, section=rc18)
            p102 = models.Pipe.create(node_1=s102, node_2=o1_1, invert_1=7.0, invert_2=6.0, length=300.0, section=rc18)
            pipes = models.Pipe.select()
            pipes = build.pipes_ordered_down_to_node(o1_1, pipes)
            self.assertEqual(pipes, [p101, p102])

import time
import unittest

from archive import iterators


class SpeedTest(unittest.TestCase):

    def setUp(self):
        self.startTime = time.time()
        self.test_data = [i for i in range(100)]

    def tearDown(self):
        t = time.time() - self.startTime
        print("%s: %.6fs" % ('speed', t))

    def test_current(self):
        for pair in iterators.pairwise(self.test_data):
            # print(pair)
            pass

    # def test_generator(self):
    #     for pair in iterators.gen_pair(self.test_data):
    #         pass

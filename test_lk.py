import unittest
from lk import *
import numpy as np

class GradientsTestCase(unittest.TestCase):
    def test_gradient_x(self):
        block = np.array([[2, 3], [1, 2]])

        assert gradient_x(block) == 2

    def test_gradient_y(self):
        block = np.array([[2, 3], [1, 5]])

        assert gradient_y(block) == -1

    def test_gradient_t(self):
        block1 = np.array([[2, 3], [1, 2]])
        block2 = np.array([[0, 1], [0, 1]])

        assert gradient_t(block1, block2) == -6

    def test_estimate_gradient(self):
        block1 = np.array([[2, 3], [1, 2]])
        block2 = np.array([[0, 1], [0, 1]])

        (ix, iy, it) = estimate_gradients(block1, block2)

        print(ix)
        print(iy)
        print(it)

        assert ix == 1
        assert iy == 0.5
        assert it == -1.5

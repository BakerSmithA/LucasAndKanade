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

        ix, iy, it = estimate_gradients(block1, block2)

        print(ix)
        print(iy)
        print(it)

        assert ix == 1
        assert iy == 0.5
        assert it == -1.5


    def test_generate_all_gradients(self):
        segment = np.array(
            [[1, 2, 3],
             [0, 1, 2],
             [0, 1, 2]]
        )

        next_segment = np.array(
            [[0, 0, 1],
             [0, 0, 1],
             [0, 0, 0]]
        )

        gs = generate_all_gradients(segment, next_segment)

        g1 = Gradient(ix=2/4, iy=2/4, it=-4/4)
        g2 = Gradient(ix=4/4, iy=2/4, it=-6/4)
        g3 = Gradient(ix=2/4, iy=0/4, it=-2/4)
        g4 = Gradient(ix=3/4, iy=1/4, it=-5/4)

        assert gs == [g1, g2, g3, g4]
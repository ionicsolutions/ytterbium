import unittest
import itertools

import numpy as np

from ..polarization import normalize


class TestSystem(unittest.TestCase):

    def test_vector_has_unit_length(self):
        for vec in itertools.product(range(20), [-4, -3], [0.33, 0.57]):
            polarization = normalize(vec)
            length = np.sqrt(np.sum(np.power(polarization, 2)))
            self.assertTrue(abs(length - 1.0) < 1e-6)

    def test_vector_must_have_three_components(self):
        with self.assertRaises(ValueError):
            polarization = normalize((3, 4, 5, 6))

        with self.assertRaises(ValueError):
            polarization = normalize((4, 0))

    def test_vector_must_be_finite(self):
        with self.assertRaises(ValueError):
            polarization = normalize((0, 0, 0))

    def test_vector_must_be_real(self):
        with self.assertRaises(ValueError):
            polarization = normalize((4 + 1j, 2, 0))


if __name__ == "__main__":
    unittest.main()
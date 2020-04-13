import unittest
import utility.data_utility as du


class TestDataUtility(unittest.TestCase):
    def test_load_data(self):
        result = du.load_data('test_dataset.csv')
        # ??


if __name__ == '__main__':
    unittest.main()

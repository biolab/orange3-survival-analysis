import unittest


class TestDeprecations(unittest.TestCase):
    def test_remove_get_column_workaround(self):
        """
        When this tests start to fail:
        1. remove this test
        2. set minimum version of Orange to 3.34 if not set yet
        3. remove workaround from __inint__.py
        """
        from datetime import datetime

        self.assertLess(datetime.today(), datetime(2024, 1, 1))


if __name__ == "__main__":
    unittest.main()

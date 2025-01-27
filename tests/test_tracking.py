import unittest
from process_video2unity_clean import process_single_video


class TestProcessing(unittest.TestCase):

    def test_process_single_video(self):
        process_single_video("results/grasp")


if __name__ == "__main__":
    unittest.main()

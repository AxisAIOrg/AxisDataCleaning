import unittest

import numpy as np

from smooth_resampled_traj import resample_trajectory


class ResampleTrajectoryEndpointTests(unittest.TestCase):
    def setUp(self):
        self.times = np.array([0.0, 0.5, 1.0])
        self.joint_names = ["joint1"]
        self.joint_data = {"joint1": np.array([0.0, 0.5, 1.0])}

    def test_keeps_final_timestep_when_dt_divides_span(self):
        new_times, resampled = resample_trajectory(
            self.times, self.joint_data, 0.1, self.joint_names
        )

        self.assertAlmostEqual(new_times[-1], 1.0)
        self.assertEqual(len(new_times), 11)
        self.assertAlmostEqual(resampled["joint1"][-1], 1.0)

    def test_keeps_exact_endpoint_with_shorter_final_interval(self):
        new_times, _ = resample_trajectory(
            self.times, self.joint_data, 0.3, self.joint_names
        )

        np.testing.assert_allclose(new_times, [0.0, 0.3, 0.6, 0.9, 1.0])
        self.assertAlmostEqual(new_times[-1], self.times[-1])


if __name__ == "__main__":
    unittest.main()

import unittest

from validate_offline_trajectory import run_checker_config


class CheckerConfigTests(unittest.TestCase):
    def test_disabled_legacy_checker_fails_closed(self):
        config = {"jointThresholdChecker": {"enabled": False, "threshold": 0}}
        self.assertFalse(run_checker_config({}, config))

    def test_malformed_legacy_entry_fails_closed(self):
        self.assertFalse(run_checker_config({}, {"jointThresholdChecker": "invalid"}))

    def test_executed_checker_can_pass(self):
        state = {"robot_joints": {"arm/joint": 1.0}}
        config = {
            "jointThresholdChecker": {
                "type": "JointThresholdChecker",
                "jointName": "arm/joint",
                "threshold": 0.0,
                "mode": "ge",
            }
        }
        self.assertTrue(run_checker_config(state, config))


if __name__ == "__main__":
    unittest.main()

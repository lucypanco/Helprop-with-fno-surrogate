import unittest

from helprop_surrogate.fno.convergence import ConvergenceMonitor


class ConvergenceMonitorTest(unittest.TestCase):
    def test_stops_on_validation_plateau_and_ignores_test_loss(self):
        monitor = ConvergenceMonitor(patience=2, min_delta=0.1, min_epochs=1)
        self.assertFalse(monitor.update(epoch=1, train_loss=3.0, val_loss=3.0, test_loss=99.0))
        self.assertFalse(monitor.update(epoch=2, train_loss=2.0, val_loss=2.0, test_loss=99.0))
        self.assertFalse(monitor.update(epoch=3, train_loss=1.0, val_loss=2.0, test_loss=0.0))
        self.assertTrue(monitor.update(epoch=4, train_loss=0.5, val_loss=2.0, test_loss=0.0))
        self.assertEqual(monitor.stopped_epoch, 4)
        self.assertEqual(monitor.best_value, 2.0)
        self.assertEqual(monitor.history[-1]["test_loss"], 0.0)


if __name__ == "__main__":
    unittest.main()

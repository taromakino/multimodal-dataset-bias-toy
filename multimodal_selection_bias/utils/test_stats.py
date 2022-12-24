import numpy as np
import torch
import unittest
from utils.stats import log_avg_prob

np.seterr(all="raise")

class MakeStatsTestCase(unittest.TestCase):
    def test_logp_y_x(self):
        p_y_xz = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
        conditional_truth = torch.log(p_y_xz.mean()).item()
        conditional_estimate = log_avg_prob(torch.log(p_y_xz)).item()
        self.assertTrue(np.allclose(conditional_truth, conditional_estimate))

        p_z = torch.tensor([0.5, 0.6, 0.7, 0.8, 0.9])
        p_z_x = torch.tensor([0.2, 0.4, 0.6, 0.8, 1.0])
        interventional_truth = torch.log(((p_z / p_z_x) * p_y_xz).mean()).item()
        interventional_estimate = interventional_logpy_x(torch.log(p_z), torch.log(p_z_x), torch.log(p_y_xz)).item()
        self.assertTrue(np.allclose(interventional_truth, interventional_estimate))
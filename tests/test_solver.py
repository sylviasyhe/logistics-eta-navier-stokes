"""
Unit Tests for Logistics N-S Solver
====================================
"""

import unittest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.logistics_ns_solver import (
    LogisticsNSEquation,
    LogisticsFlowSimulator,
    VaRCalculator
)


class TestLogisticsNSEquation(unittest.TestCase):
    """Test cases for LogisticsNSEquation class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.ns = LogisticsNSEquation(rho=1.0, mu_base=0.1)
    
    def test_carrier_viscosity(self):
        """Test carrier viscosity computation."""
        # High-quality carrier should have low viscosity
        mu_high = self.ns.carrier_viscosity(gamma=0.9, delta=0.95, R=0.98)
        mu_low = self.ns.carrier_viscosity(gamma=0.3, delta=0.4, R=0.6)
        
        self.assertLess(mu_high, mu_low)
        self.assertGreater(mu_high, 0)
        self.assertGreater(mu_low, 0)
    
    def test_commodity_viscosity(self):
        """Test commodity viscosity (power-law model)."""
        # Shear-thinning: viscosity decreases with shear rate
        mu_low_shear = self.ns.commodity_viscosity(K=2.0, n=0.7, shear_rate=0.5)
        mu_high_shear = self.ns.commodity_viscosity(K=2.0, n=0.7, shear_rate=2.0)
        
        self.assertLess(mu_high_shear, mu_low_shear)
    
    def test_combined_viscosity(self):
        """Test combined viscosity multiplication."""
        carrier_params = {'gamma': 0.8, 'delta': 0.9, 'R': 0.95}
        commodity_params = {'K': 2.0, 'n': 0.7}
        
        mu_combined = self.ns.combined_viscosity(carrier_params, commodity_params, 1.0)
        
        self.assertGreater(mu_combined, 0)
        self.assertIsInstance(mu_combined, float)
    
    def test_pressure_gradient(self):
        """Test pressure gradient computation."""
        # Higher demand should increase pressure
        pg_low = self.ns.pressure_gradient(demand=1.0, capacity=1.0)
        pg_high = self.ns.pressure_gradient(demand=2.0, capacity=1.0)
        
        self.assertGreater(pg_high, pg_low)
    
    def test_holiday_jump(self):
        """Test holiday jump term."""
        # Before holiday: no jump
        j_before = self.ns.holiday_jump(5.0, 8.0, 12.0, 18.0)
        self.assertEqual(j_before, 0.0)
        
        # During holiday: negative jump (capacity drop)
        j_during = self.ns.holiday_jump(15.0, 8.0, 12.0, 18.0)
        self.assertLess(j_during, 0)
        
        # After holiday: recovery (positive)
        j_after = self.ns.holiday_jump(25.0, 8.0, 12.0, 18.0)
        self.assertGreater(j_after, 0)


class TestLogisticsFlowSimulator(unittest.TestCase):
    """Test cases for LogisticsFlowSimulator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.ns = LogisticsNSEquation(rho=1.0, mu_base=0.1)
        self.sim = LogisticsFlowSimulator(self.ns, nx=20, nt=50)
        self.x, self.t = self.sim.initialize_grid(x_max=10.0, t_max=5.0)
    
    def test_grid_initialization(self):
        """Test grid initialization."""
        self.assertEqual(len(self.x), 20)
        self.assertEqual(len(self.t), 50)
        self.assertEqual(self.x[0], 0.0)
        self.assertEqual(self.x[-1], 10.0)
    
    def test_laplacian_1d(self):
        """Test 1D Laplacian computation."""
        v = np.sin(np.pi * self.x / 10.0)
        laplacian = self.sim.laplacian_1d(v)
        
        # Laplacian of sin(kx) should be -k²sin(kx)
        expected = -(np.pi / 10.0)**2 * v
        
        # Check approximate match (except boundaries)
        np.testing.assert_allclose(laplacian[1:-1], expected[1:-1], rtol=0.1)
    
    def test_simulation_runs(self):
        """Test that simulation runs without errors."""
        v0 = np.ones_like(self.x) * 1.0
        
        mu_func = lambda t: 0.1
        pg_func = lambda t: 0.0
        f_func = lambda t: 0.0
        s_func = lambda t: 0.0
        j_func = lambda t: 0.0
        
        v_history = self.sim.simulate(v0, mu_func, pg_func, f_func, s_func, j_func)
        
        self.assertEqual(v_history.shape, (50, 20))
        self.assertTrue(np.all(np.isfinite(v_history)))


class TestVaRCalculator(unittest.TestCase):
    """Test cases for VaRCalculator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.var_calc = VaRCalculator()
        np.random.seed(42)
        self.samples = np.random.normal(5.0, 1.0, 10000)
    
    def test_compute_var(self):
        """Test VaR computation."""
        var_95 = self.var_calc.compute_var(self.samples, 0.95)
        var_50 = self.var_calc.compute_var(self.samples, 0.50)
        
        # VaR should increase with confidence level
        self.assertGreater(var_95, var_50)
        
        # VaR_0.5 should be close to mean
        self.assertAlmostEqual(var_50, np.mean(self.samples), delta=0.1)
    
    def test_compute_expected_shortfall(self):
        """Test Expected Shortfall computation."""
        es_95 = self.var_calc.compute_expected_shortfall(self.samples, 0.95)
        var_95 = self.var_calc.compute_var(self.samples, 0.95)
        
        # ES should be greater than VaR
        self.assertGreater(es_95, var_95)
    
    def test_compute_entropy_gap(self):
        """Test entropy gap computation."""
        gap = self.var_calc.compute_entropy_gap(self.samples, 0.95)
        
        # Gap should be positive
        self.assertGreater(gap, 0)


class TestIntegration(unittest.TestCase):
    """Integration tests for complete workflow."""
    
    def test_end_to_end_simulation(self):
        """Test complete simulation workflow."""
        # Setup
        ns = LogisticsNSEquation(rho=1.0, mu_base=0.1)
        sim = LogisticsFlowSimulator(ns, nx=30, nt=100)
        x, t = sim.initialize_grid(x_max=10.0, t_max=10.0)
        
        # Initial condition
        v0 = np.ones_like(x) * 1.0
        
        # Simple scenario
        mu_func = lambda t: 0.1
        pg_func = lambda t: 0.5 if 3 <= t <= 7 else 0.0
        f_func = lambda t: 0.0
        s_func = lambda t: 1.0 if abs(t - 2) < 0.1 else 0.0
        j_func = lambda t: ns.holiday_jump(t, 3.0, 5.0, 7.0)
        
        # Run simulation
        v_history = sim.simulate(v0, mu_func, pg_func, f_func, s_func, j_func)
        
        # Analyze results
        var_calc = VaRCalculator()
        mid_idx = len(x) // 2
        v_mid = v_history[:, mid_idx]
        eta = 10.0 / (v_mid + 0.1)
        
        ldt_95 = var_calc.compute_var(eta, 0.95)
        gap_95 = var_calc.compute_entropy_gap(eta, 0.95)
        
        # Assertions
        self.assertTrue(np.all(np.isfinite(v_history)))
        self.assertGreater(ldt_95, 0)
        self.assertGreater(gap_95, 0)


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestLogisticsNSEquation))
    suite.addTests(loader.loadTestsFromTestCase(TestLogisticsFlowSimulator))
    suite.addTests(loader.loadTestsFromTestCase(TestVaRCalculator))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)

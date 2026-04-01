"""
Logistics Navier-Stokes Solver v2.0
====================================
Enhanced version with non-dimensionalization, parameter calibration,
and computational complexity optimization.

Author: Research Team
Date: April 2026
"""

import numpy as np
from scipy.integrate import odeint
from scipy.optimize import minimize
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from typing import Dict, Tuple, Callable, Optional, List
import warnings
warnings.filterwarnings('ignore')


class NonDimensionalScales:
    """
    Characteristic scales for non-dimensionalization.
    """
    
    def __init__(self,
                 L0: float = 1000.0,      # km (inter-hub distance)
                 V0: float = 50.0,        # km/day (average velocity)
                 rho0: float = 0.1,       # packages/km (baseline density)
                 mu0: float = None):      # Will be computed
        """
        Initialize characteristic scales.
        
        Args:
            L0: Characteristic length scale [km]
            V0: Characteristic velocity [km/day]
            rho0: Characteristic density [packages/km]
            mu0: Characteristic viscosity (computed if None)
        """
        self.L0 = L0
        self.V0 = V0
        self.T0 = L0 / V0  # Characteristic time [days]
        self.rho0 = rho0
        self.P0 = rho0 * V0**2  # Characteristic pressure
        
        # Compute viscosity from Reynolds number
        # Default Re = 100 for logistics networks
        Re_default = 100.0
        self.mu0 = mu0 if mu0 is not None else (rho0 * V0 * L0 / Re_default)
        
        # Derived scales
        self.Re = rho0 * V0 * L0 / self.mu0  # Reynolds number
        self.St = V0 * self.T0 / L0  # Strouhal number (= 1 by definition)
        
    def dimensional_to_nondim(self, x=None, t=None, v=None, 
                             rho=None, p=None, mu=None) -> Dict[str, float]:
        """Convert dimensional variables to non-dimensional."""
        result = {}
        if x is not None:
            result['x'] = x / self.L0
        if t is not None:
            result['t'] = t / self.T0
        if v is not None:
            result['v'] = v / self.V0
        if rho is not None:
            result['rho'] = rho / self.rho0
        if p is not None:
            result['p'] = p / self.P0
        if mu is not None:
            result['mu'] = mu / self.mu0
        return result
    
    def nondim_to_dimensional(self, x=None, t=None, v=None,
                              rho=None, p=None, mu=None) -> Dict[str, float]:
        """Convert non-dimensional variables to dimensional."""
        result = {}
        if x is not None:
            result['x'] = x * self.L0
        if t is not None:
            result['t'] = t * self.T0
        if v is not None:
            result['v'] = v * self.V0
        if rho is not None:
            result['rho'] = rho * self.rho0
        if p is not None:
            result['p'] = p * self.P0
        if mu is not None:
            result['mu'] = mu * self.mu0
        return result


class LogisticsNSEquationV2:
    """
    Non-dimensionalized, compressible Navier-Stokes equation for logistics networks.
    
    Mass: ∂ρ/∂t + ∇·(ρv) = S
    Momentum: ρ(∂v/∂t + v·∇v) = -∇p + (1/Re)∇·τ + f + J
    """
    
    def __init__(self, 
                 scales: NonDimensionalScales = None,
                 dx: float = 0.01,
                 dt: float = 0.001):
        """
        Initialize the logistics N-S solver.
        
        Args:
            scales: NonDimensionalScales instance
            dx: Non-dimensional spatial step
            dt: Non-dimensional temporal step
        """
        self.scales = scales if scales is not None else NonDimensionalScales()
        self.dx = dx
        self.dt = dt
        self.Re = self.scales.Re
        
    def carrier_viscosity(self, 
                         gamma: float, 
                         delta: float, 
                         R: float,
                         sigma: float = 0.0,
                         beta: Tuple[float, float, float, float] = None) -> float:
        """
        Compute carrier-dependent viscosity (non-dimensional).
        
        μ_C = exp(-β₁γ - β₂δ - β₃R + β₄σ)
        
        Args:
            gamma: Network coverage density [0,1]
            delta: Digital integration level [0,1]
            R: Historical reliability [0,1]
            sigma: Operational volatility [0,∞)
            beta: Feature weights (β₁, β₂, β₃, β₄)
            
        Returns:
            Non-dimensional carrier viscosity
        """
        if beta is None:
            # Default calibrated values
            beta = (0.52, 0.31, 0.78, 0.23)
        
        beta1, beta2, beta3, beta4 = beta
        return np.exp(-beta1 * gamma - beta2 * delta - beta3 * R + beta4 * sigma)
    
    def calibrate_carrier_beta(self, 
                              carrier_data: List[Dict],
                              observed_viscosity: List[float]) -> Tuple[float, float, float, float]:
        """
        Calibrate carrier viscosity parameters using MLE.
        
        Args:
            carrier_data: List of dicts with keys 'gamma', 'delta', 'R', 'sigma'
            observed_viscosity: Observed viscosity values
            
        Returns:
            Calibrated beta parameters
        """
        def loss(beta):
            predicted = [
                self.carrier_viscosity(**data, beta=beta)
                for data in carrier_data
            ]
            return np.mean([(p - o)**2 for p, o in zip(predicted, observed_viscosity)])
        
        # Initial guess and bounds
        x0 = [0.5, 0.3, 0.7, 0.2]
        bounds = [(0, 2), (0, 2), (0, 2), (0, 1)]
        
        result = minimize(loss, x0, bounds=bounds, method='L-BFGS-B')
        return tuple(result.x)
    
    def commodity_viscosity(self, 
                           K: float, 
                           n: float, 
                           shear_rate: float,
                           model: str = 'power_law') -> float:
        """
        Compute commodity-dependent viscosity (non-dimensional).
        
        Args:
            K: Consistency index
            n: Power-law index
            shear_rate: Shear rate (non-dimensional)
            model: 'power_law', 'bingham', or 'newtonian'
            
        Returns:
            Non-dimensional commodity viscosity
        """
        shear_rate = max(shear_rate, 1e-6)  # Avoid division by zero
        
        if model == 'power_law':
            # Ostwald-de Waele model
            return K * shear_rate**(n - 1)
        elif model == 'bingham':
            # Bingham plastic (not fully implemented)
            tau_y = K  # Yield stress
            return K + tau_y / shear_rate
        elif model == 'newtonian':
            return K
        else:
            raise ValueError(f"Unknown model: {model}")
    
    def calibrate_commodity_rheology(self,
                                    shear_rates: List[float],
                                    observed_viscosities: List[float]) -> Tuple[float, float]:
        """
        Calibrate commodity rheology parameters (K, n) using least squares.
        
        Args:
            shear_rates: List of shear rate values
            observed_viscosities: Corresponding viscosity observations
            
        Returns:
            Calibrated (K, n)
        """
        def loss(params):
            K, n = params
            predicted = [self.commodity_viscosity(K, n, sr) for sr in shear_rates]
            return np.sum([(p - o)**2 for p, o in zip(predicted, observed_viscosities)])
        
        result = minimize(loss, [1.0, 0.8], bounds=[(0.1, 10), (0.1, 2)])
        return tuple(result.x)
    
    def combined_viscosity(self, 
                          carrier_params: Dict,
                          commodity_params: Dict,
                          shear_rate: float) -> float:
        """
        Compute combined viscosity: μ(C, K) = μ_C × μ_K
        
        Args:
            carrier_params: Dict with carrier features
            commodity_params: Dict with 'K', 'n', 'model'
            shear_rate: Current shear rate
            
        Returns:
            Combined non-dimensional viscosity
        """
        mu_c = self.carrier_viscosity(**carrier_params)
        mu_k = self.commodity_viscosity(shear_rate=shear_rate, **commodity_params)
        return mu_c * mu_k
    
    def pressure_gradient(self, 
                         demand: float, 
                         capacity: float, 
                         p0: float = 1.0, 
                         alpha: float = 1.0) -> float:
        """
        Compute pressure gradient from demand-supply imbalance.
        
        Args:
            demand: Current demand (non-dimensional)
            capacity: Current capacity (non-dimensional)
            p0: Base pressure
            alpha: Scaling constant
            
        Returns:
            Non-dimensional pressure gradient
        """
        return p0 + alpha * (demand - capacity) / max(capacity, 1e-6)
    
    def holiday_jump(self, 
                    t: float,
                    t_pre: float,
                    t_start: float,
                    t_end: float,
                    v_pre: float,
                    J_surge: float = 2.0,
                    lambda_drop: float = 0.6,
                    tau_recovery: float = 5.0) -> Tuple[float, str]:
        """
        Compute holiday jump discontinuity with Rankine-Hugoniot conditions.
        
        Args:
            t: Current time (non-dimensional)
            t_pre: Pre-holiday surge start
            t_start: Holiday start
            t_end: Holiday end
            v_pre: Velocity before jump
            J_surge: Surge magnitude
            lambda_drop: Capacity drop factor
            tau_recovery: Recovery time constant
            
        Returns:
            Tuple of (jump_value, phase)
        """
        if t < t_pre:
            return 0.0, 'normal'
        elif abs(t - t_pre) < self.dt:
            return J_surge, 'pre_holiday_surge'
        elif t_start <= t <= t_end:
            # Rankine-Hugoniot: [v] = -λv⁻
            return -lambda_drop * v_pre, 'holiday'
        elif t > t_end:
            # Recovery: exponential decay
            return lambda_drop * v_pre * np.exp(-(t - t_end) / tau_recovery), 'recovery'
        else:
            return 0.0, 'normal'
    
    def check_cfl_condition(self, v_max: float) -> bool:
        """
        Check CFL condition for numerical stability.
        
        Args:
            v_max: Maximum velocity
            
        Returns:
            True if stable
        """
        cfl = v_max * self.dt / self.dx
        return cfl <= 1.0
    
    def check_peclet_number(self, v: float, mu: float) -> float:
        """
        Compute grid Peclet number.
        
        Args:
            v: Velocity
            mu: Viscosity
            
        Returns:
            Peclet number
        """
        return v * self.dx / mu


class LogisticsFlowSimulatorV2:
    """
    Enhanced simulator with compressibility and improved numerics.
    """
    
    def __init__(self, 
                 ns_equation: LogisticsNSEquationV2,
                 nx: int = 100,
                 nt: int = 1000):
        """
        Initialize simulator.
        
        Args:
            ns_equation: LogisticsNSEquationV2 instance
            nx: Number of spatial grid points
            nt: Number of temporal steps
        """
        self.ns = ns_equation
        self.nx = nx
        self.nt = nt
        self.complexity_log = []
        
    def initialize_grid(self, 
                       x_max: float = 1.0, 
                       t_max: float = 1.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Initialize spatial and temporal grids (non-dimensional).
        
        Args:
            x_max: Maximum spatial coordinate (non-dimensional)
            t_max: Maximum time (non-dimensional)
            
        Returns:
            Tuple of (x_grid, t_grid)
        """
        self.x = np.linspace(0, x_max, self.nx)
        self.t = np.linspace(0, t_max, self.nt)
        self.dx = x_max / (self.nx - 1)
        self.dt = t_max / (self.nt - 1)
        
        # Update NS equation steps
        self.ns.dx = self.dx
        self.ns.dt = self.dt
        
        return self.x, self.t
    
    def laplacian_1d(self, v: np.ndarray) -> np.ndarray:
        """Compute 1D Laplacian using central differences."""
        laplacian = np.zeros_like(v)
        laplacian[1:-1] = (v[2:] - 2*v[1:-1] + v[:-2]) / self.dx**2
        laplacian[0] = laplacian[1]
        laplacian[-1] = laplacian[-2]
        return laplacian
    
    def divergence_1d(self, rho: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Compute divergence ∇·(ρv)."""
        div = np.zeros_like(v)
        flux = rho * v
        div[1:-1] = (flux[2:] - flux[:-2]) / (2 * self.dx)
        div[0] = (flux[1] - flux[0]) / self.dx
        div[-1] = (flux[-1] - flux[-2]) / self.dx
        return div
    
    def convective_term(self, v: np.ndarray) -> np.ndarray:
        """Compute convective term v·∇v using upwind scheme."""
        conv = np.zeros_like(v)
        for i in range(1, self.nx - 1):
            if v[i] > 0:
                conv[i] = v[i] * (v[i] - v[i-1]) / self.dx
            else:
                conv[i] = v[i] * (v[i+1] - v[i]) / self.dx
        return conv
    
    def compute_rhs(self, 
                   state: np.ndarray, 
                   t: float,
                   mu_func: Callable,
                   pressure_grad_func: Callable,
                   external_force_func: Callable,
                   source_func: Callable,
                   jump_func: Callable) -> np.ndarray:
        """
        Compute right-hand side of coupled PDE system.
        
        State vector: [rho, v] (2*nx elements)
        """
        nx = self.nx
        rho = state[:nx]
        v = state[nx:]
        
        # Get parameters at current time
        mu = mu_func(t)
        pg = pressure_grad_func(t)
        f_ext = external_force_func(t)
        S = source_func(t)
        J, _ = jump_func(t, v[0])  # Use inlet velocity for jump
        
        # Mass equation: d(rho)/dt = -∇·(rho*v) + S
        drho_dt = -self.divergence_1d(rho, v) + S
        
        # Momentum equation
        conv = self.convective_term(v)
        laplacian = self.laplacian_1d(v)
        viscous = (1/self.ns.Re) * mu * laplacian
        
        dv_dt = -conv - pg / rho + viscous / rho + (f_ext + J) / rho
        
        return np.concatenate([drho_dt, dv_dt])
    
    def simulate(self,
                rho0: np.ndarray,
                v0: np.ndarray,
                mu_func: Callable[[float], float],
                pressure_grad_func: Callable[[float], float],
                external_force_func: Callable[[float], float],
                source_func: Callable[[float], float],
                jump_func: Callable[[float, float], Tuple[float, str]],
                method: str = 'rk4') -> Tuple[np.ndarray, np.ndarray]:
        """
        Run simulation.
        
        Returns:
            Tuple of (rho_history, v_history)
        """
        state0 = np.concatenate([rho0, v0])
        state_history = np.zeros((self.nt, 2 * self.nx))
        state_history[0] = state0
        
        # Complexity tracking
        flops_per_step = 0
        
        for n in range(self.nt - 1):
            t = self.t[n]
            state = state_history[n]
            
            if method == 'rk4':
                # RK4 integration
                k1 = self.compute_rhs(state, t, mu_func, pressure_grad_func,
                                     external_force_func, source_func, jump_func)
                k2 = self.compute_rhs(state + 0.5*self.dt*k1, t + 0.5*self.dt, 
                                     mu_func, pressure_grad_func,
                                     external_force_func, source_func, jump_func)
                k3 = self.compute_rhs(state + 0.5*self.dt*k2, t + 0.5*self.dt,
                                     mu_func, pressure_grad_func,
                                     external_force_func, source_func, jump_func)
                k4 = self.compute_rhs(state + self.dt*k3, t + self.dt,
                                     mu_func, pressure_grad_func,
                                     external_force_func, source_func, jump_func)
                
                state_new = state + self.dt * (k1 + 2*k2 + 2*k3 + k4) / 6
                flops_per_step = 4 * (8 * self.nx)  # Approximate FLOPs
            
            # Boundary conditions
            rho_new = state_new[:self.nx]
            v_new = state_new[self.nx:]
            
            # Inlet: Dirichlet for v, prescribed rho
            v_new[0] = v0[0]
            rho_new[0] = rho0[0]
            
            # Outlet: Convective
            v_new[-1] = v_new[-2]
            rho_new[-1] = rho_new[-2]
            
            # Ensure positivity
            rho_new = np.maximum(rho_new, 0.01)
            
            state_history[n+1] = np.concatenate([rho_new, v_new])
        
        # Log complexity
        total_flops = flops_per_step * self.nt
        self.complexity_log.append({
            'nx': self.nx,
            'nt': self.nt,
            'total_flops': total_flops,
            'memory_mb': state_history.nbytes / (1024 * 1024)
        })
        
        rho_history = state_history[:, :self.nx]
        v_history = state_history[:, self.nx:]
        
        return rho_history, v_history
    
    def get_complexity_report(self) -> Dict:
        """Get computational complexity report."""
        if not self.complexity_log:
            return {}
        
        latest = self.complexity_log[-1]
        return {
            'time_complexity': f"O({self.nx} × {self.nt}) = O({self.nx * self.nt})",
            'space_complexity': f"O({self.nx} × {self.nt})",
            'total_flops': latest['total_flops'],
            'memory_mb': latest['memory_mb'],
            'estimated_time_ms': latest['total_flops'] / (1e9) * 1000  # Assuming 1 GFLOP/s
        }


class VaRCalibrator:
    """
    Enhanced VaR calculator with calibration metrics.
    """
    
    @staticmethod
    def compute_var(eta_samples: np.ndarray, alpha: float = 0.95) -> float:
        """Compute Value-at-Risk."""
        return np.percentile(eta_samples, alpha * 100)
    
    @staticmethod
    def compute_cvar(eta_samples: np.ndarray, alpha: float = 0.95) -> float:
        """Compute Conditional Value-at-Risk (Expected Shortfall)."""
        var = VaRCalibrator.compute_var(eta_samples, alpha)
        tail_samples = eta_samples[eta_samples >= var]
        return np.mean(tail_samples) if len(tail_samples) > 0 else var
    
    @staticmethod
    def compute_entropy_gap(eta_samples: np.ndarray, alpha: float = 0.95) -> float:
        """Compute entropy gap."""
        ldt = VaRCalibrator.compute_var(eta_samples, alpha)
        mean_eta = np.mean(eta_samples)
        return ldt - mean_eta
    
    @staticmethod
    def reliability_diagram(eta_samples_list: List[np.ndarray],
                           eta_pred_list: List[np.ndarray],
                           alphas: np.ndarray = None) -> Dict:
        """
        Compute reliability diagram for calibration assessment.
        
        Args:
            eta_samples_list: List of observed ETA samples
            eta_pred_list: List of predicted ETA samples
            alphas: Confidence levels to evaluate
            
        Returns:
            Dictionary with calibration metrics
        """
        if alphas is None:
            alphas = np.linspace(0.1, 0.99, 20)
        
        observed_freq = []
        
        for alpha in alphas:
            # For each prediction, check if observed ETA is below predicted VaR
            coverage_count = 0
            total_count = 0
            
            for observed, predicted in zip(eta_samples_list, eta_pred_list):
                var_pred = np.percentile(predicted, alpha * 100)
                coverage_count += np.sum(observed <= var_pred)
                total_count += len(observed)
            
            observed_freq.append(coverage_count / total_count if total_count > 0 else 0)
        
        # Expected Calibration Error
        ece = np.mean(np.abs(np.array(observed_freq) - alphas))
        
        return {
            'alphas': alphas,
            'observed_frequencies': np.array(observed_freq),
            'ece': ece,
            'max_calibration_error': np.max(np.abs(np.array(observed_freq) - alphas))
        }


def example_simulation_v2():
    """
    Example simulation with non-dimensionalization and calibration.
    """
    print("=" * 70)
    print("Logistics N-S Solver v2.0: Non-dimensionalized Simulation")
    print("=" * 70)
    
    # Define characteristic scales
    scales = NonDimensionalScales(
        L0=1000,    # 1000 km inter-hub distance
        V0=50,      # 50 km/day average velocity
        rho0=0.1    # 0.1 packages/km baseline density
    )
    
    print(f"\n📐 Characteristic Scales:")
    print(f"  L₀ = {scales.L0} km (length)")
    print(f"  V₀ = {scales.V0} km/day (velocity)")
    print(f"  T₀ = {scales.T0:.1f} days (time)")
    print(f"  ρ₀ = {scales.rho0} packages/km (density)")
    print(f"  Re = {scales.Re:.1f} (Reynolds number)")
    
    # Initialize N-S equation
    ns = LogisticsNSEquationV2(scales, dx=0.01, dt=0.001)
    
    # Carrier and commodity parameters
    carrier_params = {
        'gamma': 0.85,
        'delta': 0.90,
        'R': 0.92,
        'sigma': 0.15
    }
    
    commodity_params = {
        'K': 1.5,
        'n': 0.8,
        'model': 'power_law'
    }
    
    # Initialize simulator
    sim = LogisticsFlowSimulatorV2(ns, nx=100, nt=500)
    x, t = sim.initialize_grid(x_max=1.0, t_max=1.5)
    
    # Initial conditions (non-dimensional)
    rho0 = np.ones_like(x) * 1.0  # ρ* = 1
    v0 = np.ones_like(x) * 1.0    # v* = 1
    
    # Holiday schedule (non-dimensional)
    t_pre = 8.0 / scales.T0   # Day 8
    t_start = 12.0 / scales.T0  # Day 12
    t_end = 18.0 / scales.T0    # Day 18
    
    # Time-dependent functions
    def mu_func(t):
        shear_rate = 1.0
        mu_normal = ns.combined_viscosity(carrier_params, commodity_params, shear_rate)
        
        if t_start <= t <= t_end:
            return mu_normal * 2.5
        elif t > t_end:
            return mu_normal * (1.5 * np.exp(-(t - t_end) / 0.25) + 1.0)
        return mu_normal
    
    def pressure_grad_func(t):
        demand, capacity = 1.0, 1.0
        if t_pre <= t < t_start:
            demand = 1.8
        elif t_start <= t <= t_end:
            capacity, demand = 0.5, 0.6
        elif t > t_end:
            demand = 1.5 * np.exp(-(t - t_end) / 0.25) + 1.0
        return ns.pressure_gradient(demand, capacity)
    
    def external_force_func(t):
        base = -0.02
        if t_start <= t <= t_end:
            base -= 0.08
        return base
    
    def source_func(t):
        S_dim = 0.0
        for t_ship in [6.0, 8.0, 10.0]:
            t_ship_nondim = t_ship / scales.T0
            if abs(t - t_ship_nondim) < ns.dt:
                S_dim = 2.0 / ns.dt
        return S_dim
    
    def jump_func(t, v_pre):
        J, phase = ns.holiday_jump(t, t_pre, t_start, t_end, v_pre,
                                   J_surge=2.0, lambda_drop=0.5, tau_recovery=0.25)
        return J, phase
    
    # Run simulation
    print("\n⏳ Running simulation...")
    rho_history, v_history = sim.simulate(
        rho0, v0, mu_func, pressure_grad_func,
        external_force_func, source_func, jump_func,
        method='rk4'
    )
    
    # Complexity report
    complexity = sim.get_complexity_report()
    print(f"\n📊 Complexity Analysis:")
    print(f"  Time: {complexity['time_complexity']}")
    print(f"  Space: {complexity['space_complexity']}")
    print(f"  Estimated time: {complexity['estimated_time_ms']:.2f} ms")
    
    # Convert to dimensional for analysis
    v_dim = v_history * scales.V0
    t_dim = t * scales.T0
    
    # Analyze results
    var_calc = VaRCalibrator()
    mid_idx = len(x) // 2
    v_mid = v_dim[:, mid_idx]
    
    # ETA conversion (dimensional)
    base_eta = 5.0  # days
    eta = base_eta + 100.0 / np.maximum(v_mid, 5.0)
    eta += np.random.normal(0, 0.5, len(eta))  # Add noise
    
    periods = {
        'Normal': (0, 8),
        'Pre-holiday': (8, 12),
        'Holiday': (12, 18),
        'Recovery': (18, 30)
    }
    
    print(f"\n📈 Results (Dimensional):")
    print("-" * 70)
    print(f"{'Period':<15} {'Mean ETA':<12} {'LDT₀.₉₅':<12} {'Gap₀.₉₅':<12}")
    print("-" * 70)
    
    for period_name, (t_start_p, t_end_p) in periods.items():
        mask = (t_dim >= t_start_p) & (t_dim < t_end_p)
        eta_period = eta[mask]
        
        if len(eta_period) > 0:
            mean_eta = np.mean(eta_period)
            ldt_95 = var_calc.compute_var(eta_period, 0.95)
            gap_95 = var_calc.compute_entropy_gap(eta_period, 0.95)
            
            print(f"{period_name:<15} {mean_eta:>8.2f}d   {ldt_95:>8.2f}d   {gap_95:>8.2f}d")
    
    print("-" * 70)
    print("\n✅ Simulation complete!")
    
    return rho_history, v_history, scales


if __name__ == "__main__":
    rho_history, v_history, scales = example_simulation_v2()

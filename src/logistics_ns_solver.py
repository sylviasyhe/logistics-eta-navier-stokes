"""
Logistics Navier-Stokes Solver
==============================
A physics-informed framework for global logistics ETA prediction using
non-homogeneous Navier-Stokes equations with jump discontinuities.

Author: Research Team
Date: April 2026
"""

import numpy as np
from scipy.integrate import odeint
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from typing import Dict, Tuple, Callable, Optional
import warnings
warnings.filterwarnings('ignore')


class LogisticsNSEquation:
    """
    Non-homogeneous Navier-Stokes equation for logistics networks.
    
    Equation:
        ρ(∂v/∂t + v·∇v) = -∇p + μ(C,K)∇²v + f_ext + S_merchant + J_holiday
    """
    
    def __init__(self, 
                 rho: float = 1.0,
                 mu_base: float = 0.1,
                 dx: float = 0.1,
                 dt: float = 0.01):
        """
        Initialize the logistics N-S solver.
        
        Args:
            rho: Effective package density
            mu_base: Base viscosity coefficient
            dx: Spatial discretization step
            dt: Temporal discretization step
        """
        self.rho = rho
        self.mu_base = mu_base
        self.dx = dx
        self.dt = dt
        
    def carrier_viscosity(self, 
                         gamma: float, 
                         delta: float, 
                         R: float,
                         beta: Tuple[float, float, float] = (0.5, 0.3, 0.8)) -> float:
        """
        Compute carrier-dependent viscosity.
        
        μ_C = μ_0 * exp(-β₁γ - β₂δ - β₃R)
        
        Args:
            gamma: Network coverage density
            delta: Digital integration level
            R: Historical reliability
            beta: Feature weights (β₁, β₂, β₃)
            
        Returns:
            Carrier viscosity coefficient
        """
        beta1, beta2, beta3 = beta
        return self.mu_base * np.exp(-beta1 * gamma - beta2 * delta - beta3 * R)
    
    def commodity_viscosity(self, 
                           K: float, 
                           n: float, 
                           shear_rate: float) -> float:
        """
        Compute commodity-dependent viscosity (power-law model).
        
        μ_K = K * |γ̇|^(n-1)
        
        Args:
            K: Consistency index
            n: Power-law index (n < 1 for shear-thinning)
            shear_rate: Shear rate (package processing speed)
            
        Returns:
            Commodity viscosity coefficient
        """
        return K * np.abs(shear_rate) ** (n - 1)
    
    def combined_viscosity(self, 
                          carrier_params: Dict,
                          commodity_params: Dict,
                          shear_rate: float) -> float:
        """
        Compute combined viscosity: μ(C, K) = μ_C * μ_K
        
        Args:
            carrier_params: Dict with keys 'gamma', 'delta', 'R'
            commodity_params: Dict with keys 'K', 'n'
            shear_rate: Current shear rate
            
        Returns:
            Combined viscosity coefficient
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
        
        ∇p = α * (D - C) / C_max
        
        Args:
            demand: Current demand (shipment volume)
            capacity: Current capacity
            p0: Base pressure
            alpha: Scaling constant
            
        Returns:
            Pressure gradient
        """
        return p0 + alpha * (demand - capacity) / max(capacity, 1e-6)
    
    def holiday_jump(self, 
                    t: float,
                    t_pre: float,
                    t_start: float,
                    t_end: float,
                    J_surge: float = 3.5,
                    J_drop: float = 0.6,
                    J_recovery: float = 0.2) -> float:
        """
        Compute holiday jump discontinuity term.
        
        Args:
            t: Current time
            t_pre: Pre-holiday surge start
            t_start: Holiday start
            t_end: Holiday end
            J_surge: Surge magnitude
            J_drop: Capacity drop factor
            J_recovery: Recovery rate
            
        Returns:
            Jump term value
        """
        if t < t_pre:
            return 0.0
        elif abs(t - t_pre) < self.dt:
            return J_surge  # Pre-holiday surge pulse
        elif t_start <= t <= t_end:
            return -J_drop  # Holiday capacity reduction
        elif t > t_end:
            return J_recovery * np.exp(-(t - t_end) / 5)  # Recovery
        else:
            return 0.0
    
    def merchant_source(self, 
                       t: float,
                       shipments: list) -> float:
        """
        Compute merchant source term (shipment injection).
        
        S_merchant = Σ Q_i * δ(t - t_i)
        
        Args:
            t: Current time
            shipments: List of (time, volume) tuples
            
        Returns:
            Source term value
        """
        S = 0.0
        for t_ship, Q in shipments:
            if abs(t - t_ship) < self.dt:
                S += Q / self.dt  # Approximate delta function
        return S


class LogisticsFlowSimulator:
    """
    Simulator for logistics flow dynamics using finite difference method.
    """
    
    def __init__(self, 
                 ns_equation: LogisticsNSEquation,
                 nx: int = 100,
                 nt: int = 1000):
        """
        Initialize simulator.
        
        Args:
            ns_equation: LogisticsNSEquation instance
            nx: Number of spatial grid points
            nt: Number of temporal steps
        """
        self.ns = ns_equation
        self.nx = nx
        self.nt = nt
        
    def initialize_grid(self, 
                       x_max: float = 10.0, 
                       t_max: float = 30.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Initialize spatial and temporal grids.
        
        Args:
            x_max: Maximum spatial coordinate
            t_max: Maximum time
            
        Returns:
            Tuple of (x_grid, t_grid)
        """
        self.x = np.linspace(0, x_max, self.nx)
        self.t = np.linspace(0, t_max, self.nt)
        self.dx = x_max / (self.nx - 1)
        self.dt = t_max / (self.nt - 1)
        return self.x, self.t
    
    def laplacian_1d(self, v: np.ndarray) -> np.ndarray:
        """
        Compute 1D Laplacian using central differences.
        
        ∇²v ≈ (v[i+1] - 2v[i] + v[i-1]) / dx²
        
        Args:
            v: Velocity field
            
        Returns:
            Laplacian of v
        """
        laplacian = np.zeros_like(v)
        laplacian[1:-1] = (v[2:] - 2*v[1:-1] + v[:-2]) / self.dx**2
        # Neumann boundary conditions
        laplacian[0] = laplacian[1]
        laplacian[-1] = laplacian[-2]
        return laplacian
    
    def convective_term(self, v: np.ndarray) -> np.ndarray:
        """
        Compute convective term v·∇v using upwind scheme.
        
        Args:
            v: Velocity field
            
        Returns:
            Convective term
        """
        conv = np.zeros_like(v)
        # Upwind scheme
        for i in range(1, self.nx - 1):
            if v[i] > 0:
                conv[i] = v[i] * (v[i] - v[i-1]) / self.dx
            else:
                conv[i] = v[i] * (v[i+1] - v[i]) / self.dx
        return conv
    
    def rhs(self, 
           v: np.ndarray, 
           t: float,
           mu: float,
           pressure_grad: Callable,
           external_force: Callable,
           source: Callable,
           jump: Callable) -> np.ndarray:
        """
        Compute right-hand side of N-S equation.
        
        Args:
            v: Current velocity field
            t: Current time
            mu: Viscosity coefficient
            pressure_grad: Pressure gradient function
            external_force: External force function
            source: Source term function
            jump: Jump term function
            
        Returns:
            Time derivative of velocity
        """
        # Convective term
        conv = self.convective_term(v)
        
        # Viscous term
        laplacian = self.laplacian_1d(v)
        viscous = mu * laplacian
        
        # Other terms (spatially averaged for 1D)
        p_grad = pressure_grad(t)
        f_ext = external_force(t)
        S = source(t)
        J = jump(t)
        
        # Full equation: dv/dt = -(v·∇v) - ∇p/ρ + μ∇²v/ρ + (f + S + J)/ρ
        dvdt = -conv - p_grad / self.ns.rho + viscous / self.ns.rho + (f_ext + S + J) / self.ns.rho
        
        return dvdt
    
    def simulate(self,
                v0: np.ndarray,
                mu_func: Callable[[float], float],
                pressure_grad_func: Callable[[float], float],
                external_force_func: Callable[[float], float],
                source_func: Callable[[float], float],
                jump_func: Callable[[float], float],
                method: str = 'rk4') -> np.ndarray:
        """
        Run simulation using specified ODE solver.
        
        Args:
            v0: Initial velocity field
            mu_func: Viscosity as function of time
            pressure_grad_func: Pressure gradient as function of time
            external_force_func: External force as function of time
            source_func: Source term as function of time
            jump_func: Jump term as function of time
            method: Integration method ('euler', 'rk4')
            
        Returns:
            Velocity field history [nt, nx]
        """
        v_history = np.zeros((self.nt, self.nx))
        v_history[0] = v0
        
        v = v0.copy()
        
        for n in range(self.nt - 1):
            t = self.t[n]
            mu = mu_func(t)
            
            if method == 'euler':
                # Explicit Euler
                dvdt = self.rhs(v, t, mu, pressure_grad_func, 
                               external_force_func, source_func, jump_func)
                v = v + self.dt * dvdt
                
            elif method == 'rk4':
                # Runge-Kutta 4
                k1 = self.rhs(v, t, mu, pressure_grad_func,
                             external_force_func, source_func, jump_func)
                k2 = self.rhs(v + 0.5*self.dt*k1, t + 0.5*self.dt, mu, pressure_grad_func,
                             external_force_func, source_func, jump_func)
                k3 = self.rhs(v + 0.5*self.dt*k2, t + 0.5*self.dt, mu, pressure_grad_func,
                             external_force_func, source_func, jump_func)
                k4 = self.rhs(v + self.dt*k3, t + self.dt, mu, pressure_grad_func,
                             external_force_func, source_func, jump_func)
                v = v + self.dt * (k1 + 2*k2 + 2*k3 + k4) / 6
            
            # Apply boundary conditions
            v[0] = v[1]  # Neumann at left
            v[-1] = v[-2]  # Neumann at right
            
            v_history[n+1] = v
        
        return v_history


class VaRCalculator:
    """
    Value-at-Risk calculator for logistics ETA distributions.
    """
    
    @staticmethod
    def compute_var(eta_samples: np.ndarray, 
                   alpha: float = 0.95) -> float:
        """
        Compute Value-at-Risk (VaR) for ETA distribution.
        
        LDT_α = VaR_α(T) = inf{t: P(T ≤ t) ≥ α}
        
        Args:
            eta_samples: Array of ETA samples
            alpha: Confidence level (default 0.95)
            
        Returns:
            VaR value (Latest Delivery Time)
        """
        return np.percentile(eta_samples, alpha * 100)
    
    @staticmethod
    def compute_expected_shortfall(eta_samples: np.ndarray,
                                   alpha: float = 0.95) -> float:
        """
        Compute Expected Shortfall (CVaR) for ETA distribution.
        
        ES_α = E[T | T ≥ VaR_α]
        
        Args:
            eta_samples: Array of ETA samples
            alpha: Confidence level
            
        Returns:
            Expected shortfall
        """
        var = VaRCalculator.compute_var(eta_samples, alpha)
        return np.mean(eta_samples[eta_samples >= var])
    
    @staticmethod
    def compute_entropy_gap(eta_samples: np.ndarray,
                           alpha: float = 0.95) -> float:
        """
        Compute entropy gap: Gap_α = LDT_α - E[T]
        
        Args:
            eta_samples: Array of ETA samples
            alpha: Confidence level
            
        Returns:
            Entropy gap in days
        """
        ldt = VaRCalculator.compute_var(eta_samples, alpha)
        mean_eta = np.mean(eta_samples)
        return ldt - mean_eta


def example_simulation():
    """
    Example simulation: Spring Festival cross-border fresh produce.
    """
    print("=" * 60)
    print("Logistics N-S Solver: Example Simulation")
    print("Scenario: Spring Festival Cross-border Fresh Produce")
    print("=" * 60)
    
    # Initialize N-S equation
    ns = LogisticsNSEquation(rho=1.0, mu_base=0.1)
    
    # Carrier parameters (premium service)
    carrier_params = {
        'gamma': 0.8,  # High network coverage
        'delta': 0.9,  # High digital integration
        'R': 0.95      # High reliability
    }
    
    # Commodity parameters (fresh produce, shear-thinning)
    commodity_params = {
        'K': 2.0,   # Consistency index
        'n': 0.7    # Power-law index (< 1 for shear-thinning)
    }
    
    # Holiday schedule
    t_pre = 8.0
    t_start = 12.0
    t_end = 18.0
    
    # Initialize simulator
    sim = LogisticsFlowSimulator(ns, nx=50, nt=500)
    x, t = sim.initialize_grid(x_max=10.0, t_max=30.0)
    
    # Initial velocity field
    v0 = np.ones_like(x) * 1.0
    
    # Define time-dependent functions
    def mu_func(t):
        shear_rate = 1.0  # Assume constant processing speed
        mu_normal = ns.combined_viscosity(carrier_params, commodity_params, shear_rate)
        
        # Viscosity increases during holiday due to backlog
        if t_start <= t <= t_end:
            return mu_normal * 3.5
        elif t > t_end:
            return mu_normal * (2.0 * np.exp(-(t - t_end) / 4) + 1.0)
        return mu_normal
    
    def pressure_grad_func(t):
        demand = 1.0
        capacity = 1.0
        
        if t_pre <= t < t_start:
            demand = 2.5  # Pre-holiday surge
        elif t_start <= t <= t_end:
            capacity = 0.4  # Holiday capacity reduction
            demand = 0.5
        elif t > t_end:
            demand = 1.8 * np.exp(-(t - t_end) / 5) + 1.0
        
        return ns.pressure_gradient(demand, capacity)
    
    def external_force_func(t):
        # Weather/regulatory friction
        return -0.1 if t_start <= t <= t_end else 0.0
    
    def source_func(t):
        # Merchant shipments
        shipments = [(5.0, 2.0), (7.0, 3.0), (9.0, 4.0)]  # Pre-holiday pulses
        return ns.merchant_source(t, shipments)
    
    def jump_func(t):
        return ns.holiday_jump(t, t_pre, t_start, t_end)
    
    # Run simulation
    print("\nRunning simulation...")
    v_history = sim.simulate(v0, mu_func, pressure_grad_func, 
                            external_force_func, source_func, jump_func,
                            method='rk4')
    
    # Analyze results
    print("\n" + "=" * 60)
    print("SIMULATION RESULTS")
    print("=" * 60)
    
    # Extract velocity at a specific location over time
    mid_idx = len(x) // 2
    v_mid = v_history[:, mid_idx]
    
    # Compute statistics for different periods
    periods = {
        'Normal': (0, 8),
        'Pre-holiday': (8, 12),
        'Holiday': (12, 18),
        'Recovery': (18, 30)
    }
    
    var_calc = VaRCalculator()
    
    for period_name, (t_start_p, t_end_p) in periods.items():
        mask = (t >= t_start_p) & (t < t_end_p)
        v_period = v_mid[mask]
        
        # Convert velocity to ETA (inverse relationship)
        eta_period = 10.0 / (v_period + 0.1)  # Simplified conversion
        
        mean_eta = np.mean(eta_period)
        ldt_95 = var_calc.compute_var(eta_period, 0.95)
        ldt_99 = var_calc.compute_var(eta_period, 0.99)
        gap_95 = var_calc.compute_entropy_gap(eta_period, 0.95)
        
        print(f"\n{period_name} Period:")
        print(f"  Mean ETA: {mean_eta:.2f} days")
        print(f"  LDT₀.₉₅: {ldt_95:.2f} days")
        print(f"  LDT₀.₉₉: {ldt_99:.2f} days")
        print(f"  Gap₀.₉₅: {gap_95:.2f} days")
    
    print("\n" + "=" * 60)
    print("Simulation complete!")
    print("=" * 60)
    
    return v_history, x, t


if __name__ == "__main__":
    # Run example simulation
    v_history, x, t = example_simulation()

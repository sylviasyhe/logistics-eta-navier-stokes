"""
Spring Festival Cross-border Fresh Produce Simulation
======================================================
Scenario: Guangzhou (CN) → Los Angeles (US) via Hong Kong hub
Commodity: Fresh durian (high perishability)
Period: January 15 - February 15

This script demonstrates the complete simulation workflow for a realistic
logistics scenario during the Spring Festival holiday period.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.logistics_ns_solver import (
    LogisticsNSEquation, 
    LogisticsFlowSimulator, 
    VaRCalculator
)
from src.visualization import LogisticsVisualizer


def configure_spring_festival_scenario():
    """
    Configure simulation parameters for Spring Festival scenario.
    """
    print("=" * 70)
    print("SPRING FESTIVAL CROSS-BORDER FRESH PRODUCE SIMULATION")
    print("Route: Guangzhou → Hong Kong → Los Angeles")
    print("Commodity: Fresh Durian (High Perishability)")
    print("=" * 70)
    
    # Initialize N-S equation
    ns = LogisticsNSEquation(rho=1.0, mu_base=0.1)
    
    # Carrier configuration (Mixed: Premium air + Standard last-mile)
    carrier_params = {
        'gamma': 0.75,  # Good network coverage
        'delta': 0.85,  # High digital integration
        'R': 0.90       # Good reliability
    }
    
    # Commodity configuration (Fresh produce: shear-thinning)
    commodity_params = {
        'K': 2.0,   # High consistency index (sensitive)
        'n': 0.7    # Shear-thinning (n < 1)
    }
    
    # Holiday schedule (Spring Festival 2026)
    t_pre = 8.0      # Pre-holiday surge starts (Jan 23)
    t_start = 12.0   # Holiday starts (Jan 27)
    t_end = 18.0     # Holiday ends (Feb 2)
    
    # Jump parameters
    jump_params = {
        'J_surge': 3.5,    # 3.5x volume surge
        'J_drop': 0.6,     # 60% capacity reduction
        'J_recovery': 0.2  # Recovery rate
    }
    
    print("\n📋 Configuration:")
    print(f"  Carrier Quality Score: {np.mean(list(carrier_params.values())):.2f}")
    print(f"  Commodity Sensitivity: High (K={commodity_params['K']}, n={commodity_params['n']})")
    print(f"  Holiday Period: Day {t_start} - {t_end}")
    print(f"  Expected Capacity Drop: {jump_params['J_drop']*100:.0f}%")
    
    return ns, carrier_params, commodity_params, t_pre, t_start, t_end, jump_params


def run_simulation(ns, carrier_params, commodity_params, 
                   t_pre, t_start, t_end, jump_params):
    """
    Run the logistics flow simulation.
    """
    print("\n" + "=" * 70)
    print("RUNNING SIMULATION")
    print("=" * 70)
    
    # Initialize simulator
    sim = LogisticsFlowSimulator(ns, nx=100, nt=1000)
    x, t = sim.initialize_grid(x_max=10.0, t_max=30.0)
    
    # Initial velocity field (normalized)
    v0 = np.ones_like(x) * 1.0
    
    # Time-dependent viscosity function
    def mu_func(t):
        shear_rate = 1.0
        mu_normal = ns.combined_viscosity(carrier_params, commodity_params, shear_rate)
        
        # Holiday period: viscosity increases due to backlog and delays
        if t_start <= t <= t_end:
            return mu_normal * 3.5  # 3.5x viscosity during holiday
        elif t > t_end:
            # Recovery: gradual return to normal
            return mu_normal * (2.0 * np.exp(-(t - t_end) / 4) + 1.0)
        return mu_normal
    
    # Time-dependent pressure gradient
    def pressure_grad_func(t):
        demand = 1.0
        capacity = 1.0
        
        if t_pre <= t < t_start:
            # Pre-holiday: demand surge
            demand = 2.5
        elif t_start <= t <= t_end:
            # Holiday: capacity reduced, low demand
            capacity = 0.4
            demand = 0.5
        elif t > t_end:
            # Recovery: backlog clearance
            demand = 1.8 * np.exp(-(t - t_end) / 5) + 1.0
        
        return ns.pressure_gradient(demand, capacity)
    
    # External forces (customs, weather)
    def external_force_func(t):
        base_friction = -0.05
        if t_start <= t <= t_end:
            # Additional customs delay during holiday
            base_friction -= 0.15
        return base_friction
    
    # Merchant shipments (pre-holiday pulse)
    shipments = [
        (5.0, 2.0),   # Day 5: moderate volume
        (7.0, 3.5),   # Day 7: high volume
        (9.0, 4.0),   # Day 9: peak volume
        (10.5, 2.5),  # Day 10.5: last-minute rush
    ]
    
    def source_func(t):
        return ns.merchant_source(t, shipments)
    
    # Holiday jump
    def jump_func(t):
        return ns.holiday_jump(t, t_pre, t_start, t_end, 
                              jump_params['J_surge'],
                              jump_params['J_drop'],
                              jump_params['J_recovery'])
    
    # Run simulation
    print("\n⏳ Simulating 30 days with RK4 integration...")
    v_history = sim.simulate(v0, mu_func, pressure_grad_func,
                            external_force_func, source_func, jump_func,
                            method='rk4')
    print("✅ Simulation complete!")
    
    return v_history, x, t, mu_func


def analyze_results(v_history, x, t, mu_func, t_start, t_end):
    """
    Analyze simulation results and compute risk metrics.
    """
    print("\n" + "=" * 70)
    print("RESULTS ANALYSIS")
    print("=" * 70)
    
    # Extract velocity at midpoint (representative location)
    mid_idx = len(x) // 2
    v_mid = v_history[:, mid_idx]
    
    # Convert velocity to ETA (simplified model: ETA = distance / velocity)
    distance = 10.0  # Normalized distance units
    eta = distance / (v_mid + 0.1)  # Add small epsilon to avoid division by zero
    
    # Define periods
    periods = {
        'Normal': (0, 8),
        'Pre-holiday': (8, 12),
        'Holiday': (12, 18),
        'Recovery': (18, 30)
    }
    
    # Compute metrics for each period
    var_calc = VaRCalculator()
    metrics = {}
    
    print("\n📊 Period Analysis:")
    print("-" * 70)
    print(f"{'Period':<15} {'Mean ETA':<12} {'LDT₀.₉₅':<12} {'Gap₀.₉₅':<12} {'CVaR₀.₉₅':<12}")
    print("-" * 70)
    
    for period_name, (t_start_p, t_end_p) in periods.items():
        mask = (t >= t_start_p) & (t < t_end_p)
        eta_period = eta[mask]
        
        if len(eta_period) > 0:
            mean_eta = np.mean(eta_period)
            ldt_95 = var_calc.compute_var(eta_period, 0.95)
            ldt_99 = var_calc.compute_var(eta_period, 0.99)
            gap_95 = var_calc.compute_entropy_gap(eta_period, 0.95)
            cvar_95 = var_calc.compute_expected_shortfall(eta_period, 0.95)
            
            metrics[period_name] = {
                'mean': mean_eta,
                'ldt_95': ldt_95,
                'ldt_99': ldt_99,
                'gap_95': gap_95,
                'gap_99': ldt_99 - mean_eta,
                'cvar_95': cvar_95,
                'samples': eta_period
            }
            
            print(f"{period_name:<15} {mean_eta:>8.2f}d   {ldt_95:>8.2f}d   "
                  f"{gap_95:>8.2f}d   {cvar_95:>8.2f}d")
    
    print("-" * 70)
    
    # Key insights
    normal_gap = metrics['Normal']['gap_95']
    holiday_gap = metrics['Holiday']['gap_95']
    expansion = holiday_gap / normal_gap
    
    print(f"\n🔍 Key Insights:")
    print(f"  • Entropy gap expands by {expansion:.1f}× during holiday period")
    print(f"  • Normal period buffer: {normal_gap:.1f} days")
    print(f"  • Holiday period buffer: {holiday_gap:.1f} days")
    print(f"  • Additional risk buffer needed: {holiday_gap - normal_gap:.1f} days")
    
    # Shelf life analysis for fresh produce
    shelf_life = 14.0  # days
    holiday_ldt_99 = metrics['Holiday']['ldt_99']
    
    if holiday_ldt_99 > shelf_life:
        print(f"\n⚠️  RISK ALERT:")
        print(f"  LDT₀.₉₉ ({holiday_ldt_99:.1f}d) exceeds shelf life ({shelf_life:.0f}d)")
        print(f"  Recommendation: Consider alternative routing or pre-positioning")
    
    return metrics


def generate_visualizations(v_history, x, t, metrics, t_start, t_end):
    """
    Generate visualization plots.
    """
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)
    
    viz = LogisticsVisualizer()
    
    # Plot 1: Velocity timeseries
    print("\n📈 Plot 1: Velocity timeseries...")
    mid_idx = len(x) // 2
    v_mid = v_history[:, mid_idx]
    
    fig, ax = viz.plot_velocity_timeseries(
        t, v_mid,
        holiday_periods=[(t_start, t_end)],
        title="Spring Festival: Flow Velocity Evolution"
    )
    plt.savefig('../figures/spring_festival_velocity.png', dpi=150, 
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("   Saved: figures/spring_festival_velocity.png")
    
    # Plot 2: VaR dashboard
    print("\n📈 Plot 2: VaR dashboard...")
    eta_samples = {name: m['samples'] for name, m in metrics.items()}
    
    fig, axes = viz.plot_var_dashboard(
        eta_samples, metrics,
        save_path='../figures/spring_festival_var_dashboard.png'
    )
    plt.close()
    print("   Saved: figures/spring_festival_var_dashboard.png")
    
    # Plot 3: 2D velocity field heatmap
    print("\n📈 Plot 3: Velocity field heatmap...")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    im = ax.imshow(v_history.T, aspect='auto', cmap='RdYlGn', 
                   extent=[0, 30, 0, 10], origin='lower')
    ax.axvspan(t_start, t_end, alpha=0.2, color='red', label='Holiday Period')
    ax.set_xlabel('Time (days)', fontsize=12)
    ax.set_ylabel('Network Position', fontsize=12)
    ax.set_title('Spring Festival: Velocity Field Spatiotemporal Evolution', 
                fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Velocity')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('../figures/spring_festival_heatmap.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("   Saved: figures/spring_festival_heatmap.png")
    
    print("\n✅ All visualizations generated!")


def main():
    """
    Main execution function.
    """
    # Configure scenario
    (ns, carrier_params, commodity_params, 
     t_pre, t_start, t_end, jump_params) = configure_spring_festival_scenario()
    
    # Run simulation
    v_history, x, t, mu_func = run_simulation(
        ns, carrier_params, commodity_params,
        t_pre, t_start, t_end, jump_params
    )
    
    # Analyze results
    metrics = analyze_results(v_history, x, t, mu_func, t_start, t_end)
    
    # Generate visualizations
    generate_visualizations(v_history, x, t, metrics, t_start, t_end)
    
    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)
    print("\nOutput files:")
    print("  • figures/spring_festival_velocity.png")
    print("  • figures/spring_festival_var_dashboard.png")
    print("  • figures/spring_festival_heatmap.png")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()

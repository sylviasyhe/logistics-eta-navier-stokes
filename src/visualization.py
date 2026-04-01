"""
Visualization Tools for Logistics N-S Simulation
================================================
Generate plots for ETA distributions, velocity fields, and risk analysis.

Author: Research Team
Date: April 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.collections import LineCollection
import matplotlib.patches as mpatches
from typing import Dict, List, Tuple, Optional


class LogisticsVisualizer:
    """
    Visualization toolkit for logistics simulation results.
    """
    
    def __init__(self, style: str = 'seaborn-v0_8-whitegrid'):
        """
        Initialize visualizer.
        
        Args:
            style: Matplotlib style
        """
        plt.style.use(style)
        self.colors = {
            'normal': '#27ae60',
            'pre_holiday': '#f39c12',
            'holiday': '#e74c3c',
            'recovery': '#3498db',
            'text': '#2c3e50',
            'grid': '#bdc3c7'
        }
    
    def plot_velocity_timeseries(self,
                                  t: np.ndarray,
                                  v: np.ndarray,
                                  holiday_periods: List[Tuple[float, float]],
                                  title: str = "Velocity Field Evolution",
                                  save_path: Optional[str] = None):
        """
        Plot velocity field over time with holiday annotations.
        
        Args:
            t: Time array
            v: Velocity array [time, spatial] or [time]
            holiday_periods: List of (start, end) tuples for holidays
            title: Plot title
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # If 2D, plot mean velocity
        if v.ndim == 2:
            v_plot = np.mean(v, axis=1)
        else:
            v_plot = v
        
        # Plot velocity
        ax.plot(t, v_plot, 'b-', linewidth=2, label='Flow Velocity')
        
        # Highlight holiday periods
        for i, (start, end) in enumerate(holiday_periods):
            ax.axvspan(start, end, alpha=0.2, color='red', 
                      label='Holiday' if i == 0 else '')
        
        # Add phase annotations
        if len(holiday_periods) > 0:
            pre_start = max(0, holiday_periods[0][0] - 4)
            ax.annotate('Pre-holiday\nSurge', 
                       xy=(holiday_periods[0][0] - 2, np.max(v_plot) * 0.9),
                       fontsize=10, ha='center', color=self.colors['pre_holiday'],
                       fontweight='bold')
            ax.annotate('Holiday\nDisruption',
                       xy=((holiday_periods[0][0] + holiday_periods[0][1])/2, np.min(v_plot) * 1.1),
                       fontsize=10, ha='center', color=self.colors['holiday'],
                       fontweight='bold')
        
        ax.set_xlabel('Time (days)', fontsize=12)
        ax.set_ylabel('Relative Velocity', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
        
        return fig, ax
    
    def plot_eta_distribution(self,
                             eta_samples: Dict[str, np.ndarray],
                             alphas: List[float] = [0.95, 0.99],
                             title: str = "ETA Distribution by Period",
                             save_path: Optional[str] = None):
        """
        Plot ETA distributions for different periods with VaR markers.
        
        Args:
            eta_samples: Dict mapping period names to ETA samples
            alphas: Confidence levels for VaR
            title: Plot title
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = [self.colors['normal'], self.colors['pre_holiday'], 
                 self.colors['holiday'], self.colors['recovery']]
        
        bins = np.linspace(0, 25, 80)
        
        for i, (period, samples) in enumerate(eta_samples.items()):
            color = colors[i % len(colors)]
            ax.hist(samples, bins=bins, alpha=0.5, label=period, 
                   color=color, density=True)
            
            # Mark VaR points
            for alpha in alphas:
                var = np.percentile(samples, alpha * 100)
                ax.axvline(var, color=color, linestyle='--', linewidth=1.5, alpha=0.7)
        
        ax.set_xlabel('ETA (days)', fontsize=12)
        ax.set_ylabel('Probability Density', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
        
        return fig, ax
    
    def plot_entropy_gap(self,
                        periods: List[str],
                        gaps: Dict[str, List[float]],
                        title: str = "System Entropy Gap by Period",
                        save_path: Optional[str] = None):
        """
        Plot entropy gap (LDT - Mean ETA) for different periods.
        
        Args:
            periods: List of period names
            gaps: Dict mapping gap type to list of values
            title: Plot title
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(periods))
        width = 0.35
        
        if 'gap_95' in gaps:
            bars1 = ax.bar(x - width/2, gaps['gap_95'], width, 
                          label='Gap₀.₉₅', color=self.colors['normal'], alpha=0.8)
        if 'gap_99' in gaps:
            bars2 = ax.bar(x + width/2, gaps['gap_99'], width,
                          label='Gap₀.₉₉', color=self.colors['holiday'], alpha=0.8)
        
        # Add value labels
        for bars in [bars1, bars2] if 'gap_99' in gaps else [bars1]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.1f}d',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Period', fontsize=12)
        ax.set_ylabel('Entropy Gap (days)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(periods)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
        
        return fig, ax
    
    def plot_var_dashboard(self,
                          eta_samples: Dict[str, np.ndarray],
                          metrics: Dict[str, Dict[str, float]],
                          save_path: Optional[str] = None):
        """
        Create comprehensive VaR dashboard.
        
        Args:
            eta_samples: ETA samples by period
            metrics: Computed metrics by period
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: ETA distributions
        ax1 = axes[0, 0]
        bins = np.linspace(0, 25, 60)
        colors = [self.colors['normal'], self.colors['pre_holiday'], 
                 self.colors['holiday'], self.colors['recovery']]
        
        for i, (period, samples) in enumerate(eta_samples.items()):
            ax1.hist(samples, bins=bins, alpha=0.5, label=period, 
                    color=colors[i % len(colors)], density=True)
        ax1.set_xlabel('ETA (days)')
        ax1.set_ylabel('Probability Density')
        ax1.set_title('(a) ETA Distribution by Period')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Mean ETA comparison
        ax2 = axes[0, 1]
        periods = list(metrics.keys())
        mean_etas = [m['mean'] for m in metrics.values()]
        ldt_95s = [m['ldt_95'] for m in metrics.values()]
        
        x = np.arange(len(periods))
        width = 0.35
        ax2.bar(x - width/2, mean_etas, width, label='Mean ETA', color=self.colors['normal'])
        ax2.bar(x + width/2, ldt_95s, width, label='LDT₀.₉₅', color=self.colors['holiday'])
        ax2.set_xlabel('Period')
        ax2.set_ylabel('Days')
        ax2.set_title('(b) Mean ETA vs LDT')
        ax2.set_xticks(x)
        ax2.set_xticklabels(periods, rotation=15)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Plot 3: Entropy gap
        ax3 = axes[1, 0]
        gaps_95 = [m['gap_95'] for m in metrics.values()]
        gaps_99 = [m['gap_99'] for m in metrics.values()]
        
        ax3.bar(x - width/2, gaps_95, width, label='Gap₀.₉₅', color='#3498db')
        ax3.bar(x + width/2, gaps_99, width, label='Gap₀.₉₉', color='#e74c3c')
        ax3.set_xlabel('Period')
        ax3.set_ylabel('Days')
        ax3.set_title('(c) Entropy Gap (System Uncertainty)')
        ax3.set_xticks(x)
        ax3.set_xticklabels(periods, rotation=15)
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Plot 4: Risk metrics table
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        table_data = []
        for period, m in metrics.items():
            table_data.append([
                period,
                f"{m['mean']:.1f}",
                f"{m['ldt_95']:.1f}",
                f"{m['gap_95']:.1f}",
                f"{m['cvar_95']:.1f}"
            ])
        
        table = ax4.table(
            cellText=table_data,
            colLabels=['Period', 'Mean', 'LDT₀.₉₅', 'Gap₀.₉₅', 'CVaR₀.₉₅'],
            loc='center',
            cellLoc='center',
            colColours=['#f0f0f0']*5
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax4.set_title('(d) Risk Metrics Summary', y=0.8, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
        
        return fig, axes
    
    def plot_flow_field(self,
                       X: np.ndarray,
                       Y: np.ndarray,
                       U: np.ndarray,
                       V: np.ndarray,
                       obstacles: Optional[List[Tuple[float, float, float]]] = None,
                       title: str = "Flow Field Visualization",
                       save_path: Optional[str] = None):
        """
        Plot 2D flow field with velocity vectors and streamlines.
        
        Args:
            X: X coordinates
            Y: Y coordinates
            U: X velocity component
            V: Y velocity component
            obstacles: List of (x, y, radius) for obstacles
            title: Plot title
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Velocity magnitude
        magnitude = np.sqrt(U**2 + V**2)
        
        # Quiver plot
        q = ax.quiver(X, Y, U, V, magnitude, cmap='viridis', 
                     scale=10, width=0.003)
        
        # Streamlines
        ax.streamplot(X, Y, U, V, color=magnitude, cmap='viridis',
                     density=1.5, linewidth=1.5, arrowstyle='->')
        
        # Plot obstacles
        if obstacles:
            for x, y, r in obstacles:
                circle = plt.Circle((x, y), r, facecolor='red', 
                                  edgecolor='darkred', alpha=0.5)
                ax.add_patch(circle)
        
        ax.set_xlabel('Network Distance')
        ax.set_ylabel('Network Distance')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_aspect('equal')
        
        # Colorbar
        cbar = plt.colorbar(q, ax=ax, shrink=0.5)
        cbar.set_label('Velocity Magnitude')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
        
        return fig, ax


def generate_example_visualizations():
    """
    Generate example visualizations for the paper.
    """
    print("Generating example visualizations...")
    
    viz = LogisticsVisualizer()
    
    # Generate synthetic data
    np.random.seed(42)
    
    # ETA samples for different periods
    eta_samples = {
        'Normal': np.random.normal(4.5, 0.8, 5000),
        'Pre-holiday': np.random.normal(5.2, 1.2, 5000),
        'Holiday': np.random.gamma(2, 4.35, 5000),
        'Recovery': np.random.normal(6.3, 2.0, 5000)
    }
    
    # Compute metrics
    from logistics_ns_solver import VaRCalculator
    var_calc = VaRCalculator()
    
    metrics = {}
    for period, samples in eta_samples.items():
        metrics[period] = {
            'mean': np.mean(samples),
            'ldt_95': var_calc.compute_var(samples, 0.95),
            'ldt_99': var_calc.compute_var(samples, 0.99),
            'gap_95': var_calc.compute_entropy_gap(samples, 0.95),
            'gap_99': var_calc.compute_entropy_gap(samples, 0.99),
            'cvar_95': var_calc.compute_expected_shortfall(samples, 0.95)
        }
    
    # Generate plots
    viz.plot_var_dashboard(eta_samples, metrics, 
                          save_path='/mnt/okcomputer/output/logistics-eta-pde/figures/var_dashboard.png')
    
    print("Visualizations saved to figures/ directory")


if __name__ == "__main__":
    generate_example_visualizations()

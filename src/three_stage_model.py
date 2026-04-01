"""
Three-Stage ETA Prediction Model
=================================
Stage 1: Merchant Shipping (Gaussian Process)
Stage 2: Inter-Hub Transportation (Navier-Stokes)
Stage 3: Last-Mile Delivery (Diffusion)

Author: Research Team
Date: April 2026
"""

import numpy as np
from scipy.stats import norm, gamma
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class MerchantProfile:
    """Merchant shipping behavior profile."""
    merchant_id: str
    tier: str  # 'S', 'A', 'B', 'C'
    mu_base: float  # hours
    sigma_base: float  # hours
    category: str
    
    # Tier-specific parameters
    TIER_PARAMS = {
        'S': {'mu_range': (2, 4), 'sigma_range': (0.5, 1)},
        'A': {'mu_range': (6, 12), 'sigma_range': (2, 4)},
        'B': {'mu_range': (12, 24), 'sigma_range': (4, 8)},
        'C': {'mu_range': (24, 72), 'sigma_range': (8, 24)},
    }


class GaussianMerchantModel:
    """
    Stage 1: Merchant shipping behavior modeled as Gaussian process.
    
    T_merchant ~ N(μ_merchant, σ_merchant²)
    """
    
    def __init__(self):
        self.merchant_profiles = {}
        
    def register_merchant(self, profile: MerchantProfile):
        """Register a merchant profile."""
        self.merchant_profiles[profile.merchant_id] = profile
    
    def compute_mu_merchant(self,
                           merchant_id: str,
                           order_time: float,
                           holiday_time: Optional[float] = None) -> float:
        """
        Compute expected merchant shipping time.
        
        Args:
            merchant_id: Merchant identifier
            order_time: Time of order placement (hours)
            holiday_time: Time of upcoming holiday (if any)
            
        Returns:
            Expected shipping time (hours)
        """
        if merchant_id not in self.merchant_profiles:
            # Default to tier C if unknown
            profile = MerchantProfile(
                merchant_id=merchant_id,
                tier='C',
                mu_base=36,
                sigma_base=12,
                category='general'
            )
        else:
            profile = self.merchant_profiles[merchant_id]
        
        mu = profile.mu_base
        
        # Time-of-day effect (merchants ship less at night)
        hour_of_day = order_time % 24
        if hour_of_day < 6 or hour_of_day > 22:
            mu *= 1.3  # 30% longer at night
        
        # Day-of-week effect
        day_of_week = (order_time // 24) % 7
        if day_of_week in [5, 6]:  # Weekend
            mu *= 1.2
        
        # Holiday proximity effect
        if holiday_time is not None:
            time_to_holiday = holiday_time - order_time
            if 0 < time_to_holiday < 72:  # Within 3 days of holiday
                # Pre-holiday surge: faster shipping
                mu *= 0.8
            elif -48 < time_to_holiday < 0:  # During holiday
                # Holiday delay
                mu *= 2.5
        
        return mu
    
    def compute_sigma_merchant(self, merchant_id: str) -> float:
        """Compute merchant shipping time standard deviation."""
        if merchant_id not in self.merchant_profiles:
            return 12.0  # Default
        return self.merchant_profiles[merchant_id].sigma_base
    
    def sample_shipping_time(self,
                            merchant_id: str,
                            order_time: float,
                            holiday_time: Optional[float] = None,
                            n_samples: int = 1) -> np.ndarray:
        """
        Sample merchant shipping times.
        
        Args:
            merchant_id: Merchant identifier
            order_time: Order placement time
            holiday_time: Holiday time (if any)
            n_samples: Number of samples
            
        Returns:
            Array of shipping times (hours)
        """
        mu = self.compute_mu_merchant(merchant_id, order_time, holiday_time)
        sigma = self.compute_sigma_merchant(merchant_id)
        return np.random.normal(mu, sigma, n_samples)
    
    def fit_from_history(self,
                        merchant_id: str,
                        shipping_times: List[float],
                        order_times: List[float]) -> MerchantProfile:
        """
        Fit merchant profile from historical data.
        
        Args:
            merchant_id: Merchant identifier
            shipping_times: Historical shipping times (hours)
            order_times: Corresponding order times
            
        Returns:
            Fitted merchant profile
        """
        mu_est = np.mean(shipping_times)
        sigma_est = np.std(shipping_times)
        
        # Determine tier based on performance
        if mu_est < 6 and sigma_est < 2:
            tier = 'S'
        elif mu_est < 18 and sigma_est < 6:
            tier = 'A'
        elif mu_est < 36 and sigma_est < 12:
            tier = 'B'
        else:
            tier = 'C'
        
        profile = MerchantProfile(
            merchant_id=merchant_id,
            tier=tier,
            mu_base=mu_est,
            sigma_base=sigma_est,
            category='general'
        )
        
        self.register_merchant(profile)
        return profile


class LastMileDiffusionModel:
    """
    Stage 3: Last-mile delivery as diffusion process.
    """
    
    def __init__(self,
                 v_lastmile_base: float = 20.0,  # km/day
                 t_service: float = 3.0,  # minutes per stop
                 d_diffusion: float = 5.0):  # diffusion coefficient
        """
        Initialize last-mile model.
        
        Args:
            v_lastmile_base: Base last-mile speed (km/day)
            t_service: Time per delivery stop (minutes)
            d_diffusion: Diffusion coefficient
        """
        self.v_lastmile_base = v_lastmile_base
        self.t_service = t_service
        self.d_diffusion = d_diffusion
    
    def compute_lastmile_time(self,
                             distance: float,
                             address_density: float,
                             n_stops_ahead: int,
                             time_of_day: float) -> float:
        """
        Compute last-mile delivery time.
        
        Args:
            distance: Distance from last hub to destination (km)
            address_density: Addresses per km²
            n_stops_ahead: Number of deliveries before this one
            time_of_day: Current time (hour)
            
        Returns:
            Last-mile time (days)
        """
        # Base travel time
        v_effective = self.v_lastmile_base
        
        # Address density effect (higher density → slower due to traffic)
        if address_density > 1000:  # Urban
            v_effective *= 0.7
        elif address_density > 500:  # Suburban
            v_effective *= 0.85
        # Rural: no change
        
        # Time-of-day effect
        if 8 <= time_of_day <= 10 or 17 <= time_of_day <= 19:  # Rush hour
            v_effective *= 0.6
        elif 22 <= time_of_day or time_of_day <= 6:  # Night
            v_effective *= 1.1
        
        travel_time = distance / v_effective
        
        # Service time for prior stops
        service_time = (n_stops_ahead * self.t_service) / (60 * 24)  # Convert to days
        
        # Diffusion uncertainty
        diffusion_delay = np.random.exponential(self.d_diffusion / 24)  # Days
        
        return travel_time + service_time + diffusion_delay


@dataclass
class RouteEdge:
    """Edge in logistics network."""
    origin: str
    destination: str
    distance: float  # km
    route_type: str  # 'domestic', 'crossborder', 'mixed'
    carrier: str
    base_capacity: float  # packages/day


class MultiRouteSelector:
    """
    Multi-route selection using variational formulation.
    """
    
    def __init__(self,
                 lambda_risk: float = 0.5,
                 gamma_cost: float = 0.1):
        """
        Initialize route selector.
        
        Args:
            lambda_risk: Risk aversion parameter
            gamma_cost: Cost sensitivity
        """
        self.lambda_risk = lambda_risk
        self.gamma_cost = gamma_cost
        self.network = {}  # Adjacency list
        
    def add_edge(self, edge: RouteEdge):
        """Add edge to network."""
        if edge.origin not in self.network:
            self.network[edge.origin] = []
        self.network[edge.origin].append(edge)
    
    def compute_edge_viscosity(self,
                              edge: RouteEdge,
                              carrier_params: Dict,
                              commodity_params: Dict,
                              t: float) -> float:
        """
        Compute time-varying edge viscosity.
        
        Args:
            edge: Route edge
            carrier_params: Carrier features
            commodity_params: Commodity features
            t: Current time
            
        Returns:
            Edge viscosity
        """
        # Base viscosity from carrier and commodity
        from logistics_ns_solver_v2 import LogisticsNSEquationV2
        ns = LogisticsNSEquationV2()
        
        mu_base = ns.combined_viscosity(
            carrier_params,
            commodity_params,
            shear_rate=1.0
        )
        
        # Route type adjustment
        route_multiplier = {
            'domestic': 1.2,
            'crossborder': 2.5,
            'mixed': 1.8
        }
        
        mu = mu_base * route_multiplier.get(edge.route_type, 1.5)
        
        return mu
    
    def k_shortest_paths(self,
                        origin: str,
                        destination: str,
                        k: int = 5) -> List[List[RouteEdge]]:
        """
        Find k shortest paths by distance.
        
        Args:
            origin: Origin hub
            destination: Destination hub
            k: Number of paths
            
        Returns:
            List of paths (each path is list of edges)
        """
        # Simple BFS-based k-shortest (Yen's algorithm simplified)
        paths = []
        
        def dfs(current: str, target: str, path: List[RouteEdge], visited: set):
            if len(paths) >= k:
                return
            if current == target:
                paths.append(path.copy())
                return
            if current not in self.network:
                return
            
            for edge in self.network[current]:
                if edge.destination not in visited:
                    visited.add(edge.destination)
                    path.append(edge)
                    dfs(edge.destination, target, path, visited)
                    path.pop()
                    visited.remove(edge.destination)
        
        dfs(origin, destination, [], {origin})
        
        # Sort by total distance
        paths.sort(key=lambda p: sum(e.distance for e in p))
        
        return paths[:k]
    
    def evaluate_path(self,
                     path: List[RouteEdge],
                     merchant_model: GaussianMerchantModel,
                     lastmile_model: LastMileDiffusionModel,
                     merchant_id: str,
                     order_time: float,
                     n_samples: int = 100) -> Dict:
        """
        Evaluate a candidate path.
        
        Args:
            path: List of edges
            merchant_model: Merchant shipping model
            lastmile_model: Last-mile model
            merchant_id: Merchant identifier
            order_time: Order time
            n_samples: Number of Monte Carlo samples
            
        Returns:
            Dictionary with metrics
        """
        # Stage 1: Merchant shipping
        t_merchant_samples = merchant_model.sample_shipping_time(
            merchant_id, order_time, n_samples=n_samples
        ) / 24  # Convert to days
        
        # Stage 2: Inter-hub transport
        total_distance = sum(e.distance for e in path)
        
        # Simplified transport time (would use full N-S in production)
        avg_speed = 500 if path[0].route_type == 'domestic' else 300  # km/day
        t_transport = total_distance / avg_speed
        
        # Add variability based on route type
        if any(e.route_type == 'crossborder' for e in path):
            # Customs delay (Gamma distribution)
            customs_delay = gamma.rvs(2, scale=1.0, size=n_samples)
        else:
            customs_delay = np.zeros(n_samples)
        
        # Stage 3: Last-mile
        last_edge = path[-1]
        t_lastmile = lastmile_model.compute_lastmile_time(
            distance=10,  # Assume 10km last mile
            address_density=500,
            n_stops_ahead=5,
            time_of_day=14
        )
        
        # Total ETA
        t_total_samples = t_merchant_samples + t_transport + customs_delay + t_lastmile
        
        # Compute metrics
        mean_eta = np.mean(t_total_samples)
        var_eta = np.var(t_total_samples)
        var_95 = np.percentile(t_total_samples, 95)
        var_99 = np.percentile(t_total_samples, 99)
        
        # Cost
        path_cost = sum(e.distance * 0.05 for e in path)  # $0.05 per km
        
        # Physics-informed score
        score = mean_eta + self.lambda_risk * (var_95 - mean_eta) + self.gamma_cost * path_cost
        
        return {
            'mean_eta': mean_eta,
            'var_eta': var_eta,
            'var_95': var_95,
            'var_99': var_99,
            'cost': path_cost,
            'score': score,
            'path': path,
            'stage_breakdown': {
                'merchant': np.mean(t_merchant_samples),
                'transport': t_transport,
                'customs': np.mean(customs_delay),
                'lastmile': t_lastmile
            }
        }
    
    def select_optimal_route(self,
                            origin: str,
                            destination: str,
                            merchant_model: GaussianMerchantModel,
                            lastmile_model: LastMileDiffusionModel,
                            merchant_id: str,
                            order_time: float,
                            k: int = 5) -> Dict:
        """
        Select optimal route using variational formulation.
        
        Args:
            origin: Origin hub
            destination: Destination hub
            merchant_model: Merchant model
            lastmile_model: Last-mile model
            merchant_id: Merchant ID
            order_time: Order time
            k: Number of candidate paths
            
        Returns:
            Best route with metrics
        """
        # Generate candidate paths
        candidate_paths = self.k_shortest_paths(origin, destination, k)
        
        if not candidate_paths:
            return None
        
        # Evaluate each path
        path_evaluations = []
        for path in candidate_paths:
            eval_result = self.evaluate_path(
                path, merchant_model, lastmile_model,
                merchant_id, order_time
            )
            path_evaluations.append(eval_result)
        
        # Select best
        best = min(path_evaluations, key=lambda x: x['score'])
        
        return best


class ThreeStageETAModel:
    """
    Complete three-stage ETA prediction model.
    """
    
    def __init__(self):
        self.merchant_model = GaussianMerchantModel()
        self.lastmile_model = LastMileDiffusionModel()
        self.route_selector = MultiRouteSelector()
    
    def predict_eta(self,
                   origin: str,
                   destination: str,
                   merchant_id: str,
                   order_time: float,
                   commodity_type: str = 'general',
                   n_samples: int = 100) -> Dict:
        """
        Predict ETA using three-stage model.
        
        Args:
            origin: Origin hub
            destination: Destination hub
            merchant_id: Merchant identifier
            order_time: Order placement time
            commodity_type: Commodity category
            n_samples: Monte Carlo samples
            
        Returns:
            ETA prediction with uncertainty
        """
        # Select optimal route
        route_result = self.route_selector.select_optimal_route(
            origin, destination,
            self.merchant_model, self.lastmile_model,
            merchant_id, order_time
        )
        
        if route_result is None:
            return {'error': 'No route found'}
        
        return {
            'eta_mean': route_result['mean_eta'],
            'eta_std': np.sqrt(route_result['var_eta']),
            'ldt_95': route_result['var_95'],
            'ldt_99': route_result['var_99'],
            'route': [(e.origin, e.destination) for e in route_result['path']],
            'cost': route_result['cost'],
            'stage_breakdown': route_result['stage_breakdown']
        }


def example_three_stage():
    """Example usage of three-stage model."""
    print("=" * 70)
    print("Three-Stage ETA Model Example")
    print("=" * 70)
    
    # Initialize model
    model = ThreeStageETAModel()
    
    # Register merchants
    merchants = [
        MerchantProfile('M001', 'S', 3, 0.8, 'electronics'),
        MerchantProfile('M002', 'B', 18, 6, 'clothing'),
        MerchantProfile('M003', 'C', 36, 12, 'general'),
    ]
    for m in merchants:
        model.merchant_model.register_merchant(m)
    
    # Build network
    edges = [
        RouteEdge('Guangzhou', 'Shenzhen', 150, 'domestic', 'CarrierA', 1000),
        RouteEdge('Shenzhen', 'HongKong', 50, 'crossborder', 'CarrierA', 500),
        RouteEdge('HongKong', 'LosAngeles', 12000, 'crossborder', 'CarrierB', 200),
        RouteEdge('Guangzhou', 'Shanghai', 1400, 'domestic', 'CarrierC', 1500),
        RouteEdge('Shanghai', 'LosAngeles', 11000, 'crossborder', 'CarrierB', 250),
    ]
    for e in edges:
        model.route_selector.add_edge(e)
    
    # Predict ETA for different merchants
    print("\n📦 ETA Predictions:")
    print("-" * 70)
    print(f"{'Merchant':<12} {'Tier':<8} {'ETA Mean':<12} {'LDT₉₅':<12} {'Route':<30}")
    print("-" * 70)
    
    for merchant in merchants:
        result = model.predict_eta(
            'Guangzhou', 'LosAngeles',
            merchant.merchant_id,
            order_time=0
        )
        
        route_str = ' → '.join([f"{o}-{d}" for o, d in result['route']])
        print(f"{merchant.merchant_id:<12} {merchant.tier:<8} "
              f"{result['eta_mean']:>8.2f}d   {result['ldt_95']:>8.2f}d   {route_str}")
    
    print("-" * 70)
    
    # Show stage breakdown
    print("\n📊 Stage Breakdown (Merchant M001):")
    result = model.predict_eta('Guangzhou', 'LosAngeles', 'M001', 0)
    breakdown = result['stage_breakdown']
    print(f"  Merchant shipping: {breakdown['merchant']:.2f} days")
    print(f"  Inter-hub transport: {breakdown['transport']:.2f} days")
    print(f"  Customs delay: {breakdown['customs']:.2f} days")
    print(f"  Last-mile: {breakdown['lastmile']:.2f} days")
    print(f"  Total: {sum(breakdown.values()):.2f} days")
    
    print("\n✅ Example complete!")


if __name__ == "__main__":
    example_three_stage()

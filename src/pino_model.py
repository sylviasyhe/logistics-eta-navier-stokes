"""
Physics-Informed Neural Operator (PINO) for Logistics ETA Prediction
====================================================================
Hybrid architecture combining Fourier Neural Operators with physics constraints.

Author: Research Team
Date: April 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List


class SpectralConv1d(nn.Module):
    """
    1D Spectral convolution layer for Fourier Neural Operator.
    Operates in Fourier space with learnable complex weights.
    """
    
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            modes: Number of Fourier modes to retain
        """
        super(SpectralConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        
        # Complex weights for Fourier modes
        self.scale = 1.0 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes, 2)
        )
    
    def compl_mul1d(self, input: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Complex multiplication in Fourier space.
        
        Args:
            input: [batch, in_channels, modes]
            weights: [in_channels, out_channels, modes]
            
        Returns:
            [batch, out_channels, modes]
        """
        # (batch, in_channel, modes), (in_channel, out_channel, modes) -> (batch, out_channel, modes)
        return torch.einsum("bix,iox->box", input, weights)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through spectral convolution.
        
        Args:
            x: Input tensor [batch, channels, spatial]
            
        Returns:
            Output tensor [batch, channels, spatial]
        """
        batchsize = x.shape[0]
        
        # FFT
        x_ft = torch.fft.rfft(x)
        
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1, 
                            dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes] = self.compl_mul1d(
            x_ft[:, :, :self.modes], 
            torch.view_as_complex(self.weights)
        )
        
        # IFFT
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


class FourierLayer(nn.Module):
    """
    Fourier layer combining spectral and pointwise convolutions.
    """
    
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super(FourierLayer, self).__init__()
        self.spectral_conv = SpectralConv1d(in_channels, out_channels, modes)
        self.pointwise_conv = nn.Conv1d(in_channels, out_channels, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: F^{-1}(R * F(x)) + W(x)
        """
        return self.spectral_conv(x) + self.pointwise_conv(x)


class LogisticsFNO(nn.Module):
    """
    Fourier Neural Operator for logistics velocity field prediction.
    """
    
    def __init__(self, 
                 modes: int = 12,
                 width: int = 64,
                 n_layers: int = 4,
                 input_dim: int = 3,  # [v, mu, t]
                 output_dim: int = 1):
        """
        Args:
            modes: Number of Fourier modes
            width: Channel width
            n_layers: Number of FNO layers
            input_dim: Input feature dimension
            output_dim: Output feature dimension
        """
        super(LogisticsFNO, self).__init__()
        
        self.modes = modes
        self.width = width
        self.n_layers = n_layers
        
        # Input projection
        self.input_proj = nn.Conv1d(input_dim, width, 1)
        
        # Fourier layers
        self.fourier_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.fourier_layers.append(FourierLayer(width, width, modes))
        
        # Output projection
        self.output_proj = nn.Conv1d(width, output_dim, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through FNO.
        
        Args:
            x: Input [batch, input_dim, spatial]
            
        Returns:
            Output [batch, output_dim, spatial]
        """
        # Input projection
        x = self.input_proj(x)
        
        # Fourier layers with residual connections
        for layer in self.fourier_layers:
            x = F.gelu(layer(x) + x)
        
        # Output projection
        x = self.output_proj(x)
        return x


class InputEncoder(nn.Module):
    """
    Encodes logistics inputs into feature embeddings.
    """
    
    def __init__(self,
                 merchant_hist_len: int = 10,
                 merchant_hidden: int = 32,
                 carrier_feat_dim: int = 5,
                 commodity_feat_dim: int = 3,
                 embedding_dim: int = 64):
        super(InputEncoder, self).__init__()
        
        # Merchant history encoder (LSTM)
        self.merchant_lstm = nn.LSTM(
            input_size=3,  # [ship_time, volume, destination]
            hidden_size=merchant_hidden,
            num_layers=2,
            batch_first=True
        )
        
        # Carrier-commodity joint encoder
        self.carrier_commodity_mlp = nn.Sequential(
            nn.Linear(carrier_feat_dim + commodity_feat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # Holiday positional encoding
        self.holiday_encoder = nn.Sequential(
            nn.Linear(2, 32),  # [sin, cos] encoding
            nn.ReLU()
        )
        
        # Combined projection
        self.combined_proj = nn.Linear(merchant_hidden + 32 + 32, embedding_dim)
        
    def positional_encoding(self, t: torch.Tensor, period: float = 30.0) -> torch.Tensor:
        """
        Sinusoidal positional encoding for time.
        
        Args:
            t: Time values [batch]
            period: Period for encoding
            
        Returns:
            Encoded time [batch, 2]
        """
        angle = t * 2 * np.pi / period
        return torch.stack([torch.sin(angle), torch.cos(angle)], dim=-1)
    
    def forward(self,
               merchant_history: torch.Tensor,
               carrier_features: torch.Tensor,
               commodity_features: torch.Tensor,
               time: torch.Tensor) -> torch.Tensor:
        """
        Encode all inputs into embedding.
        
        Args:
            merchant_history: [batch, hist_len, 3]
            carrier_features: [batch, carrier_feat_dim]
            commodity_features: [batch, commodity_feat_dim]
            time: [batch]
            
        Returns:
            Embedding [batch, embedding_dim]
        """
        # Merchant encoding
        _, (h_n, _) = self.merchant_lstm(merchant_history)
        merchant_emb = h_n[-1]  # [batch, merchant_hidden]
        
        # Carrier-commodity encoding
        cc_input = torch.cat([carrier_features, commodity_features], dim=-1)
        cc_emb = self.carrier_commodity_mlp(cc_input)  # [batch, 32]
        
        # Holiday encoding
        time_enc = self.positional_encoding(time)
        time_emb = self.holiday_encoder(time_enc)  # [batch, 32]
        
        # Combine
        combined = torch.cat([merchant_emb, cc_emb, time_emb], dim=-1)
        embedding = self.combined_proj(combined)
        
        return embedding


class DistributionalHead(nn.Module):
    """
    Outputs ETA probability distribution via quantile regression.
    """
    
    def __init__(self, 
                 input_dim: int = 64,
                 hidden_dim: int = 128,
                 n_quantiles: int = 9):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            n_quantiles: Number of quantiles to predict
        """
        super(DistributionalHead, self).__init__()
        
        self.n_quantiles = n_quantiles
        self.quantiles = torch.linspace(0.1, 0.9, n_quantiles)
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_quantiles)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict quantiles.
        
        Args:
            x: Input features [batch, input_dim]
            
        Returns:
            Quantile predictions [batch, n_quantiles]
        """
        return self.mlp(x)
    
    def compute_var(self, predictions: torch.Tensor, alpha: float = 0.95) -> torch.Tensor:
        """
        Extract VaR (LDT) from quantile predictions.
        
        Args:
            predictions: [batch, n_quantiles]
            alpha: Confidence level
            
        Returns:
            VaR values [batch]
        """
        # Find closest quantile
        idx = torch.argmin(torch.abs(self.quantiles - alpha))
        return predictions[:, idx]


class HybridPINO(nn.Module):
    """
    Hybrid Physics-Informed Neural Operator for Logistics ETA.
    Combines FNO for spatial-temporal dynamics with physics constraints.
    """
    
    def __init__(self,
                 fno_modes: int = 12,
                 fno_width: int = 64,
                 fno_layers: int = 4,
                 embedding_dim: int = 64,
                 n_quantiles: int = 9):
        super(HybridPINO, self).__init__()
        
        # Input encoder
        self.encoder = InputEncoder(embedding_dim=embedding_dim)
        
        # FNO for velocity field
        self.fno = LogisticsFNO(
            modes=fno_modes,
            width=fno_width,
            n_layers=fno_layers,
            input_dim=embedding_dim + 1,  # embedding + initial velocity
            output_dim=1
        )
        
        # Distributional head
        self.dist_head = DistributionalHead(
            input_dim=fno_width,
            n_quantiles=n_quantiles
        )
        
    def forward(self,
               merchant_history: torch.Tensor,
               carrier_features: torch.Tensor,
               commodity_features: torch.Tensor,
               time: torch.Tensor,
               initial_velocity: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through Hybrid PINO.
        
        Args:
            merchant_history: [batch, hist_len, 3]
            carrier_features: [batch, carrier_feat_dim]
            commodity_features: [batch, commodity_feat_dim]
            time: [batch]
            initial_velocity: [batch, spatial]
            
        Returns:
            Dictionary with 'velocity_field' and 'eta_quantiles'
        """
        # Encode inputs
        embedding = self.encoder(merchant_history, carrier_features, 
                                commodity_features, time)  # [batch, embedding_dim]
        
        # Expand embedding to spatial dimension
        spatial_dim = initial_velocity.shape[-1]
        embedding_expanded = embedding.unsqueeze(-1).expand(-1, -1, spatial_dim)
        
        # Combine with initial velocity
        fno_input = torch.cat([embedding_expanded, initial_velocity.unsqueeze(1)], dim=1)
        
        # FNO forward
        velocity_field = self.fno(fno_input)  # [batch, 1, spatial]
        velocity_field = velocity_field.squeeze(1)  # [batch, spatial]
        
        # Global pooling for distributional head
        velocity_pooled = torch.mean(velocity_field, dim=-1)  # [batch]
        
        # Predict ETA quantiles
        eta_quantiles = self.dist_head(velocity_pooled.unsqueeze(-1))
        
        return {
            'velocity_field': velocity_field,
            'eta_quantiles': eta_quantiles,
            'velocity_pooled': velocity_pooled
        }


class PhysicsInformedLoss(nn.Module):
    """
    Physics-informed loss function combining data fidelity and PDE constraints.
    """
    
    def __init__(self,
                 lambda_pde: float = 0.1,
                 lambda_jump: float = 1.0,
                 rho: float = 1.0):
        super(PhysicsInformedLoss, self).__init__()
        self.lambda_pde = lambda_pde
        self.lambda_jump = lambda_jump
        self.rho = rho
        
    def pde_residual(self,
                    v: torch.Tensor,
                    mu: torch.Tensor,
                    pressure_grad: torch.Tensor,
                    dt: float,
                    dx: float) -> torch.Tensor:
        """
        Compute PDE residual: ρ(Dv/Dt) + ∇p - μ∇²v - f
        
        Args:
            v: Velocity field [batch, time, spatial]
            mu: Viscosity [batch, time]
            pressure_grad: Pressure gradient [batch, time]
            dt: Time step
            dx: Spatial step
            
        Returns:
            PDE residual
        """
        batch_size, nt, nx = v.shape
        
        # Time derivative: ∂v/∂t
        dv_dt = (v[:, 1:, :] - v[:, :-1, :]) / dt
        
        # Spatial derivatives
        v_interior = v[:, :-1, 1:-1]
        
        # Convective term: v * ∂v/∂x (upwind)
        dv_dx = (v_interior - v[:, :-1, :-2]) / dx
        conv = v_interior * dv_dx
        
        # Laplacian: ∂²v/∂x²
        laplacian = (v[:, :-1, 2:] - 2*v_interior + v[:, :-1, :-2]) / dx**2
        
        # Viscous term
        mu_expanded = mu[:, :-1].unsqueeze(-1)
        viscous = mu_expanded * laplacian
        
        # Pressure gradient
        pg_expanded = pressure_grad[:, :-1].unsqueeze(-1)
        
        # PDE residual
        residual = self.rho * (dv_dt[:, :, 1:-1] + conv) + pg_expanded - viscous
        
        return torch.mean(residual**2)
    
    def jump_loss(self,
                 v: torch.Tensor,
                 holiday_times: List[int],
                 jump_magnitude: float) -> torch.Tensor:
        """
        Compute jump loss for holiday discontinuities.
        
        Args:
            v: Velocity field [batch, time, spatial]
            holiday_times: List of holiday transition time indices
            jump_magnitude: Expected jump magnitude
            
        Returns:
            Jump loss
        """
        loss = 0.0
        for t_jump in holiday_times:
            if t_jump > 0 and t_jump < v.shape[1] - 1:
                v_before = v[:, t_jump - 1, :]
                v_after = v[:, t_jump, :]
                jump_actual = v_after - v_before
                loss += torch.mean((jump_actual - jump_magnitude)**2)
        return loss / max(len(holiday_times), 1)
    
    def quantile_loss(self,
                     predictions: torch.Tensor,
                     targets: torch.Tensor,
                     quantiles: torch.Tensor) -> torch.Tensor:
        """
        Pinball loss for quantile regression.
        
        Args:
            predictions: [batch, n_quantiles]
            targets: [batch]
            quantiles: [n_quantiles]
            
        Returns:
            Quantile loss
        """
        errors = targets.unsqueeze(1) - predictions  # [batch, n_quantiles]
        loss = torch.max(
            (quantiles - 1) * errors,
            quantiles * errors
        )
        return torch.mean(loss)
    
    def forward(self,
               predictions: Dict[str, torch.Tensor],
               targets: torch.Tensor,
               v_history: torch.Tensor,
               mu_history: torch.Tensor,
               pg_history: torch.Tensor,
               holiday_times: List[int],
               quantiles: torch.Tensor,
               dt: float,
               dx: float) -> Dict[str, torch.Tensor]:
        """
        Compute total loss.
        
        Returns:
            Dictionary with individual and total loss components
        """
        # Data fidelity (quantile loss)
        loss_data = self.quantile_loss(predictions['eta_quantiles'], targets, quantiles)
        
        # PDE residual
        loss_pde = self.pde_residual(v_history, mu_history, pg_history, dt, dx)
        
        # Jump loss
        loss_jump = self.jump_loss(v_history, holiday_times, jump_magnitude=-0.6)
        
        # Total loss
        loss_total = loss_data + self.lambda_pde * loss_pde + self.lambda_jump * loss_jump
        
        return {
            'total': loss_total,
            'data': loss_data,
            'pde': loss_pde,
            'jump': loss_jump
        }


def create_example_model() -> HybridPINO:
    """
    Create an example Hybrid PINO model.
    """
    model = HybridPINO(
        fno_modes=12,
        fno_width=64,
        fno_layers=4,
        embedding_dim=64,
        n_quantiles=9
    )
    return model


def test_forward_pass():
    """
    Test forward pass with dummy data.
    """
    print("Testing Hybrid PINO forward pass...")
    
    model = create_example_model()
    
    # Dummy inputs
    batch_size = 4
    hist_len = 10
    spatial_dim = 50
    
    merchant_history = torch.randn(batch_size, hist_len, 3)
    carrier_features = torch.randn(batch_size, 5)
    commodity_features = torch.randn(batch_size, 3)
    time = torch.randn(batch_size)
    initial_velocity = torch.randn(batch_size, spatial_dim)
    
    # Forward pass
    outputs = model(merchant_history, carrier_features, commodity_features, 
                   time, initial_velocity)
    
    print(f"Velocity field shape: {outputs['velocity_field'].shape}")
    print(f"ETA quantiles shape: {outputs['eta_quantiles'].shape}")
    print(f"Velocity pooled shape: {outputs['velocity_pooled'].shape}")
    
    # Test distributional head
    var_95 = model.dist_head.compute_var(outputs['eta_quantiles'], alpha=0.95)
    print(f"VaR_0.95 shape: {var_95.shape}")
    
    print("Forward pass test passed!")


if __name__ == "__main__":
    test_forward_pass()

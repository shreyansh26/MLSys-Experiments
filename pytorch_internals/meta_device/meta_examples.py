import torch
import torch.nn as nn
from typing import Dict, Any
import time

def memory_analysis_example():
    """
    Using meta device to analyze memory requirements without allocating actual memory
    """
    # Create a large model - Don't allocate actual memory
    with torch.device('meta'):
        model = nn.Sequential(
            nn.Linear(1024, 8192),
            nn.ReLU(),
            nn.Linear(8192, 1024),
            nn.ReLU(),
            nn.Linear(1024, 8192),
            nn.ReLU(),
            nn.Linear(8192, 1024),
            nn.ReLU(),
            nn.Linear(1024, 8192),
            nn.ReLU(),
            nn.Linear(8192, 1024),
            nn.ReLU(),
            nn.Linear(1024, 8192),
            nn.ReLU(),
            nn.Linear(8192, 1024),
        )
    
    # Create meta tensor input
    x = torch.empty(128, 1024, device='meta')
    
    # Get memory statistics
    total_params = sum(p.numel() for p in model.parameters())
    input_size = x.numel() * x.element_size()
    output = model(x)
    output_size = output.numel() * output.element_size()
    
    print(f"Model parameters: {total_params:,}")
    print(f"Input size: {input_size / 1024 / 1024:.2f} MB")
    print(f"Output size: {output_size / 1024 / 1024:.2f} MB")

def parameter_initialization_example():
    """
    Using meta device for efficient parameter initialization strategies
    """
    def custom_init(module):
        if isinstance(module, nn.Linear):
            real_weight = torch.randn_like(module.weight, device='cpu')
            module.weight = nn.Parameter(real_weight.to('meta'))

    # Create model on meta
    model = nn.Sequential(
        nn.Linear(512, 1024),
        nn.ReLU(),
        nn.Linear(1024, 256)
    ).to('meta')
    
    # Apply custom initialization
    model.apply(custom_init)
    
    return model

class MetaModuleWrapper(nn.Module):
    """
    Wrapper that creates meta device copies for architecture search
    """
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model.to('meta')
        self.current_device = 'meta'
    
    def to_real(self, device='cuda'):
        """Convert meta model to real model on specified device"""
        if self.current_device == 'meta':
            real_model = self.base_model.to_empty(device=device)
            # Initialize weights if needed
            for module in real_model.modules():
                if hasattr(module, 'reset_parameters'):
                    module.reset_parameters()
            self.current_device = device
            return real_model
        return self.base_model

def architecture_search_example():
    """
    Using meta device for efficient architecture search
    """
    architectures = [
        nn.Sequential(nn.Linear(100, 200), nn.ReLU(), nn.Linear(200, 10)),
        nn.Sequential(nn.Linear(100, 500), nn.ReLU(), nn.Linear(500, 10)),
        nn.Sequential(nn.Linear(100, 1000), nn.ReLU(), nn.Linear(1000, 10))
    ]
    
    meta_models = [MetaModuleWrapper(arch) for arch in architectures]
    
    # Quick memory analysis
    for i, model in enumerate(meta_models):
        params = sum(p.numel() for p in model.base_model.parameters())
        print(f"Architecture {i} parameters: {params:,}")
    
    # Convert best model to real device when needed
    best_model = meta_models[0].to_real('cuda')  # or 'cpu'
    return best_model

def shape_inference_example():
    """
    Using meta device for shape inference in complex architectures
    """
    class ComplexModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(64, 10)
        
        def forward(self, x):
            x = self.pool(torch.relu(self.conv1(x)))
            x = self.adaptive_pool(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)
    
    # Create model and move to meta
    model = ComplexModel().to('meta')
    
    # Test with different input shapes
    shapes = [(1, 3, 32, 32), (1, 3, 64, 64), (1, 3, 128, 128)]
    
    for shape in shapes:
        x = torch.empty(*shape, device='meta')
        out = model(x)
        print(f"Input shape: {shape}, Output shape: {tuple(out.shape)}")

if __name__ == "__main__":
    print("\nMemory Analysis Example:")
    memory_analysis_example()
    
    print("\nParameter Initialization Example:")
    model = parameter_initialization_example()
    print(model)
    
    print("\nArchitecture Search Example:")
    best_model = architecture_search_example()
    print(best_model)
    
    print("\nShape Inference Example:")
    shape_inference_example() 
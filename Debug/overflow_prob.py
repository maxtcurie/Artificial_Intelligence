import torch

# Test with float32
x_float32 = torch.tensor([10**20], dtype=torch.float32)
print(f"Float32 tensor: {x_float32}")
print(f"Float32 tensor value: {x_float32.item()}")

# Test with float64
x_float64 = torch.tensor([10**20], dtype=torch.float64)
print(f"Float64 tensor: {x_float64}")
print(f"Float64 tensor value: {x_float64.item()}")

# Test arithmetic to check for overflow
y_float32 = x_float32 ** 2 
y_float64 = x_float64 ** 2

print(f"Float32 tensor after multiplication: {y_float32}")
print(f"Float64 tensor after multiplication: {y_float64}")

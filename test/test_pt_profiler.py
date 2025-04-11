import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity


# Define a simple module with a convolution and an activation.
class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        # Mark the module's forward call with a record_function to help with identification.
        with record_function("Conv_forward"):
            x = self.conv(x)
        # Mark the module's forward call with a record_function to help with identification.
        with record_function("ReLU_forward"):
            y = self.relu(x)
        return y


# Create an instance of the module and prepare a dummy input on GPU.
model = MyModule().cuda()
input_tensor = torch.randn(32, 3, 224, 224, device="cuda")

# Define a simple MSE loss
criterion = nn.MSELoss()

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True
) as prof:
    for i in range(2):  # run multiple iterations if needed
        # Forward pass
        output = model(input_tensor)

        # Make sure we have a scalar loss to call backward on
        loss = criterion(output, torch.zeros_like(output))

        # Backward pass
        with record_function("MyModule_backward"):
            loss.backward()

# Print the summary table sorted by CUDA time
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

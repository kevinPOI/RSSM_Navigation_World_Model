import torch
import torch.nn as nn
import torch.optim as optim

class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        # Define the downsized dimensions
        downsized_dim = 1024
        
        # Linear layer to downsize the flattened inputs
        input_size_a = 5 * 3 * 128 * 128
        input_size_b = 5 * 5
        self.downsize_a = nn.Linear(input_size_a, downsized_dim)
        self.downsize_b = nn.Linear(input_size_b, downsized_dim)
        
        # Linear layer to map the combined downsized inputs to the output size
        combined_input_size = 2 * downsized_dim
        output_size = 128
        self.fc = nn.Sequential(
            nn.Linear(combined_input_size, combined_input_size),
            nn.ReLU(),
            nn.Linear(combined_input_size, output_size),
            nn.ReLU()
        )

    def forward(self, a, b):
        # Flatten the inputs
        a_flat = a.view(a.size(0), -1)  # Shape: [batch_size, 5 * 3 * 128 * 128]
        b_flat = b.view(b.size(0), -1)  # Shape: [batch_size, 5 * 5]
        
        # Downsize the flattened inputs
        a_down = self.downsize_a(a_flat)  # Shape: [batch_size, downsized_dim]
        b_down = self.downsize_b(b_flat)  # Shape: [batch_size, downsized_dim]
        
        # Concatenate the downsized inputs
        combined = torch.cat((a_down, b_down), dim=1)  # Shape: [batch_size, 2 * downsized_dim]
        
        # Map the combined input to the output size
        out_flat = self.fc(combined)  # Shape: [batch_size, output_size]
        
        # Reshape to match the desired output shape [1, 5, 3, 128, 128]
        return out_flat
    def get_loss(self, a, b):
        out = self.forward(a,b)
        return out.mean()

# Instantiate the model
# model = DummyModel()

# # Example inputs
# a = torch.rand(1, 5, 3, 128, 128)  # Shape [1, 5, 3, 128, 128]
# b = torch.rand(1, 5, 5)            # Shape [1, 5, 5]

# # Forward pass
# output = model(a, b)

# # Define the loss function
# loss_fn = lambda x: x.mean()  # Average of the output tensor

# # Calculate the loss
# loss = loss_fn(output)

# # Print output and loss
# print(f"Output shape: {output.shape}")
# print(f"Loss: {loss.item()}")

# # Example optimizer setup for training
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# optimizer.zero_grad()
# loss.backward()
# optimizer.step()

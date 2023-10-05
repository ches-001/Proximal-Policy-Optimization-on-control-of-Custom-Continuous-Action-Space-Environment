import torch
import torch.nn as nn


class ActorNet(nn.Module):
    def __init__(
            self, 
            input_size:int, 
            output_size:int, 
            hidden_size:int=128, 
            log_std_min: float=-20.0,
            log_std_max: float=1.0):
        
        super(ActorNet, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size//2)
        self.fc3 = nn.Linear(self.hidden_size//2, self.hidden_size//4)
        self.mu_layer = nn.Linear(self.hidden_size//4, self.output_size)
        self.log_std_layer = nn.Linear(self.hidden_size//4, self.output_size)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.fc1(x)
        output = self.fc2(self.relu(output))
        output = self.fc3(self.relu(output))

        mu = self.mu_layer(self.relu(output)).tanh()
        log_std = self.log_std_layer(self.relu(output)).tanh()
        covar = torch.exp(
            (self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1))
        )**2
        covar = (
            covar.unsqueeze(2).expand(*covar.size(), covar.size(1)) * 
            torch.eye(covar.size(1), device=covar.device)
        )
        return mu, covar
    

class CriticNet(nn.Module):
    def __init__(self, input_size:int, hidden_size:int=128):
        super(CriticNet, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size//2)
        self.fc3 = nn.Linear(self.hidden_size//2, self.hidden_size//4)
        self.fc4 = nn.Linear(self.hidden_size//4, 1)
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.fc1(x)
        output = self.fc2(self.relu(output))
        output = self.fc3(self.relu(output))
        output = self.fc4(self.leaky_relu(output))
        return output
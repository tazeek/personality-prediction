import torch
import torch.nn as nn

# Create CNN model
class CNN(nn.Module):
    def __init__(self, output_size):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 3))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3))
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 1))

        self.fc1 = nn.Linear(21392, 128)
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x):

        # Conv2D input: [batch_size, channels, height, width]
        x = x.unsqueeze(0)

        x1 = torch.relu(self.conv1(x))
        x2 = torch.relu(self.conv2(x1))
        x3 = torch.relu(self.conv3(x2))

        #x2 = torch.relu(self.conv2(x))
        #x3 = torch.relu(self.conv3(x))

        # Concatenate outputs
        #x = torch.cat((x1, x2, x3), dim=1) 

        # Flatten the tensor 
        x3 = x3.view(x3.size(0), -1) 
        x3 = torch.relu(self.fc1(x3))
        logits = self.fc2(x3)

        return torch.sigmoid(logits)
    
# For plain model
class LLMClassifer(nn.Module):

    def __init__(self, dropout = 0.5):
        super(LLMClassifer, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(768, 5)

    def forward(self, x):

        logits = self.fc(x)
        return torch.sigmoid(logits)

# Create LSTM model
class LSTMNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMNetwork, self).__init__()

        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def features_extraction(self, x):

        # Initialize hidden state with zeros
        #h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        #c0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)

        # Forward pass through LSTM
        out, _ = self.lstm(x)

        # Take output from the last time step
        #out = out[:, -1, :]

        return out

    def forward(self, x):

        # Fully connected layer
        logits = self.fc(self.features_extraction(x))

        return torch.sigmoid(logits)

# Create GRU model
class GRUNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRUNetwork, self).__init__()

        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, num_layers=2, batch_first=True)

        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, output_size)

        self.relu = nn.LeakyReLU()

    def features_extraction(self, x):
        # Initialize hidden state with zeros
        # 2 for 2 layers
        #h0 = torch.zeros(2, x.size(0), self.hidden_size).to(x.device)  

        # Forward pass through GRU
        out, _ = self.gru(x)

        # Take output from the last time step
        #out = out[:, -1, :]

        # Fully connected layers
        out = self.fc1(out)

        return out

    def forward(self, x):

        # Fully connected layers
        out = self.features_extraction(x)
        out = self.relu(out)
        logits = self.fc2(out)

        return torch.sigmoid(logits)


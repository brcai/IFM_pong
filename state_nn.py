import torch
from torch import nn
import torch.nn.init as init
import random
import torch.nn.functional as F

seed = 1234

torch.manual_seed(seed)
random.seed(seed) 

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'


class ParametricTanh(nn.Module):
    def __init__(self, init_scale=1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(init_scale))

    def forward(self, x):
        return torch.tanh(self.alpha * x)


class Neuron(nn.Module):
    def __init__(self, idx, incoming_indices):
        super().__init__()
        self.idx = idx
        self.incoming_indices = incoming_indices
        self.act = nn.Tanh()

        if len(incoming_indices) > 0:
            fan_in = len(incoming_indices)
            fan_out = 1
            limit = (6 / (fan_in + fan_out)) ** 0.5
            self.weights = nn.Parameter(torch.empty(fan_in).uniform_(-limit, limit))
            self.param = nn.Parameter(torch.empty(len(self.incoming_indices), len(self.incoming_indices)))
            init.xavier_uniform_(self.param)
        else:
            self.weights = nn.Parameter(torch.empty(0))

        self.bias = nn.Parameter(torch.zeros(1))

        self.state_decay = ParametricTanh(0.5)

    def forward(self, model_inputs, neuron_inputs, prev_state):
        """
        inputs: shape (batch_size, num_neurons)
        prev_state: shape (batch_size,)
        """
        if len(self.incoming_indices) == 0:
            total_input = model_inputs[:, self.idx] + self.bias
        else:
            input_vec = torch.stack([neuron_inputs[:, i] for i in self.incoming_indices], dim=1)
            total_input = torch.matmul(input_vec, self.weights) + self.bias

            total_input += self.state_decay(prev_state)

        new_state = F.leaky_relu(total_input)
        output = self.act(total_input)
        return new_state, output


class ExplicitNeuralNetwork(nn.Module):
    def __init__(self, num_neurons=50, num_inputs=2, num_outputs=2, connection_prob=0.8, device='cpu'):
        super().__init__()
        self.num_neurons = num_neurons
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.input_indices = list(range(num_inputs))
        self.output_indices = list(range(num_inputs, num_inputs + num_outputs))

        self.device = device

        self.connections = self._build_connections(num_inputs, num_outputs, connection_prob)

        self.neurons = nn.ModuleList([
            Neuron(i, self.connections[i]) for i in range(num_neurons)
        ])


    def _build_connections(self, num_inputs, num_outputs, prob):
        connections = {}

        for i in range(num_inputs):
            connections[i] = []

        for i in range(num_inputs, num_inputs + num_outputs):
            connections[i] = [
                j for j in range(num_inputs + num_outputs, self.num_neurons)
                if random.random() < prob
            ]

        for i in range(num_inputs + num_outputs, self.num_neurons):
            connections[i] = [
                j for j in range(self.num_neurons)
                if i != j and j not in self.output_indices and random.random() < prob
            ]
        return connections

    def forward(self, input_seq, states=None, neuron_inputs=None):
        batch_size, seq_len, _ = input_seq.shape
        device = input_seq.device

        if states is None:
            states = torch.zeros(batch_size, self.num_neurons).to(device)

        if neuron_inputs is None:
            neuron_inputs = torch.zeros(batch_size, self.num_neurons).to(device)
        
        model_inputs = torch.zeros(batch_size, self.num_inputs).to(device)

        for t in range(seq_len):

            for idx in self.input_indices:
                model_inputs[:, idx] = input_seq[:, t, idx]

            new_states = torch.zeros(batch_size, self.num_neurons).to(device)
            new_outputs = torch.zeros(batch_size, self.num_neurons).to(device)
            for idx, neuron in enumerate(self.neurons):
                if idx in self.output_indices and t==seq_len-1:
                    a = 10
                s, o = neuron.forward(model_inputs, neuron_inputs, states[:, idx])
                new_states[:, idx] = s
                new_outputs[:, idx] = o

            states = new_states
            neuron_inputs = new_outputs

        outputs = neuron_inputs[:, self.output_indices]
        return outputs, states, neuron_inputs


def train_explicit_model(model, X, Y, epochs=100, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        pred, _, _ = model(X.to(device))
        loss = loss_fn(pred, Y.to(device))
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        print(f"Epoch {epoch:03d} | Loss: {loss.item():.6f}                 ")


if __name__ == "__main__":
    model = ExplicitNeuralNetwork(num_neurons=50, num_inputs=1, num_outputs=1, device=device).to(device)

    X = torch.randn(10, 20, 1)  # (batch, seq_len, input_size)
    Y = torch.tanh(torch.rand(10, 1))     # (batch, output_size)

    train_explicit_model(model, X, Y, epochs=300)

    test_X = torch.randn(5, 20, 1)  # (batch=5, seq_len=20, input_size=1)

    with torch.no_grad():
        pred, _, _ = model(test_X.to(device))
        print(Y)
        print(pred)


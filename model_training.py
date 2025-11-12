import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import random

from state_nn import ExplicitNeuralNetwork

seed = 0

torch.manual_seed(seed)
random.seed(seed) 

l = 20
batch_size = 32
device = 'cpu'
num_files = 1
n_neurons = 300
p_connect = 0.2
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x_all = []
y_all = []

x = [[] for i in range(num_files)]
y = [[] for i in range(num_files)]

for i in range(num_files):

    fp = open("dt/c"+str(i)+".txt")

    idx = 0
    for line in fp.readlines():
        x_txt, y_txt = line.strip().split('\t')
        x[i].append([int(itm) for itm in x_txt.split(',')])
        y[i].append([int(itm) for itm in y_txt.split(',')])
    
        idx += 1
        #if idx > 3000:
        #    break

    fp.close()

x_new = [[] for i in range(num_files)]
y_new = [[] for i in range(num_files)]

for i in range(num_files):

    for j in range(0, len(x[i]) - l, l):
        x_new[i].append(x[i][j:j+l])
        y_new[i].append(y[i][j+l])

cnt = int(len(x_new[0])/batch_size)

for i in range(cnt):

    tmp_x = []
    tmp_y = []
    
    for j in range(num_files):
        tmp_x.extend([x_new[j][i+k*cnt] for k in range(batch_size)])
        tmp_y.extend([y_new[j][i+k*cnt] for k in range(batch_size)])

    per_x = torch.tensor(tmp_x, dtype=torch.float32)
    per_y = torch.tensor(tmp_y, dtype=torch.float32).squeeze()
    x_all.append(per_x)
    y_all.append(per_y)

model = ExplicitNeuralNetwork(num_neurons=n_neurons, num_inputs=24, num_outputs=3, connection_prob=p_connect, device=device).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)


num_epochs = 5000
for epoch in range(num_epochs):
    running_loss = 0.0
    states = None
    neuron_outputs = None
    count = 0
    for batch_x, batch_y in zip(x_all, y_all):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        # Forward pass
        outputs, states, neuron_outputs = model(batch_x, states, neuron_outputs)
        #print(outputs)
        loss = criterion(outputs, batch_y)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        #print("batch: ", loss.item())
        running_loss += loss.item()
        
        states = states.detach()
        neuron_outputs = neuron_outputs.detach()
        #print(count)
        count += 1

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss:.4f}")
    if running_loss < 0.0264:
        break


torch.save(model.state_dict(), "save/final_model_"+str(n_neurons)+"_"+str(p_connect)+"_"+str(batch_size)+"_"+str(l)+"_"+str(num_files)+".pth")
print("Training finished, nyaa~! 🐾")

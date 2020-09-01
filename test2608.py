import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as fun
import torch.optim as optim

#raw input
train = pd.read_csv('')
test = pd.read_csv('')

trainset = torch.utils.data.DataLoader(train, batch_size = 5,shuffle = True)
testset = torch.utils.data.DataLoader(test, batch_size = 5,shuffle = True)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(20, 64)
        self.fc2 = nn.Linear(64, 254)
        self.fc3 = nn.Linear(254, 254)
        self.fc4 = nn.Linear(254, 4)

    def forward(self, x):
        x = fun.logsigmoid(self.fc1(x))
        x = fun.logsigmoid(self.fc2(x))
        x = fun.logsigmoid(self.fc3(x))
        x = self.fc4(x)
        return fun.softmax(x, dim = 1)
        return x

net = Net()

optimizer = optim.Adam(net.parameters(),lr=0.01)
Epoch = 5

for epoch in range(Epoch):
    for data in trainset:
        X, y = data
        net.zero_grad()
        output = net(X.view(-1,20))
        loss = fun.nll_loss(output, y)
        loss.backward()
        optimizer.step()
    print(loss)

with torch.no_grad():
    for data in trainset:
        X, y = data
        output = net(X.view(-1, 20))
        for idx, i in enumerate(output):
            if torch.argmax(i) == y[idx]
                correct += 1
            total += 1
print("Accuracy:", round(correct/total, 3))
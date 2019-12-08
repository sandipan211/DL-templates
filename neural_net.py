
# coding: utf-8

# In[1]:


import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


# In[2]:


params = {
    'batch_size': 200,
    'learning_rate': 0.01,
    'epochs': 10,
    'log_interval': 10,
    'momentum': 0.9
}


# In[3]:


def load_data_and_transform():

    train_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=True, download=True,                                                               transform=transforms.Compose([                                                                  transforms.ToTensor(),                                                               transforms.Normalize((0.1307,), (0.3081,))                                                                                            ])),                                                               batch_size=params['batch_size'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(datasets.MNIST('../data', train=False,                                                              transform=transforms.Compose([                                                              transforms.ToTensor(),                                                              transforms.Normalize((0.1307,), (0.3081,))                                                                                         ])),                                                              batch_size=params['batch_size'], shuffle=True)
    
    return train_loader, test_loader


# In[4]:


class Net(nn.Module):

        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(28*28, 512)
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256,10)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)

            return F.log_softmax(x)


# In[5]:


neural_net = Net()
print(neural_net)

optimizer = optim.SGD(neural_net.parameters(), lr=params['learning_rate'], momentum=params['momentum'])
out_loss = nn.NLLLoss()
    


# In[6]:


train_data, test_data = load_data_and_transform()


# In[10]:


# seeing the contents of train_data loaded by loader
l = list(enumerate(train_data))
print(l[0])


# In[16]:


# training loop

for epoch in range(0,params['epochs']):
    total_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_data):
        data, target = Variable(data), Variable(target)
        data = data.view(-1, 28*28)    # reshaping data to (batch_size, 784)
        optimizer.zero_grad()          # zero out all the gradients explicitly at first
        net_out = neural_net(data)
        loss_out = out_loss(net_out, target)
        loss_out.backward()
        optimizer.step()
        
        total_loss += loss_out.item()
        if batch_idx % params['log_interval'] == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format                   (epoch, batch_idx * len(data), len(train_data.dataset),                    100. * batch_idx / len(train_data), total_loss/params['log_interval']))
            total_loss = 0.0


# In[19]:


# run a test loop
test_loss = 0
correct = 0
for data, target in test_data:
    data, target = Variable(data, volatile=True), Variable(target)
    # Basically, set the input to a network to volatile if you are doing inference only and won't 
    # be running backpropagation in order to conserve memory.
    data = data.view(-1, 28 * 28)
    net_out = neural_net(data)
    # sum up batch loss
    test_loss += out_loss(net_out, target).item()
    pred = net_out.data.max(1)[1]  # get the index of the max log-probability
    correct += pred.eq(target.data).sum()

test_loss /= len(test_data.dataset)
print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_data.dataset),
    100. * correct / len(test_data.dataset)))


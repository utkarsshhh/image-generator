import torch
from torchvision import datasets,transforms

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
trainset = datasets.MNIST('MNIST_data/',download=True,train=True,transform= transform)
trainloader = torch.utils.data.DataLoader(trainset,batch_size = 64,shuffle = True)

def sigmoid_function(x):
    '''

    This function transforms any input parameter to the range of (0,1) according
    to the sigmoid function f(x) = 1/(1+exp(-x))

    Input:
    x - a tensor of random numerical values

    Output:
    returns the tensor with the values transformed in between the range (0,1)
    '''

    return (1/(1+torch.exp(-x)))

torch.manual_seed(7)

features = torch.randn((1,5))
weights = torch.randn_like(features)
bias = torch.randn((1,1))

neural_output = torch.add(torch.mm(features,torch.transpose(weights,0,1)),bias)

y = sigmoid_function(neural_output)

print (y)


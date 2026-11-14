import torch.nn as nn

class MyModel (nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(63, 10)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(10, 10)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(10, 5)
       


    def forward (self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
   

        return x

        

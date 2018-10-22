import torch
import torch.nn as nn
import torch.nn.functional as F

## SCORES:
# units: [64,32,16],    useDropout:False            || 600 Episodes     ||--->  14.20
# units: [128,64,32],   useDropout:False            || 600 Episodes     ||--->  14.23 
# units: [256,128,64],  useDropout:False            || 600 Episodes     ||--->  13.90
# units: [128,64,32],   useDropout:True / p=0.5     || 900 Episodes     ||--->  8.00
# units: [256,128,64],  useDropout:True / p=0.5     || 700 Episodes     ||--->  9.16  
# units: [128,64,32],   useDropout:True / p=0.15    || 600 Episodes     ||--->  14.40
# units: [128,128,64],  useDropout:True / p=0.15    || 619 Episodes     ||--->  15.01 (best) 
# units: [256,128,64],  useDropout:True / p=0.15    || 1300 Episodes    ||--->  15.28   
# units: [256,128,64,32],  useDropout:True / p=0.15    || 1300 Episodes    ||--->  15.28   
# units: [256,128,128,32],  useDropout:True / p=0.15    || 1300 Episodes    ||--->  15.28   

class QNetwork(nn.Module):

    def __init__(self, state_size, action_size, seed, useDropout=True, dropOutProb=0.15):
        super(QNetwork, self).__init__()
        self.useDropout = useDropout
        self.dropOutProb = dropOutProb
        self.units = [256,128,128,32]
        self.seed = torch.manual_seed(seed)
        
        print("useDropout:", self.useDropout)
        print("units:", ''.join(str(unit)+", " for unit in self.units))

        self.fc1 = nn.Linear(state_size, self.units[0])
        self.fc2 = nn.Linear(self.units[0], self.units[1])
        self.fc3 = nn.Linear(self.units[1], self.units[2])
        self.fc4 = nn.Linear(self.units[2], self.units[3])
        self.fc5 = nn.Linear(self.units[3], action_size)

    def forward(self, state):
        if self.useDropout:
            x = F.relu(self.fc1(state))
            x = F.dropout(x, p=self.dropOutProb, training=self.training)
            x = F.relu(self.fc2(x))
            x = F.dropout(x, p=self.dropOutProb, training=self.training)
            x = F.relu(self.fc3(x))
            x = F.dropout(x, p=self.dropOutProb, training=self.training)
            x = F.relu(self.fc4(x))
            return self.fc5(x)
        else:
            x = F.relu(self.fc1(state))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = F.relu(self.fc4(x))
            return self.fc5(x)
import random
import numpy as np
import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, residual=False, variant=False):
        super(GraphConvolution, self).__init__() 
        self.variant = variant
        if self.variant:
            self.in_features = 2 * in_features 
        else:
            self.in_features = in_features

        self.out_features = out_features
        self.residual = residual
        self.weight = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def whale_optimizer(self, positions, max_iter=10):
        dim = positions.shape[1]
        SearchAgents_no = positions.shape[0]
        lower_value = np.amin(positions)
        upper_value = np.amax(positions)
        
        leader_pos = positions[random.randint(0, SearchAgents_no - 1)]
        iteration = 0
        
        while iteration < max_iter:
            a = 2 - iteration * (2 / max_iter)
            a2 = -1 + iteration * (-1 / max_iter)
            
            for i in range(SearchAgents_no):
                r1 = random.random()
                r2 = random.random()
                
                A = 2 * a * r1 - a
                C = 2 * r2
                
                b = 1
                l = (a2 - 1) * random.random() + 1
                
                p = random.random()
                
                for j in range(dim):
                    if p < 0.5:
                        if abs(A) >= 1:
                            rand_leader_index = random.randint(0, SearchAgents_no - 1)
                            x_rand = positions[rand_leader_index, :]
                            d_x_rand = abs(C * x_rand[j] - positions[i, j]) 
                            positions[i, j] = abs(x_rand[j] - A * d_x_rand)
                        else:
                            D_Leader = abs(C * leader_pos[j] - positions[i, j]) 
                            positions[i, j] = abs(leader_pos[j] - A * D_Leader)      
                    else:
                        distance2Leader = abs(leader_pos[j] - positions[i, j])
                        positions[i, j] = abs(distance2Leader * math.exp(b * l) * math.cos(l * 2 * math.pi) + leader_pos[j])
                
            positions = np.clip(positions, lower_value, upper_value)
            leader_pos = positions[random.randint(0, SearchAgents_no - 1)]
            iteration += 1
        
        return positions

    def forward(self, input, adj, h0, lamda, alpha, l):
        theta = math.log(lamda / l + 1)
        hi = torch.spmm(adj, input)
        if self.variant:
            support = torch.cat([hi, h0], 1)
            r = (1 - alpha) * hi + alpha * h0
        else:
            support = (1 - alpha) * hi + alpha * h0
            r = support
        
        # Convert tensor to numpy for WOA
        support_np = support.detach().numpy()
        optimized_support = self.whale_optimizer(support_np)
        
        # Convert back to tensor
        optimized_support = torch.tensor(optimized_support, dtype=torch.float32)
        
        output = theta * torch.mm(optimized_support, self.weight) + (1 - theta) * r
        if self.residual:
            output = output + input
        return output

# Example usage
in_features = 10
out_features = 5
gcn = GraphConvolution(in_features, out_features)
input = torch.randn((5, in_features))
adj = torch.eye(5)  # Example adjacency matrix
h0 = torch.randn((5, in_features))
lamda = 0.5
alpha = 0.1
l = 1

# Forward pass through GCN
output = gcn(input, adj, h0, lamda, alpha, l)
print(output)

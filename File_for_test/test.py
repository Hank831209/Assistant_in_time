import torch
import torch.nn.functional as F

# _, predicted = torch.max(F.softmax(net_output.data, dim=1), dim=1)[1]
a = torch.tensor([
    [0.1, 0.2, 0.7],
    [0.3, 0.6, 0.1]
])

b = F.softmax(a, dim=1)
print(b)
c = torch.max(F.softmax(a, dim=1), dim=1)[1]
print(c)


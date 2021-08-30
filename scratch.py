import torch
import torch.nn.functional as F

x = torch.randn(3, 5, 10)
y = torch.randn(3, 5, 10)

attn = torch.bmm(x, x.transpose(1, 2))

# each sentence in the batch has a shape
weights = torch.softmax(attn, dim=2)
print(weights.shape)


z = torch.bmm(weights, x)


print(attn)
print(attn.shape)
print(z.shape)
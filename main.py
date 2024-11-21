import torch as torch



class PointsDataset(torch.utils.data.Dataset):


    def __init__(self,path):
        super().__init__()
        with open(path, 'r') as f:
            self.data = f.readlines()
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    

df = PointsDataset("dataset1.txt")
#print(df[0])
        
class LineModule(torch.nn.Module):
    
    def __init__(self):
        super(LineModule, self).__init__()
        self.w = torch.nn.Parameter(torch.randn(1))
        
    def forward(self, x):
        return self.w * x

model = LineModule()
#print(list(model.parameters()))
#print(model(torch.tensor([1.])))


dl = torch.utils.data.DataLoader(
                                    df,
                                    batch_size=8
                                )
for epoch in range(0, 1000):
    for batch in dl:
        output = model(batch.input)
        error = torch.nn.functional.mse_loss(output, batch.target)
        optimizer = torch.optim.SDG(model.parameters(), lr=0.001)
        optimizer.zero_grad()
        error.backward()
        optimizer.step()



















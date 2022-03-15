import torch

# define a floating point model
class M(torch.nn.Module):
    def __init__(self):
        super(M, self).__init__()
        self.fc = torch.nn.Linear(4, 4)

    def forward(self, x):
        x = self.fc(x)
        return x

#回傳Tensor的最小值跟最大值
def MinMax(tensor):
    tensor = torch.flatten(tensor)
    tensorlist = tensor.detach().tolist()
    Min = tensorlist[0]
    Max = tensorlist[0]
    for i in range(len(tensorlist)):
        if tensorlist[i] < Min:
            Min = tensorlist[i]
        if tensorlist[i] > Max:
            Max = tensorlist[i]
    return Min,Max

#根據Model找出裡面參數的最小值
def MinMax_Model(model):
    state_dict = model.state_dict()
    rMin, rMax, i = 0.0, 0.0, 0
    for k,v in state_dict.items():
        if i == 0:
            rMin, rMax = MinMax(v)
        tempMin, tempMax = MinMax(v)
        if tempMin < rMin:
            rMin = tempMin 
        if tempMax > rMax:
            rMax = tempMax 
        i += 1
    return rMin, rMax
#將float量化成int
def Quan(Q, S, Z):
    return int(Q/S + Z)

#根據給定的最小值最大值量化tensor裡的值
def fp32toint8(tensor, qMin, qMax, rMin, rMax):
    shape = tensor.shape
    tensor = torch.flatten(tensor)
    S = (rMax - rMin)/(qMax - qMin)
    Z = qMax - (rMax/S)
    tensorlist = tensor.detach().tolist()
    for i in range(len(tensorlist)):
        tensorlist[i] = Quan(tensorlist[i],S,Z) 
    tensor = torch.CharTensor(tensorlist)
    tensor = torch.reshape(tensor,shape)
    return tensor

def QuanModelParameter(model, qMin, qMax):
    state_dict = model.state_dict()
    rMin, rMax = MinMax_Model(model)
    for k,v in state_dict.items():
        state_dict[k] = fp32toint8(v, qMin, qMax, rMin, rMax)
    for p in model.parameters():
        p.data = p.data.to(torch.int8)
    model.load_state_dict(state_dict)
    return model
    
def main():
    qMin,qMax = -128,127
    model = M()
    for parameter in model.parameters():
        print(parameter)
    model = QuanModelParameter(model, qMin, qMax)
    for parameter in model.parameters():
        print(parameter)

if __name__ == "__main__":
    main()
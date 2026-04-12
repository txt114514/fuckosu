import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

print("==== PyTorch MNIST Debug Example ====")

# 设备检测
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

if DEVICE == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

# 超参数
BATCH_SIZE = 64
EPOCHS = 2
LR = 0.001

print("Batch size:", BATCH_SIZE)
print("Epochs:", EPOCHS)

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

print("\nLoading dataset...")

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    transform=transform
)

print("Train dataset size:", len(train_dataset))
print("Test dataset size:", len(test_dataset))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

print("DataLoader ready")

# 模型
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1,32,3,1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3,1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Linear(64*5*5,128),
            nn.ReLU(),
            nn.Linear(128,10)
        )

    def forward(self,x):

        x=self.conv(x)
        x=torch.flatten(x,1)
        x=self.fc(x)

        return x

model=Net().to(DEVICE)

print("\nModel structure:")
print(model)

criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=LR)

print("\n==== Start Training ====\n")

for epoch in range(EPOCHS):

    print("Epoch:",epoch+1)

    model.train()

    for batch_idx,(data,target) in enumerate(train_loader):

        data=data.to(DEVICE)
        target=target.to(DEVICE)

        if batch_idx==0:
            print("Input batch shape:",data.shape)
            print("Target shape:",target.shape)

        output=model(data)

        loss=criterion(output,target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx%100==0:
            print(
                f"Batch {batch_idx} | Loss {loss.item():.4f}"
            )

print("\n==== Training Finished ====")

print("\n==== Testing ====")

model.eval()

correct=0

with torch.no_grad():

    for data,target in test_loader:

        data=data.to(DEVICE)
        target=target.to(DEVICE)

        output=model(data)

        pred=output.argmax(dim=1)

        correct+=(pred==target).sum().item()

accuracy=correct/len(test_dataset)

print("Test Accuracy:",accuracy)

# 保存模型
torch.save(model.state_dict(),"mnist_model.pt")

print("\nModel saved -> mnist_model.pt")

print("\n==== Inference Example ====")

sample,label=test_dataset[0]

with torch.no_grad():

    pred=model(sample.unsqueeze(0).to(DEVICE))

pred_digit=pred.argmax().item()

print("True label:",label)
print("Predicted :",pred_digit)

print("\n==== Program Finished ====")
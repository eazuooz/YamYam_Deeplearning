import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize, RandomCrop, RandomHorizontalFlip


# 1) 데이터 전처리 정의
#    mean/std는 CIFAR10 전체 학습 데이터를 기준으로 미리 계산된 값
transforms_train = Compose([
    RandomCrop((32, 32), padding=4),       # 랜덤으로 이미지 일부 제거 후 패딩
    RandomHorizontalFlip(p=0.5),           # y축 기준으로 대칭 (데이터 증강)
    ToTensor(),
    Normalize(mean=(0.4914, 0.4822, 0.4465),
              std=(0.247,  0.243,  0.261)),
])

transforms_test = Compose([
    ToTensor(),
    Normalize(mean=(0.4914, 0.4822, 0.4465),
              std=(0.247,  0.243,  0.261)),
])


# 2) 데이터 로드 (학습용 50,000장 / 평가용 10,000장)
training_data = CIFAR10(root="./", train=True,  download=True, transform=transforms_train)
test_data     = CIFAR10(root="./", train=False, download=True, transform=transforms_test)

print(f"학습 데이터: {len(training_data)}개, 평가 데이터: {len(test_data)}개")
print(f"클래스: {training_data.classes}")


# 3) 데이터로더 생성
train_loader = DataLoader(training_data, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_data,     batch_size=32, shuffle=False)


# 4) 모델 정의
device = "cuda" if torch.cuda.is_available() else "cpu"

class BasicBlock(nn.Module):
    """합성곱 2개 + MaxPooling으로 구성된 기본 블록"""
    def __init__(self, in_channels, out_channels, hidden_dim):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_dim,  kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim,  out_channels, kernel_size=3, padding=1)
        self.relu  = nn.ReLU()
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        return x

class CNN(nn.Module):
    """BasicBlock 3개 + 완전연결층으로 구성된 CNN 분류기"""
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        # 합성곱 블록: 32×32 → 16×16 → 8×8 → 4×4
        self.block1 = BasicBlock(in_channels=3,   out_channels=32,  hidden_dim=16)
        self.block2 = BasicBlock(in_channels=32,  out_channels=128, hidden_dim=64)
        self.block3 = BasicBlock(in_channels=128, out_channels=256, hidden_dim=128)
        # 완전연결층: 256×4×4 = 4096
        self.fc1  = nn.Linear(4096, 2048)
        self.fc2  = nn.Linear(2048, 256)
        self.fc3  = nn.Linear(256,  num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)              # 출력 모양: (-1, 256, 4, 4)
        x = torch.flatten(x, start_dim=1)  # 2차원 특징맵을 1차원으로
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = CNN(num_classes=10)
model.to(device)


# 5) 학습
lr    = 1e-3
optim = Adam(model.parameters(), lr=lr)

for epoch in range(100):
    model.train()
    for data, label in train_loader:
        optim.zero_grad()
        preds = model(data.to(device))
        loss  = nn.CrossEntropyLoss()(preds, label.to(device))
        loss.backward()
        optim.step()

    if epoch == 0 or (epoch + 1) % 10 == 0:
        print(f"epoch {epoch+1:3d} | loss: {loss.item():.4f}")


# 6) 모델 저장 및 불러오기
torch.save(model.state_dict(), "CIFAR10.pth")
model.load_state_dict(torch.load("CIFAR10.pth", map_location=device, weights_only=True))


# 7) 평가
model.eval()
num_corr = 0

with torch.no_grad():
    for data, label in test_loader:
        output   = model(data.to(device))
        preds    = output.data.max(1)[1]              # 가장 높은 점수의 클래스를 예측값으로 선택
        num_corr += preds.eq(label.to(device).data).sum().item()

print(f"Accuracy: {num_corr / len(test_data):.4f}")


# 8) 샘플 이미지 시각화 (학습 데이터 9장 저장)
classes = training_data.classes

for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(training_data.data[i])   # CIFAR10은 RGB이므로 cmap 불필요
    plt.title(classes[training_data.targets[i]], fontsize=8)
    plt.axis("off")

plt.suptitle("CIFAR10 Sample Images")
plt.tight_layout()
plt.savefig("cifar10_samples.png")   # plt.show() 대신 파일로 저장해 실행 흐름이 멈추지 않도록 함
print("샘플 이미지 → cifar10_samples.png 저장 완료")

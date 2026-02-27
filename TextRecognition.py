import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

# 1) 데이터 로드 (학습용 60,000장 / 평가용 10,000장)
training_data = MNIST(root="./", train=True,  download=True, transform=ToTensor())
test_data     = MNIST(root="./", train=False, download=True, transform=ToTensor())

print(f"학습 데이터: {len(training_data)}개, 평가 데이터: {len(test_data)}개")

# 2) 데이터로더 생성
train_loader = DataLoader(training_data, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_data,     batch_size=32, shuffle=False)  # 평가용은 순서 유지

# 3) 모델 정의
device = "cuda" if torch.cuda.is_available() else "cpu"  # GPU가 있으면 GPU 사용

model = nn.Sequential(
    nn.Linear(784, 64),   # 입력: 28×28 = 784 픽셀을 1차원으로 펼친 값
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 10)     # 출력: 숫자 0~9 총 10개 클래스
)
model.to(device)

# 4) 학습
lr    = 1e-3
optim = Adam(model.parameters(), lr=lr)

for epoch in range(20):
    model.train()
    for data, label in train_loader:
        optim.zero_grad()
        data  = torch.reshape(data, (-1, 784)).to(device)  # (B,1,28,28) → (B,784) 형태로 변환
        preds = model(data)
        loss  = nn.CrossEntropyLoss()(preds, label.to(device))
        loss.backward()
        optim.step()
    print(f"epoch {epoch+1:2d} | loss: {loss.item():.4f}")

# 5) 모델 저장 및 불러오기
torch.save(model.state_dict(), "MNIST.pth")
model.load_state_dict(torch.load("MNIST.pth", map_location=device, weights_only=True))

# 6) 평가
model.eval()
num_corr = 0

with torch.no_grad():  # 평가 단계에서는 기울기 계산 불필요
    for data, label in test_loader:
        data   = torch.reshape(data, (-1, 784)).to(device)
        output = model(data)
        preds  = output.data.max(1)[1]                          # 가장 높은 점수의 클래스를 예측값으로 선택
        corr   = preds.eq(label.to(device).data).sum().item()   # 정답과 일치하는 개수 계산
        num_corr += corr

print(f"Accuracy: {num_corr / len(test_data):.4f}")

# 7) 샘플 이미지 시각화 (학습 데이터 9장 저장)
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(training_data.data[i], cmap="gray")
    plt.axis("off")
plt.suptitle("MNIST Sample Images")
plt.tight_layout()
plt.savefig("mnist_samples.png")  # plt.show() 대신 파일로 저장해 실행 흐름이 멈추지 않도록 함
print("샘플 이미지 → mnist_samples.png 저장 완료")

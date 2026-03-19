"""
CIFAR-10 분류 - ResNet (잔차 신경망)
======================================
ResNet의 핵심 아이디어: 입력을 출력에 더하는 "잔차 연결(skip connection)"을 통해
기울기 소실 문제를 해결하고 더 깊은 네트워크를 학습 가능하게 한다.
"""

import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import tqdm
import matplotlib.pyplot as plt

from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import (
    Compose, ToTensor, Normalize,
    RandomCrop, RandomHorizontalFlip
)


# ────────────────────────────────────────────────
# 1) 데이터 전처리 정의
# ────────────────────────────────────────────────
# 학습 데이터: 데이터 증강(augmentation) 적용
transforms_train = Compose([
    RandomCrop((32, 32), padding=4),        # 이미지 주변에 패딩 후 랜덤 크롭 (위치 다양성)
    RandomHorizontalFlip(p=0.5),            # 50% 확률로 좌우 반전 (대칭 다양성)
    ToTensor(),                             # PIL/numpy → torch.Tensor [0,1]
    Normalize(                              # 채널별 평균·표준편차로 정규화
        mean=(0.4914, 0.4822, 0.4465),      # CIFAR-10 전체 데이터 기준 사전 계산값
        std =(0.247,  0.243,  0.261)
    ),
])

# 평가 데이터: 증강 없이 정규화만 적용
transforms_test = Compose([
    ToTensor(),
    Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std =(0.247,  0.243,  0.261)
    ),
])


# ────────────────────────────────────────────────
# 2) 데이터 로드
#    - 학습용: 50,000장 / 평가용: 10,000장
#    - 클래스: airplane, automobile, bird, cat, deer,
#              dog, frog, horse, ship, truck  (총 10종)
# ────────────────────────────────────────────────
training_data = CIFAR10(root="./", train=True,  download=True, transform=transforms_train)
test_data     = CIFAR10(root="./", train=False, download=True, transform=transforms_test)

print(f"학습 데이터: {len(training_data)}개 / 평가 데이터: {len(test_data)}개")
print(f"클래스: {training_data.classes}")


# ────────────────────────────────────────────────
# 3) 데이터로더 생성
# ────────────────────────────────────────────────
train_loader = DataLoader(training_data, batch_size=32, shuffle=True)   # 학습: 셔플 O
test_loader  = DataLoader(test_data,     batch_size=32, shuffle=False)  # 평가: 셔플 X


# ────────────────────────────────────────────────
# 4) ResNet 모델 정의
# ────────────────────────────────────────────────
class ResidualBlock(nn.Module):
    """
    ResNet 기본 블록 (Basic Block)
    ─────────────────────────────
    입력 x를 두 개의 합성곱 층에 통과시킨 후,
    원본 x를 더해서(skip connection) 출력한다.

    채널 수나 해상도가 달라지는 경우 projection(1×1 Conv)으로 차원을 맞춘다.

    구조:
        x ──► Conv3×3 ──► BN ──► ReLU ──► Conv3×3 ──► BN ──►(+)──► ReLU
        └─────────────── (필요시 projection) ───────────────────┘
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        # 메인 경로 (두 개의 3×3 합성곱)
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)

        self.relu  = nn.ReLU()

        # 잔차 경로: 채널 수 또는 해상도가 달라지면 1×1 Conv로 projection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))   # Conv → BN → ReLU
        out = self.bn2(self.conv2(out))             # Conv → BN (ReLU 전)
        out += self.shortcut(x)                     # 잔차 연결 (skip connection)
        out = self.relu(out)                        # 합산 후 ReLU
        return out


class ResNet(nn.Module):
    """
    CIFAR-10용 소형 ResNet
    ──────────────────────
    입력:  3 × 32 × 32 (RGB 이미지)
    출력:  num_classes 차원 로짓(logit)

    구조:
        conv1  : 3 → 64, 3×3  (입력층)
        layer1 : 64 → 64  (stride=1, 해상도 유지)
        layer2 : 64 → 128 (stride=2, 해상도 절반)
        layer3 : 128 → 256 (stride=2, 해상도 절반)
        GAP    : 전역 평균 풀링 → 256차원 벡터
        fc     : 256 → num_classes
    """
    def __init__(self, num_classes=10):
        super(ResNet, self).__init__()

        # 입력 합성곱: 3채널 이미지를 64채널 특징맵으로 변환
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # 잔차 블록 레이어 (채널 수가 늘어나면서 해상도는 줄어듦)
        self.layer1 = ResidualBlock(64,  64,  stride=1)   # 32×32 유지
        self.layer2 = ResidualBlock(64,  128, stride=2)   # 32×32 → 16×16
        self.layer3 = ResidualBlock(128, 256, stride=2)   # 16×16 → 8×8

        # 전역 평균 풀링: 8×8 특징맵 → 1×1 (공간 차원 제거)
        self.gap = nn.AdaptiveAvgPool2d(1)

        # 분류기
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv1(x)    # → (B, 64, 32, 32)
        x = self.layer1(x)   # → (B, 64, 32, 32)
        x = self.layer2(x)   # → (B, 128, 16, 16)
        x = self.layer3(x)   # → (B, 256, 8, 8)
        x = self.gap(x)      # → (B, 256, 1, 1)
        x = torch.flatten(x, start_dim=1)  # → (B, 256)
        x = self.fc(x)       # → (B, num_classes)
        return x


# ────────────────────────────────────────────────
# 5) 모델 초기화
# ────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"사용 디바이스: {device}")

model = ResNet(num_classes=10)
model.to(device)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"학습 가능한 파라미터 수: {total_params:,}")


# ────────────────────────────────────────────────
# 6) 학습
# ────────────────────────────────────────────────
lr    = 1e-4                                  # 학습률 (ResNet은 작은 lr 권장)
optim = Adam(model.parameters(), lr=lr)
loss_fn = nn.CrossEntropyLoss()               # 다중 클래스 분류 손실 함수

NUM_EPOCHS = 30

for epoch in range(NUM_EPOCHS):
    model.train()                             # Dropout·BN을 학습 모드로 전환
    iterator = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{NUM_EPOCHS}")

    for data, label in iterator:
        data, label = data.to(device), label.to(device)

        optim.zero_grad()                     # 이전 스텝의 기울기 초기화
        preds = model(data)                   # 순전파 (forward pass)
        loss  = loss_fn(preds, label)         # 손실 계산
        loss.backward()                       # 역전파 (backpropagation)
        optim.step()                          # 파라미터 업데이트

        iterator.set_postfix(loss=f"{loss.item():.4f}")


# ────────────────────────────────────────────────
# 7) 모델 저장 & 불러오기
# ────────────────────────────────────────────────
SAVE_PATH = "ResNet.pth"
torch.save(model.state_dict(), SAVE_PATH)
print(f"모델 저장 완료 → {SAVE_PATH}")

# 저장된 가중치 재로드 (추론 전 검증 목적)
model.load_state_dict(torch.load(SAVE_PATH, map_location=device, weights_only=True))


# ────────────────────────────────────────────────
# 8) 평가
# ────────────────────────────────────────────────
model.eval()                                  # Dropout·BN을 평가 모드로 전환
num_corr = 0

with torch.no_grad():                         # 기울기 계산 비활성화 (메모리·속도 절약)
    for data, label in test_loader:
        output = model(data.to(device))
        preds  = output.argmax(dim=1)         # 각 샘플에서 가장 높은 로짓의 인덱스 = 예측 클래스
        num_corr += preds.eq(label.to(device)).sum().item()

accuracy = num_corr / len(test_data)
print(f"테스트 정확도: {accuracy:.4f}  ({num_corr}/{len(test_data)})")


# ────────────────────────────────────────────────
# 9) 샘플 이미지 시각화 (3×3 그리드)
# ────────────────────────────────────────────────
classes = training_data.classes

plt.figure(figsize=(6, 6))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(training_data.data[i])         # CIFAR10은 uint8 RGB이므로 cmap 불필요
    plt.title(classes[training_data.targets[i]], fontsize=9)
    plt.axis("off")

plt.suptitle("CIFAR-10 Sample Images", fontsize=12)
plt.tight_layout()
plt.savefig("cifar10_resnet_samples.png", dpi=150)
print("샘플 이미지 → cifar10_resnet_samples.png 저장 완료")

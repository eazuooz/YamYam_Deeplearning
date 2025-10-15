import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1) 데이터 로드
housing = fetch_california_housing()
X = housing.data
y = housing.target.reshape(-1, 1)  # (N,1)로 맞춤

# 2) train/val 분리
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 3) 표준화(입력은 필수, 타깃은 선택)
x_scaler = StandardScaler().fit(X_tr)
X_tr = x_scaler.transform(X_tr)
X_val = x_scaler.transform(X_val)

# (선택) 타깃도 표준화하면 수렴이 더 안정적일 수 있음
# y_scaler = StandardScaler().fit(y_tr)
# y_tr = y_scaler.transform(y_tr); y_val = y_scaler.transform(y_val)

# 4) 텐서/데이터로더
X_tr_t = torch.from_numpy(X_tr).float()
y_tr_t = torch.from_numpy(y_tr).float()
X_val_t = torch.from_numpy(X_val).float()
y_val_t = torch.from_numpy(y_val).float()

train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=128, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=256, shuffle=False)

# 5) 모델
model = nn.Sequential(
    nn.Linear(X_tr.shape[1], 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
)

criterion = nn.MSELoss()
optim = Adam(model.parameters(), lr=1e-3)

# 6) 학습 루프(에포크 ‘평균 손실’ 로깅)
for epoch in range(200):
    model.train()
    tr_loss_sum, n_tr = 0.0, 0
    for xb, yb in train_loader:
        optim.zero_grad()
        preds = model(xb)               # (B,1)
        loss = criterion(preds, yb)     # (B,1) vs (B,1)
        loss.backward()
        optim.step()
        tr_loss_sum += loss.item() * xb.size(0)
        n_tr += xb.size(0)
    tr_loss = tr_loss_sum / n_tr

    # 검증
    model.eval()
    with torch.no_grad():
        val_loss_sum, n_val = 0.0, 0
        for xb, yb in val_loader:
            preds = model(xb)
            vloss = criterion(preds, yb)
            val_loss_sum += vloss.item() * xb.size(0)
            n_val += xb.size(0)
        val_loss = val_loss_sum / n_val

    if epoch % 20 == 0:
        print(f"epoch {epoch:3d} | train MSE: {tr_loss:.4f} | val MSE: {val_loss:.4f}")

# 7) 예측(스케일 되돌리기 선택)
model.eval()
with torch.no_grad():
    one_pred = model(X_val_t[:1]).item()
    one_real = y_val_t[:1].item()
    # 만약 y를 표준화했다면 역변환 필요:
    # one_pred = y_scaler.inverse_transform([[one_pred]])[0][0]
    # one_real = y_scaler.inverse_transform([[one_real]])[0][0]

print(f"예측값: {one_pred:.2f}, 실제값: {one_real:.2f}")

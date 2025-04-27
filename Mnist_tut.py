import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# CUDA(그래픽카드) 사용 가능하면 사용, 아니면 CPU 사용
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST 평균, 표준편차로 정규화
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = datasets.MNIST('./data', train=False, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")


dataiter = iter(train_loader)
images, labels = next(dataiter)

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.imshow(images[i].squeeze().numpy(), cmap='gray')
    plt.title(f'Label: {labels[i].item()}')
    plt.axis('off')
plt.tight_layout()
plt.show()


class MnistCNN(nn.Module):
    def __init__(self):
        super(MnistCNN, self).__init__()
        # 첫 번째 합성곱(Convolution) 레이어
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 두 번째 합성곱 레이어
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 완전연결(FC) 레이어
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        
        # 드롭아웃(과적합 방지)
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 64 * 7 * 7)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


def train(model, device, train_loader, criterion, optimizer, epoch):
    model.train()  # 모델을 학습 모드로 전환
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)  # 데이터를 GPU/CPU로 이동
        optimizer.zero_grad()  # 기울기(gradient) 초기화
        outputs = model(data)  # 모델에 데이터 넣고 결과 얻기
        loss = criterion(outputs, target)  # 정답과 비교해서 loss(오차) 계산
        loss.backward()  # 오차를 바탕으로 역전파(gradient 계산)
        optimizer.step()  # 계산된 gradient로 파라미터(가중치) 업데이트

        running_loss += loss.item()
        _, predicted = outputs.max(1)  # 예측 결과(가장 확률 높은 값) 뽑기
        total += target.size(0)
        correct += predicted.eq(target).sum().item()  # 맞춘 개수 세기

        # 100번마다 중간 결과 출력
        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch} | Batch: {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f} | Acc: {100.*correct/total:.2f}%')
    
    train_loss = running_loss / len(train_loader)
    train_acc = 100. * correct / total
    return train_loss, train_acc


def test(model, device, test_loader, criterion):
    model.eval()  # 모델을 평가(테스트) 모드로 전환
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():  # 테스트할 때는 gradient 계산 안 함(속도↑, 메모리↓)
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            test_loss += criterion(outputs, target).item()
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    test_loss /= len(test_loader)
    test_acc = 100. * correct / total
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{total} ({test_acc:.2f}%)')
    return test_loss, test_acc


num_epochs = 5
train_losses = []
test_losses = []
train_accs = []
test_accs = []

for epoch in range(1, num_epochs + 1):
    train_loss, train_acc = train(model, device, train_loader, criterion, optimizer, epoch)
    test_loss, test_acc = test(model, device, test_loader, criterion)
    
    # 그래프 그릴 때 쓰려고 저장
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    train_accs.append(train_acc)
    test_accs.append(test_acc)
    
    print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')

print('Finished Training')

def predict_single_image(image_tensor, model):
    """이미지 한 장에 대해 예측"""
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)  # 배치 차원 추가
        output = model(image_tensor)
        prob = F.softmax(output, dim=1)
        pred_prob, pred_label = torch.max(prob, 1)
    return pred_label.item(), pred_prob.item()

# 테스트셋에서 랜덤으로 이미지 한 장 뽑기
idx = np.random.randint(0, len(test_dataset))
sample_image, sample_label = test_dataset[idx]

# 예측
pred_digit, confidence = predict_single_image(sample_image, model)

# 결과 시각화
plt.figure(figsize=(6, 6))
plt.imshow(sample_image.squeeze().numpy(), cmap='gray')
plt.title(f"Prediction: {pred_digit} (Confidence: {confidence:.2f})\nTrue Label: {sample_label}", size=16)
plt.axis('off')
plt.show()
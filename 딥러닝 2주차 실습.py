
# 관련 패키지 import
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
mnist = torchvision.datasets.MNIST(root='./data', train=True, download=True)
print(mnist.data.size())
print(mnist.targets.size())
num = 6000
plt.imshow(mnist.data[num],cmap="Greys",interpolation="nearest")
plt.show()
print(mnist.targets[num])

num = 5000
plt.imshow(mnist.data[num],cmap="Greys",interpolation="nearest")
plt.show()
print(mnist.targets[num])

"""## Hyperparameter"""

# Device configuration, gpu 사용 가능한 경우 device를 gpu로 설정하고 사용 불가능하면 cpu로 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
# Hyper-parameters
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.0001

"""## Dataset and Dataloader"""

# 파이토치에서 제공하는 MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./data',train=True, transform=transforms.ToTensor(),download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

# 배치 단위로 네트워크에 데이터를 넘겨주는 Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

"""## 모델 구조 설계"""

class NeuralNet(nn.Module):
  def __init__(self, input_size, hidden_size, num_classes):
    super(NeuralNet, self).__init__()
    self.fc1 = nn.Linear(input_size, hidden_size)
    self.fc2 = nn.Linear(hidden_size, hidden_size)
    self.fc3 = nn.Linear(hidden_size, num_classes)

  def forward(self, x):
    out = F.relu(self.fc1(x))
    out = F.relu(self.fc2(out))
    out = self.fc3(out)
    return out
    
model = NeuralNet(input_size, hidden_size, num_classes).to(device)
# .to(device) : 모델을 지정한 device로 올려줌

print(model)

"""## Softmax and cross Entropy, Optimizer"""

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# model.parameters -> 가중치 w들을 의미

"""## Training"""

def evaluation(data_loader):
  correct = 0
  total = 0
  for images, labels in data_loader:
    images = images.reshape(-1, 28*28).to(device)
    labels = labels.to(device)
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
  return correct/total

loss_arr = []
total_step = len(train_loader)
max_value = 0
for epoch in range(num_epochs):
  model.train()
  for i, (images, labels) in enumerate(train_loader):
    # 이미지와 정답(label)을 device로 올림
    images = images.reshape(-1, 28*28).to(device) 
    labels = labels.to(device)
    # Feedforward 과정
    outputs = model(images)
    # Loss 계산
    loss = criterion(outputs, labels)
    # Backward and optimize
    optimizer.zero_grad() # iteration 마다 gradient를 0으로 초기화
    loss.backward() # 가중치 w에 대해 loss를 미분
    optimizer.step() # 가중치들을 업데이트
    if (i+1) % 100 == 0:
      loss_arr.append(loss)
    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
    .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
  with torch.no_grad():
    model.eval()
    acc = evaluation(test_loader)
    if max_value < acc :
      max_value = acc
      print("max dev accuracy: ", max_value)
      torch.save(model.state_dict(), 'model.ckpt')

"""## Test"""

# Test 과정 : 학습한 모델의 성능을 확인하는 과정
# 이 과정에서는 gradient를 계산할 필요가 없음!
with torch.no_grad():
  correct = 0
  total = 0
  for images, labels in test_loader:
    images = images.reshape(-1, 28*28).to(device)
    labels = labels.to(device)
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
  print('Accuracy of the network on the 10000 test images: {} %'
  .format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')
plt.plot(loss_arr)
plt.show()


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CocoDetection
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.rpn import AnchorGenerator

# 하이퍼파라미터 설정
batch_size = 4
num_epochs = 10
learning_rate = 0.001

# 데이터 로딩 및 전처리
transform = transforms.Compose([transforms.ToTensor()])

# 데이터셋 경로와 어노테이션 파일 경로 설정 (실제 데이터셋에 맞게 변경 필요)
dataset = CocoDetection(root='path_to_data_folder',
                       annFile='path_to_annotations.json',
                       transform=transform)

data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 모델 정의
backbone = fasterrcnn_resnet50_fpn(pretrained=True)
model = fasterrcnn_resnet50_fpn(pretrained=True)

anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                  aspect_ratios=((0.5, 1.0, 2.0),))

roi_pooler = nn.Sequential(nn.AdaptiveMaxPool2d((7, 7)), nn.Flatten())

model.rpn.anchor_generator = anchor_generator
model.rpn.head = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
                              nn.ReLU(),
                              nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
                              nn.ReLU())

model.roi_heads.box_head = nn.Sequential(nn.Linear(in_features=1024, out_features=1024),
                                        nn.ReLU(),
                                        nn.Linear(in_features=1024, out_features=1024),
                                        nn.ReLU())

model.roi_heads.box_predictor.cls_score = nn.Linear(in_features=1024, out_features=num_classes)
model.roi_heads.box_predictor.bbox_pred = nn.Linear(in_features=1024, out_features=4)

# 손실 함수 및 옵티마이저 설정
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 학습 루프
for epoch in range(num_epochs):
    for images, targets in data_loader:
        optimizer.zero_grad()
        images = [image for image in images]
        targets = [{'boxes': target['boxes'], 'labels': target['labels']} for target in targets]
        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        loss.backward()
        optimizer.step()

# 모델 저장
torch.save(model.state_dict(), 'bird_detection_model.pth')

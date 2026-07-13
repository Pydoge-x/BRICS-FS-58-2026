import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import timm
import torch
import torch.nn as nn
from dataset import get_train_loader, get_test_loader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
torch.backends.cudnn.benchmark = True
# 1. 配置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 7
EPOCHS = 30
MODEL_NAME = "mobilevit_xs"

# 2. 加载预训练模型
# timm.create_model 自动下载预训练权重，num_classes=7自动替换最后一层
model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=NUM_CLASSES)
model = model.to(DEVICE)

# 3. 损失函数和优化器
criterion = nn.CrossEntropyLoss() # 使用交叉熵损失函数，多分类任务的标准损失函数

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 4. 评估函数
def evaluate(model, loader):
    model.eval() # 进行评估
    all_preds, all_labels = [], [] # 创建两个列表用于存放预测的结果和标签
 
    with torch.no_grad(): # 关闭梯度计算，因为在模型评估时不需要进行反向传播
        for images, lables in tqdm(loader, desc="评估中"): # 加进度条，可视化评估进度
            images, lables = images.to(DEVICE), lables.to(DEVICE) # 放到GPU里计算
            with torch.amp.autocast('cuda'):  # 评估也用混合精度加速
                outputs = model(images)  # 前向传播，得到 Logits
            preds = torch.argmax(outputs, dim=1) # 获取预测类别,outputs的第一个维度是类别维度，因此是1
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(lables.cpu().numpy())
        
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')
    return acc,precision,recall,f1

if __name__ == '__main__':
    # 在 __main__ 中创建 DataLoader，避免 Windows 多进程 spawn 问题
    train_loader = get_train_loader()
    test_loader = get_test_loader()
    print(f"模型：{MODEL_NAME}, 设备：{DEVICE}")
    print(f"分类头输出维度：{model.get_classifier().out_features}")
    scaler = torch.amp.GradScaler('cuda')
    best_score = 0.0
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for images, lables in tqdm(train_loader, desc=f"Epoch {epoch + 1} / {EPOCHS}"):
            images, lables = images.to(DEVICE), lables.to(DEVICE)

            # 模型训练
            optimizer.zero_grad() # 梯度清零
            with torch.amp.autocast('cuda'):
                outputs = model(images) # 前向传播
                loss = criterion(outputs, lables) # 计算损失loss
            
            scaler.scale(loss).backward()
            scaler.step(optimizer=optimizer) #  更新参数
            scaler.update() # 动态调整放大倍数

            running_loss += loss.item()

        acc, p, r, f1 = evaluate(model, test_loader)
        print(f"Epoch {epoch + 1}: Loss{running_loss / len(train_loader):.4f}"
              f"Acc={acc:.4f} P={p:.4f} R={r:.4f} F1={f1:.4f}")
        score = (acc + f1) / 2
        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), f"../models/{MODEL_NAME}_best.pth")
            print(f"  → 新最佳模型已保存 (score={best_score:.4f}, Acc={acc:.4f}, F1={f1:.4f})")
    print(f"训练完成，最佳综合得分: {best_score:.4f}")
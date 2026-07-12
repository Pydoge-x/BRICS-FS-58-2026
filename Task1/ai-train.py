import timm
import torch
import torch.nn as nn
from dataset import train_loader, test_loader, train_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

# ============ 配置 ============
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 7
EPOCHS = 10
MODEL_NAME = "resnet50"  # 也可选: mobilenetv3_small_100, mobilevit_xs, vgg16

# ============ 1. 加载预训练模型 ============
# timm.create_model 自动下载预训练权重，num_classes=7 自动替换最后一层
model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=NUM_CLASSES)
model = model.to(DEVICE)

# ============ 2. 损失函数和优化器 ============
# TODO_1: 多分类任务的标准损失函数是什么？
criterion = nn.______()

# TODO_2: 优化器选哪个？学习率设多少？（提示：预训练模型微调常用 1e-3 或 1e-4）
optimizer = torch.optim.______(model.parameters(), lr=______)

print(f"模型: {MODEL_NAME} | 设备: {DEVICE}")
print(f"分类头输出维度: {model.get_classifier().out_features}")

# ============ 4. 评估函数 ============
def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # TODO: 四项指标全部设置 average='macro'
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average=______)
    recall = recall_score(all_labels, all_preds, average=______)
    f1 = f1_score(all_labels, all_preds, average=______)
    return acc, precision, recall, f1

# ============ 5. 训练循环 ============
if __name__ == '__main__':
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            # TODO: 训练五步走
            optimizer.______()                    # 1. 清空梯度
            outputs = ______(images)              # 2. 前向传播
            loss = criterion(outputs, labels)     # 3. 计算损失
            loss.______()                          # 4. 反向传播
            optimizer.______()                    # 5. 更新参数
            
            running_loss += loss.item()
        
        acc, p, r, f1 = evaluate(model, test_loader)
        print(f"Epoch {epoch+1}: Loss={running_loss/len(train_loader):.4f} "
              f"Acc={acc:.4f} P={p:.4f} R={r:.4f} F1={f1:.4f}")
    
    torch.save(model.state_dict(), "fer_resnet50.pth")
    print("模型已保存")
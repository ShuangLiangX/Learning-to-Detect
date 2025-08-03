import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, roc_auc_score
import os
import numpy as np
import random
import json
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
class SmallAutoencoder(nn.Module):
    def __init__(self, input_dim, bottleneck_dim=4):
        super(SmallAutoencoder, self).__init__()
        
        hidden1 = max(input_dim // 2, bottleneck_dim * 2)  # 保证至少比 bottleneck 大
        hidden2 = max(input_dim // 4, bottleneck_dim * 2)  # 再缩小

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, bottleneck_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat
def main():
    # ======================
    # 1. 准备数据
    # ======================
    set_seed(42)
    model = 'vicuna'
    mode = "safe"
    import sys
    sys.path.append(f"/home/u2021201665/code/SCAV-{model}")
    from utils import train_data,test_data
    from classifier import LinearClassifier

    void=False
    device = "cuda:7"
    dtype = torch.float16
    p0=0.9
    np.random.seed(42)
    layer_chosen = []
    test_pos,test_neg = test_data(device=device,void=void)
    #pos 
    test_pos = test_pos.to(dtype)
    #neg
    with open(f"SCAV-{model}/instructions/advbench-safe.json",'r') as f:
        neg = json.load(f)
    tensors = torch.load(f"/home/u2021201665/asset/QA-{model}/advbench_answer.pth")
    indices = [int(i)-246 for i in list(neg.keys())]
    test_neg = tensors[indices].to(device)
    # print(test_pos.shape)
    # exit(0)
    layers = [i+1 for i in range(32)]
    classifiers = {}
    scores = []
    save_dir = f"SCAV-{model}/pth"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"✅ 创建目录: {save_dir}")
    else:
        print(f"📂 目录已存在: {save_dir}")

    for i in layers:
        if void:
            classifier_path = f"SCAV-{model}/pth/advbench-none/{i}.pth"
        else:
            classifier_path = f"SCAV-{model}/pth/advbench/{i}.pth"
            
        classifier = LinearClassifier() # 假设输入维度 4096，输出 1
        classifier.load_state_dict(torch.load(classifier_path, map_location=device))  # 加载权重
        classifier.to(device,dtype=dtype)  # 转移到指定 device
        classifier.eval()
        classifiers[i] = classifier
        input1 = test_pos[:,i,:].to(device,dtype=dtype)
        input2 = test_neg[:,i,:].to(device,dtype=dtype)
        num1 = input1.shape[0]
        num2 = input2.shape[0]

        # 预测结果
        with torch.no_grad():
            output1 = classifier(input1)
            output2 = classifier(input2)
        score = (1-torch.sum(output1)/num1+torch.sum(output2)/num2)/2
        # print(score)
        scores.append(score)
        if score >=p0:
            layer_chosen.append(i)
        else:
            continue
    print(layer_chosen)

    possibilities_safe = []
    possibilities_unsafe = []
    for i in layer_chosen:
        safe_input = test_pos[:, i, :]
        unsafe_input = test_neg[:, i, :]
        with torch.no_grad():
            outputs = classifiers[i](safe_input)
            possibilities_safe.append(outputs)
            
            outputs = classifiers[i](unsafe_input.to(dtype))
            possibilities_unsafe.append(outputs)

     # 你已经准备好的 possibilities_safe 和 possibilities_unsafe
    possibilities_safe = torch.stack(possibilities_safe, dim=1).float().cpu().numpy()
    possibilities_unsafe = torch.stack(possibilities_unsafe, dim=1).float().cpu().numpy()

    from time import perf_counter
    start = perf_counter()

    n_safe_total = possibilities_safe.shape[0]
    n_test_safe = 80
    indices = np.arange(n_safe_total)
    np.random.shuffle(indices)
    test_indices = indices[:n_test_safe]
    train_indices = indices[n_test_safe:]

    X_train_safe = possibilities_safe[train_indices]
    X_test_safe = possibilities_safe[test_indices]

    # ================= 构造训练和测试集 =================
    X_train = torch.tensor(X_train_safe, dtype=torch.float32)

    X_test_all = np.concatenate([X_test_safe, possibilities_unsafe], axis=0)
    y_test_all = np.concatenate([np.zeros(X_test_safe.shape[0]), np.ones(possibilities_unsafe.shape[0])])
    X_test = torch.tensor(X_test_all, dtype=torch.float32)
    print(mode)

    # ============== 训练自编码器 ==============
    X_train = X_train.to(device)
    autoencoder = SmallAutoencoder(input_dim=X_train.shape[1], bottleneck_dim=2)
    autoencoder.to(device)
    autoencoder.train()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=1e-3)

    epochs = 2000
    
    from time import perf_counter
    start = perf_counter()
    
    best_loss = 1
    for epoch in range(epochs):
        optimizer.zero_grad()
        X_hat = autoencoder(X_train)
        loss = criterion(X_hat, X_train)

        if (epoch+1) % 50 == 0:
            if best_loss -loss < 1e-5:
                break
            if loss < best_loss:
                best_loss = loss
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")
        loss.backward()
        optimizer.step()
    elapsed = perf_counter() - start
    print(f"Time elapsed: {elapsed:.6f} seconds")
    exit(0)
    # ============== 测试重构误差 + ROC AUC ==============
    # print(X_test.shape)
    # exit(0)
    a = torch.tensor([0.5 for i in range(len(layer_chosen))])
    autoencoder.eval()
    with torch.no_grad():
        X_test_hat = autoencoder(X_test)
        reconstruction_errors = ((X_test_hat - X_test)**2).mean(dim=1).cpu().numpy()
        a_hat = autoencoder(a)
    # auc = roc_auc_score(y_test_all, reconstruction_errors)
    # print(f"✅ ROC AUC: {auc:.4f}")

    threshold = np.percentile(reconstruction_errors[:X_test_safe.shape[0]], 99)
    # print(threshold)
    print(((a-a_hat)**2).mean(dim=0).cpu().numpy())
    # y_pred = (reconstruction_errors > threshold).astype(int)
    # print(classification_report(y_test_all, y_pred, digits=4))

    # torch.save(autoencoder.state_dict(), f"{save_dir}/autoencoder-{p0}.pth")
    # print(f"✅ 已保存 autoencoder 权重到: {save_dir}/autoencoder-{p0}.pth")

if __name__ == "__main__":
    main()
    
    
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from sklearn.metrics import classification_report, roc_auc_score
# import os
# import numpy as np
# import copy
# import random
# def set_seed(seed):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)  # if using multi-GPU
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#     os.environ['PYTHONHASHSEED'] = str(seed)
# class SmallAutoencoder(nn.Module):
#     def __init__(self, input_dim, bottleneck_dim=4):
#         super(SmallAutoencoder, self).__init__()
        
#         hidden1 = max(input_dim // 2, bottleneck_dim * 2)  # 保证至少比 bottleneck 大
#         hidden2 = max(input_dim // 4, bottleneck_dim * 2)  # 再缩小

#         self.encoder = nn.Sequential(
#             nn.Linear(input_dim, hidden1),
#             nn.ReLU(),
#             nn.Linear(hidden1, hidden2),
#             nn.ReLU(),
#             nn.Linear(hidden2, bottleneck_dim)
#         )

#         self.decoder = nn.Sequential(
#             nn.Linear(bottleneck_dim, hidden2),
#             nn.ReLU(),
#             nn.Linear(hidden2, hidden1),
#             nn.ReLU(),
#             nn.Linear(hidden1, input_dim)
#         )

#     def forward(self, x):
#         z = self.encoder(x)
#         x_hat = self.decoder(z)
#         return x_hat
# def main():
#     # ======================
#     # 1. 准备数据
#     # ======================
#     set_seed(42)
#     model = 'qwen'
#     import sys
#     sys.path.append(f"/home/u2021201665/code/SCAV-{model}")
#     from utils import train_data,test_data
#     from classifier import LinearClassifier

#     void=False
#     device = "cuda:6"
#     dtype = torch.float16
#     p0=0.9
#     np.random.seed(42)
#     layer_chosen = []
#     test_pos,test_neg = test_data(device=device,void=void)
#     test_pos = test_pos.to(dtype)
#     test_neg = test_neg.to(dtype)

#     layers = [i+1 for i in range(32)]
#     classifiers = {}
#     scores = []
#     save_dir = f"SCAV-{model}/pth"
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
#         print(f"✅ 创建目录: {save_dir}")
#     else:
#         print(f"📂 目录已存在: {save_dir}")

#     for i in layers:
#         if void:
#             classifier_path = f"SCAV-{model}/pth/advbench-none/{i}.pth"
#         else:
#             classifier_path = f"SCAV-{model}/pth/advbench/{i}.pth"
            
#         classifier = LinearClassifier() # 假设输入维度 4096，输出 1
#         classifier.load_state_dict(torch.load(classifier_path, map_location=device))  # 加载权重
#         classifier.to(device,dtype=dtype)  # 转移到指定 device
#         classifier.eval()
#         classifiers[i] = classifier
#         input1 = test_pos[:,i,:].to(device,dtype=dtype)
#         input2 = test_neg[:,i,:].to(device,dtype=dtype)
#         num1 = input1.shape[0]
#         num2 = input2.shape[0]

#         # 预测结果
#         with torch.no_grad():
#             output1 = classifier(input1)
#             output2 = classifier(input2)
#         score = (1-torch.sum(output1)/num1+torch.sum(output2)/num2)/2
#         # print(score)
#         scores.append(score)
#         if score >=p0:
#             layer_chosen.append(i)
#         else:
#             continue
#     print(layer_chosen)

#     possibilities_safe = []
#     possibilities_unsafe = []
#     for i in layer_chosen:
#         safe_input = test_pos[:, i, :]
#         unsafe_input = test_neg[:, i, :]
#         with torch.no_grad():
#             outputs = classifiers[i](safe_input)
#             possibilities_safe.append(outputs)
            
#             outputs = classifiers[i](unsafe_input)
#             possibilities_unsafe.append(outputs)

#      # 你已经准备好的 possibilities_safe 和 possibilities_unsafe
#     possibilities_safe = torch.stack(possibilities_safe, dim=1).float().cpu().numpy()
#     possibilities_unsafe = torch.stack(possibilities_unsafe, dim=1).float().cpu().numpy()

#     # ============== 划分训练、验证、测试集 ==============
#     n_safe_total = possibilities_safe.shape[0]
#     assert n_safe_total == 400, "你目前一共应有400条安全样本"

#     n_train_safe = 320
#     n_val_safe = 40
#     n_test_safe = 40

#     # 随机打乱索引
#     indices = np.arange(n_safe_total)
#     np.random.shuffle(indices)

#     # 划分
#     train_indices = indices[:n_train_safe]
#     val_indices = indices[n_train_safe:n_train_safe+n_val_safe]
#     test_indices = indices[n_train_safe+n_val_safe:]

#     # 取出数据
#     X_train_safe = possibilities_safe[train_indices]
#     X_val_safe = possibilities_safe[val_indices]
#     X_test_safe = possibilities_safe[test_indices]

#     # ============== 构造训练、验证、测试集 ==============
#     X_train = torch.tensor(X_train_safe, dtype=torch.float32)
#     X_val   = torch.tensor(X_val_safe, dtype=torch.float32)

#     # 测试集中包含安全 + 有害
#     X_test_all = np.concatenate([X_test_safe, possibilities_unsafe], axis=0)
#     y_test_all = np.concatenate([
#         np.zeros(X_test_safe.shape[0]), 
#         np.ones(possibilities_unsafe.shape[0])
#     ])

#     X_test = torch.tensor(X_test_all, dtype=torch.float32)
#     y_test = torch.tensor(y_test_all, dtype=torch.float32)

#     # ============== 定义模型 ==============
#     autoencoder = SmallAutoencoder(input_dim=X_train.shape[1], bottleneck_dim=2)
#     criterion = nn.MSELoss()
#     optimizer = optim.Adam(autoencoder.parameters(), lr=1e-3)

#     # ============== Early stopping 参数 ==============
#     patience = 30     # 连续多少次未提升就停止
#     min_delta = 1e-5  # 最小改善幅度
#     best_val_loss = float('inf')
#     patience_counter = 0

#     best_model_state = copy.deepcopy(autoencoder.state_dict())  # 初始化备份

#     # ============== 训练循环 ==============
#     epochs = 5000
#     train_losses = []
#     val_losses = []

#     for epoch in range(epochs):
#         # 训练阶段
#         autoencoder.train()
#         optimizer.zero_grad()
#         X_hat = autoencoder(X_train)
#         train_loss = criterion(X_hat, X_train)
#         train_loss.backward()
#         optimizer.step()

#         # 验证阶段
#         autoencoder.eval()
#         with torch.no_grad():
#             X_val_hat = autoencoder(X_val)
#             val_loss = criterion(X_val_hat, X_val)

#         train_losses.append(train_loss.item())
#         val_losses.append(val_loss.item())

#         # 打印日志
#         if (epoch+1) % 50 == 0:
#             print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss.item():.6f}, Val Loss: {val_loss.item():.6f}")

#         # Early stopping 检查
#         if val_loss.item() < best_val_loss - min_delta:
#             best_val_loss = val_loss.item()
#             patience_counter = 0
#             best_model_state = copy.deepcopy(autoencoder.state_dict())  # 保存最好模型
#         else:
#             patience_counter += 1

#         if patience_counter >= patience:
#             print(f"Early stopping at epoch {epoch+1}")
#             break

#     # ============== 恢复最优模型参数 ==============
#     autoencoder.load_state_dict(best_model_state)
#     print("训练完成，恢复至验证集最优模型。")

#     # ============== 测试重构误差 + ROC AUC ==============
#     autoencoder.eval()
#     with torch.no_grad():
#         X_test_hat = autoencoder(X_test)
#         reconstruction_errors = ((X_test_hat - X_test)**2).mean(dim=1).cpu().numpy()

#     auc = roc_auc_score(y_test_all, reconstruction_errors)
#     print(f"✅ ROC AUC: {auc:.4f}")

#     threshold = np.percentile(reconstruction_errors[:X_test_safe.shape[0]], 95)
#     y_pred = (reconstruction_errors > threshold).astype(int)
#     print(classification_report(y_test_all, y_pred, digits=4))

#     torch.save(autoencoder.state_dict(), f"{save_dir}/autoencoder-{p0}.pth")
#     print(f"✅ 已保存 autoencoder 权重到: {save_dir}/autoencoder-{p0}.pth")

# if __name__ == "__main__":
#     main()
    
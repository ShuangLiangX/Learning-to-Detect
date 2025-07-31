import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from classifier import LinearClassifier
import argparse
import json
import random
from utils import train_data,test_data
# 自定义数据集类
class TensorDataset(Dataset): 
    def __init__(self, tensor_data, label, selected_dim=25, dtype=torch.float16):
        """
        tensor_data: [samples, 33, 4096]
        label: 样本对应的标签
        selected_dim: 33维中选取的特定维度 (默认为25)
        dtype: 数据类型，默认是 torch.float32
        """
        # 选取指定维度的数据，形状变为 [samples, 4096]，并统一数据类型
        self.data = tensor_data[:, selected_dim, :].to(dtype)
        self.label = torch.full((tensor_data.shape[0],), label, dtype=dtype)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]


# 训练函数
def train_classifier(safety, unsafety, model_path, selected_dim=25, epochs=30, batch_size=5, lr=0.01, device='cuda', dtype=torch.float32):
    safety = safety.to(device)
    unsafety = unsafety.to(device)
    dataset1 = TensorDataset(safety, label=0, selected_dim=selected_dim, dtype=dtype)
    dataset2 = TensorDataset(unsafety, label=1, selected_dim=selected_dim, dtype=dtype)
    full_dataset = torch.utils.data.ConcatDataset([dataset1, dataset2])
    dataloader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)

    model = LinearClassifier(input_dim=safety.shape[2], output_dim=1).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)


    patience = 5
    wait = 0
    best_loss = float('inf')

    for epoch in range(epochs):
        total_loss = 0.0
        num = 0
        model.train()
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num += 1
        total_loss /= num
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss:.8f}")

        scheduler.step(total_loss)  # 自动调整 lr

        # 检测与上一轮的 loss 差别
        if epoch > 0 and abs(total_loss - prev_loss) < 1e-4:
            print("Loss difference is less than 1e-4. Stopping training.")
            break

        prev_loss = total_loss  # 更新上一轮的 loss

    # 保存训练后的模型参数
    torch.save(model.state_dict(), model_path)
    print(f"Model parameters saved to: {model_path}")

    return model

# 测试函数
def test_classifier(model_path, test_data, selected_dim=25,device='cuda:7',input_dim=4096,output_dim=1, dtype=torch.float16):
    # 初始化模型
    model = LinearClassifier(input_dim=input_dim, output_dim=output_dim)
    model.load_state_dict(torch.load(model_path,map_location=device))  # 加载保存的模型参数
    model = model.to(device=device,dtype=dtype)
    model.eval()

    # 选取指定维度的数据进行测试
    test_input = test_data[:, selected_dim, :].to(dtype).to(device) # [samples, 4096]

    # 预测结果
    with torch.no_grad():
        outputs = model(test_input)

    # print("Predicted class labels:", predictions)
    return outputs



def main():
    # 创建参数解析器
    parser = argparse.ArgumentParser(description="Train and test a linear classifier.")
    
    # 添加训练参数
    parser.add_argument("--train", action="store_true", help="Run training mode.")
    parser.add_argument("--model_type", type=str, default="linear_classifier2.pth", help="Path to save the trained model.")
    # parser.add_argument("--selected_dim", type=int, default=16, help="Selected dimension for the model.")
    parser.add_argument("--epochs", type=int, default=5000, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for training.")
    parser.add_argument("--void", type=bool, default=False, help="Learning rate for training.")
    # 添加测试参数 
    parser.add_argument("--test", action="store_true", help="Run testing mode.")
    parser.add_argument("--test_file", type=str, default="test.pth", help="Path to test data file.")
    
    # 解析参数
    args = parser.parse_args()
    void = args.void
    # 根据参数执行训练或测试
    if args.train:
        if 'pixart' == args.model_type:
            data = torch.load("/home/u2021201665/asset/QA-vicuna/pixart_answer.pth")
            safety = data[10:,:,:]
            unsafety = data[0:10,:,:] 
        elif "SafetyBenchvoid-none" == args.model_type:
            safety = torch.load("/home/u2021201665/asset/QA-vicuna/deepseekvoid-none_answer.pth")[:10,:,:]
            unsafety = torch.load("/home/u2021201665/asset/QA-vicuna/SafetyBenchvoid-none_answer.pth")[:10,:,:]      
        elif "SafetyBenchvoid" == args.model_type:

            safe = torch.load("/home/u2021201665/asset/QA-vicuna/deepseekvoid_answer.pth")
            unsafe = torch.load("/home/u2021201665/asset/QA-vicuna/SafetyBenchvoid_answer.pth")
            # print("ssss")
            with open("instructions/deepseekvoid-safe.json",'r') as f:
                data_safe = json.load(f)
            with open("instructions/SafetyBenchvoid-safe.json",'r') as f:
                data_unsafe = json.load(f)
            safe_indice = [int(i) for i in list(data_safe.keys())]
            unsafe_indice = list(int(i) for i in data_unsafe.keys())
                    # 从 safe_indice 中随机挑选 10 个作为训练集
            train_safe = random.sample(safe_indice, 10)
            # 对于 unsafe_indice，我们可以将其全部作为训练集或测试集，或者也进行类似的划分
            train_unsafe = random.sample(unsafe_indice, 10)  # 假设我们也挑选 10 个
            safety = safe[train_safe]
            unsafety = unsafe[train_unsafe]              
        elif "SafetyBench0" == args.model_type:
            # print("ssss")
            with open("data/GQA-safety.json",'r') as f:
                data_safe = json.load(f)
            with open("data/SafetyBench0-safe.json",'r') as f:
                data_unsafe = json.load(f)
            safe = torch.load("/home/u2021201665/asset/QA-vicuna/GQA_answer.pth")
            unsafe = torch.load("/home/u2021201665/asset/QA-vicuna/SafetyBench0_answer.pth")

            safe_indice = [int(i) for i in list(data_safe.keys())]
            unsafe_indice = list(int(i) for i in data_unsafe.keys())
                    # 从 safe_indice 中随机挑选 10 个作为训练集
            train_safe = random.sample(safe_indice, 40)
            # 对于 unsafe_indice，我们可以将其全部作为训练集或测试集，或者也进行类似的划分
            train_unsafe = random.sample(unsafe_indice, 40)  # 假设我们也挑选 10 个
            safety = safe[train_safe]
            unsafety = unsafe[train_unsafe]
        elif "SafetyBench0-des" == args.model_type:
            safety = torch.load("/home/u2021201665/asset/QA-vicuna/GQA-des_answer.pth")[:10,:,:]
            unsafety = torch.load("/home/u2021201665/asset/QA-vicuna/SafetyBench0-des_answer.pth")[:10,:,:]    
        elif "advbench" in args.model_type:
            safety,unsafety = train_data()
        elif "advbench-none" in args.model_type:
            safety,unsafety = train_data(void=void)
        for i in range(33):
            trained_model = train_classifier(
                safety,
                unsafety,
                f'pth/{args.model_type}/{i}.pth',
                i,
                args.epochs,
                args.batch_size,
                args.lr
            )
    elif args.test:
        SafetyBench_scores = []
        pixart_scores = []
        SafetyBenchvoid_scores = []
        train_score = []
        test_score = []
        harmful = []
        a = []
        test_unsafe = []
        test_safe = []
        
        safetybench = []
        fig = []
        vajm = []
        umk = []
        hades = []
        
        device="cuda:7"
        for i in range(33): 
            print(i)
            if  i==0:
                continue
            dim = i
            type = args.model_type
            model_path = f'pth/{type}/{i}.pth'
            # model_path = f"/home/u2021201665/asset/llava-test/pixart/weights/{i}.pth"

            # #=======================safetybench0==================
            # with open("instructions/GQA.json",'r') as f:
            #     data_safe = json.load(f)
            # with open("instructions/SafetyBench0-safe.json",'r') as f:
            #     data_unsafe = json.load(f)
            # safe = torch.load("/home/u2021201665/asset/QA-vicuna/GQA_answer.pth", map_location=device)
            # unsafe = torch.load("/home/u2021201665/asset/QA-vicuna/SafetyBench0_answer.pth", map_location=device)

            # safe_indice = [int(i) for i in list(data_safe.keys())]
            # unsafe_indice = list(int(i) for i in data_unsafe.keys())

            # safe = safe[safe_indice]
            # unsafe = unsafe[unsafe_indice]




            # # # # 加载测试数据

            # i = test_classifier(model_path, safe, selected_dim=dim)
            # print("GQA score:",1-torch.sum(i)/i.shape[0])
            # score1 = (1-torch.sum(i)/i.shape[0]).cpu().item()
            # i = test_classifier(model_path,unsafe,selected_dim=dim)
            # print("SafetyBench0 score:",torch.sum(i)/i.shape[0])
            # score2 = (torch.sum(i)/i.shape[0]).cpu().item()
            # SafetyBench_scores.append((score1+score2)/2)
            # #i = test_classifier(model_path, test_unsafe_tensor, selected_dim=16)
            # # print(i)
            # # print(test_unsafe[i])
            # #==========safetybench3-filter============
            # with open("data/SafetyBench3-unsafe.json",'r') as f:
            #     data = json.load(f)
            # data_tensor = torch.load("/home/u2021201665/asset/QA/SafetyBench0_answer.pth")
            # indice = [int(i) for i in list(data.keys()) if i not in list(data_unsafe.keys())]
            # data = data_tensor[indice]
            # i = test_classifier(model_path,data_tensor,selected_dim=dim)
            # print("Safetybench3-unsafe score:",1-torch.sum(i)/i.shape[0])
            # score = (1-torch.sum(i)/i.shape[0]).cpu().item()
            # harmful.append(score)
            # # a.append(i[0])

        
            # data_tensor = torch.load("/home/u2021201665/asset/QA/pixart_answer.pth")    
            # i=test_classifier(model_path,data_tensor[10:,:,:],selected_dim=dim)
            # print("pixart_fine score:",1-torch.sum(i)/i.shape[0])
            # score1 = (1-torch.sum(i)/i.shape[0]).cpu().item()
            # i=test_classifier(model_path,data_tensor[:10,:,:],selected_dim=dim)
            # print("pixart_harmscore:",torch.sum(i)/i.shape[0])
            # score2 = (torch.sum(i)/i.shape[0]).cpu().item()
            # pixart_scores.append((score1+score2)/2)

            # data_tensor = torch.load("/home/u2021201665/asset/QA-vicuna/deepseekvoid_answer.pth", map_location=device)
            # i=test_classifier(model_path,data_tensor,selected_dim=dim)
            # print("1 score:",1-torch.sum(i)/i.shape[0])
            # score1 = (1-torch.sum(i)/i.shape[0]).cpu().item()

            # data_tensor = torch.load("/home/u2021201665/asset/QA-vicuna/SafetyBenchvoid_answer.pth", map_location=device)
            # i=test_classifier(model_path,data_tensor,selected_dim=dim)
            # print("2 score:",torch.sum(i)/i.shape[0])
            # score2 = (torch.sum(i)/i.shape[0]).cpu().item()
            # SafetyBenchvoid_scores.append((score1+score2)/2)
            #==========advbench-train================
    
            safety,unsafety = train_data(device=device,void=void)
            i=test_classifier(model_path,safety,selected_dim=dim)
            print("safe train score:",1-torch.sum(i)/i.shape[0])
            score1 = 1-torch.sum(i)/i.shape[0]

            i=test_classifier(model_path,unsafety,selected_dim=dim)
            print("unsafe train score:",torch.sum(i)/i.shape[0])
            score2 = torch.sum(i)/i.shape[0]
            train_score.append((score1+score2).cpu().item()/2)
            #==========advbench-test================
            safety,unsafety = test_data(device=device,void=void)
            i=test_classifier(model_path,safety,selected_dim=dim)
            test_safe.append(torch.sum(i).cpu().item()/i.shape[0])
            print("safe test score:",1-torch.sum(i)/i.shape[0])
            score1 = 1-torch.sum(i)/i.shape[0]
            i=test_classifier(model_path,unsafety,selected_dim=dim)
            print("unsafe test score:",torch.sum(i)/i.shape[0])
            score2 = torch.sum(i)/i.shape[0]
            test_unsafe.append(torch.sum(i).cpu().item()/i.shape[0])
            test_score.append((score1+score2).cpu().item()/2)
            # #==========advbench-test-None================
            # safety = torch.load("/home/u2021201665/asset/QA-vicuna/gpt-None_answer.pth", map_location=device)
            # unsafety = torch.load("/home/u2021201665/asset/QA-vicuna/advbench-test-None_answer.pth", map_location=device) 
            # i=test_classifier(model_path,safety,selected_dim=dim)
            # print("safe-None test score:",1-torch.sum(i)/i.shape[0])
            # score1 = 1-torch.sum(i)/i.shape[0]
            # i=test_classifier(model_path,unsafety,selected_dim=dim)
            # print("unsafe-None test score:",torch.sum(i)/i.shape[0])
            # score2 = torch.sum(i)/i.shape[0]
            # test_score.append((score1+score2).cpu().item()/2)
            unsafe = torch.load(f"/home/u2021201665/asset/QA-vicuna/SafetyBench_answer.pth",map_location=device)
            i=test_classifier(model_path,unsafe,selected_dim=dim)
            safetybench.append(torch.sum(i).cpu().item()/i.shape[0])
            
            unsafe = torch.load(f"/home/u2021201665/asset/QA-vicuna/FigImage_answer.pth",map_location=device) 
            i=test_classifier(model_path,unsafe,selected_dim=dim)
            fig.append(torch.sum(i).cpu().item()/i.shape[0])
            
            unsafe = torch.load(f"/home/u2021201665/asset/QA-vicuna/SafetyBench-vajm_answer.pth",map_location=device)
            i=test_classifier(model_path,unsafe,selected_dim=dim)
            vajm.append(torch.sum(i).cpu().item()/i.shape[0])
            
            unsafe = torch.load(f"/home/u2021201665/asset/QA-vicuna/SafetyBench-umk_answer.pth",map_location=device)
            i=test_classifier(model_path,unsafe,selected_dim=dim)
            umk.append(torch.sum(i).cpu().item()/i.shape[0])
            
            unsafe = torch.load(f"/home/u2021201665/asset/QA-vicuna/HADES_answer.pth",map_location=device)
            i=test_classifier(model_path,unsafe,selected_dim=dim)
            hades.append(torch.sum(i).cpu().item()/i.shape[0])
        # print("SafetyBench0:",SafetyBench_scores)
        # print("pixart:",pixart_scores)
        # print("SafetyBenchvoid:",SafetyBenchvoid_scores)
        # # print("train:",train_score)
        print("test:",test_score)
        # print("harmful",harmful)
        # print(a)
        print("test_safe=",test_safe)
        print("test_unsafe=",test_unsafe)
        print("MM-SafetyBench=",safetybench)
        print("FigImg=",fig)
        print("VAJM=",vajm)
        print("UMK=",umk)
        print("HADES=",hades)
    else:
        print("Please specify either --train or --test mode.")

if __name__ == "__main__":
    main()

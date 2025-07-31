import torch
import sys
sys.path.append("SCAV-vicuna")
from classifier import LinearClassifier
import argparse
import json
import random
import numpy as np
from sklearn.metrics import precision_recall_curve,auc
from sklearn.metrics import roc_curve
import joblib
import random
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import pickle
from autoencoder import SmallAutoencoder
import torch.nn.functional as F
from time import perf_counter
random.seed(42)
# 自定义数据集类

def test_classifier(model_path, test_data, selected_dim=25,device='cuda:7',input_dim=4096,output_dim=1, dtype=torch.float16):
    # 初始化模型
    model = LinearClassifier(input_dim=input_dim, output_dim=output_dim)
    model.load_state_dict(torch.load(model_path,map_location=device))  # 加载保存的模型参数
    model = model.to(dtype).to(device)
    model.eval()

    # 选取指定维度的数据进行测试
    test_input = test_data[:, selected_dim, :].to(dtype).to(device) # [samples, 4096]

    # 预测结果
    with torch.no_grad():
        outputs = model(test_input)

    return outputs

def evaluate_AUPRC(true_labels, scores):
    precision_arr, recall_arr, threshold_arr = precision_recall_curve(true_labels, scores)
    auprc = auc(recall_arr, precision_arr)

    # 计算所有点的 F1 分数
    f1_scores = 2 * (precision_arr * recall_arr) / (precision_arr + recall_arr + 1e-8)

    # 找到最大 F1 分数及其对应的阈值（注意 threshold 数量比 precision/recall 少1）
    best_index = f1_scores[:-1].argmax()  # 最后一项没有对应 threshold
    best_threshold = threshold_arr[best_index]
    best_f1 = f1_scores[best_index]

    # print(f"Best F1: {best_f1:.4f} at threshold: {best_threshold:.4f}")
    return auprc,best_f1


def evaluate_AUROC(true_labels, scores):
    fpr, tpr, thresholds = roc_curve(true_labels, scores)
    # print(thresholds)
    auroc = auc(fpr, tpr)
    return auroc      


def main():
    # 创建参数解析器
    parser = argparse.ArgumentParser(description="Train and test a linear classifier.")
    
    # 添加训练参数
    parser.add_argument("--pth", type=str, default="advbench", help="Path to save the trained model.")
    parser.add_argument("--dataset", type=str, default="FigStep", help="Path to save the trained model.")
    parser.add_argument("--model", type=str, default="vicuna")
    parser.add_argument("--layers", type=str, default="all")
    parser.add_argument("--void", type=bool, default=False)
    parser.add_argument("--single", type=bool, default=False)
    parser.add_argument("--mode", type=str, default="autoencoder")
    
    # 解析参数
    args = parser.parse_args()
    device = "cpu"
    # 根据参数执行训练或测试
    pth = args.pth
    model_name = args.model
    layer_type = args.layers
    # a = ["safetybench","FigImg","vajm","umk","HADES"]
    a = ["FigImg"]
    p0=0.9     
    score_list = []
    for dataset in a:
        if dataset == 'FigStep':
            safe = torch.load(f"/home/u2021201665/asset/QA-{model_name}/benign_questions_answer.pth",map_location=device)
            unsafe = torch.load(f"/home/u2021201665/asset/QA-{model_name}/safebench_answer.pth",map_location=device)  
  
        elif dataset == "XSTest":
            safe = torch.load(f"/home/u2021201665/asset/QA-{model_name}/xstest_safe_answer.pth",map_location=device)
            unsafe = torch.load(f"/home/u2021201665/asset/QA-{model_name}/xstest_unsafe_answer.pth",map_location=device)  
                
        elif dataset == "safetybench":
            safe = torch.load(f"/home/u2021201665/asset/QA-{model_name}/mm-vet_answer.pth",map_location=device)
            unsafe = torch.load(f"/home/u2021201665/asset/QA-{model_name}/SafetyBench_answer.pth",map_location=device)
        elif dataset == "FigImg":
            safe = torch.load(f"/home/u2021201665/asset/QA-{model_name}/mm-vet_answer.pth",map_location=device)
            unsafe = torch.load(f"/home/u2021201665/asset/QA-{model_name}/FigImage_answer.pth",map_location=device) 
            
        elif dataset == "vajm":
            with open("HiddenData/SafetyBench-vajm.json", "r") as f:
                data = json.load(f)            
            
            safe = torch.load(f"/home/u2021201665/asset/QA-{model_name}/mm-vet_answer.pth",map_location=device) 
                       
            unsafe = torch.load(f"/home/u2021201665/asset/QA-{model_name}/SafetyBench-vajm_answer.pth",map_location=device)

            random_numbers = random.sample(range(unsafe.shape[0]), 218)
            unsafe = unsafe[random_numbers]
            
        elif dataset == "umk":
            with open("HiddenData/SafetyBench-umk.json", "r") as f:
                data = json.load(f)            
            
            safe = torch.load(f"/home/u2021201665/asset/QA-{model_name}/mm-vet_answer.pth",map_location=device)            
            unsafe = torch.load(f"/home/u2021201665/asset/QA-{model_name}/SafetyBench-umk_answer.pth",map_location=device)
            random_numbers = random.sample(range(unsafe.shape[0]), 218)
            unsafe = unsafe[random_numbers]
        elif dataset == "ours":
            safe = torch.load(f"/home/u2021201665/asset/QA-{model_name}/mm-vet_answer.pth",map_location=device)            
            unsafe = torch.load(f"/home/u2021201665/asset/QA-{model_name}/SafetyBench-ours_answer.pth",map_location=device)
            random_numbers = random.sample(range(unsafe.shape[0]), 218)
            unsafe = unsafe[random_numbers]
        elif dataset == "HADES":
            safe = torch.load(f"/home/u2021201665/asset/QA-{model_name}/mm-vet_answer.pth",map_location=device)            
            unsafe = torch.load(f"/home/u2021201665/asset/QA-{model_name}/HADES_answer.pth",map_location=device)
            random_numbers = random.sample(range(unsafe.shape[0]), 218)
            unsafe = unsafe[random_numbers]
            
        data = torch.cat([safe,unsafe],dim=0)

        if args.model == "vicuna":     
            if p0 == 0.99:       
                layers = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32] 
            elif p0 == 0.97:
                layers = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
            elif p0 == 0.95:                
                layers = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
            elif p0== 0.9:
                layers = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
            else:
                layers = [1,2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
            autoencoder_loaded = SmallAutoencoder(input_dim=len(layers), bottleneck_dim=2)
            autoencoder_loaded.load_state_dict(torch.load(f"SCAV-{model_name}/pth/autoencoder-{p0}.pth"))
            autoencoder_loaded.eval()
            sim = torch.tensor([
                7.3804975e-02, 3.4381390e-02, 3.0855209e-02, 1.8159200e-02, 1.4107537e-02,
                8.9998487e-03, 7.0529669e-03, 6.7855953e-03, 5.5714250e-03, 6.6276016e-03,
                5.1632882e-03, 4.4914633e-03, 2.9602870e-03, 2.2428117e-03, 1.5119941e-03,
                1.0528385e-03, 7.0228276e-04, 6.1876251e-04, 2.6273206e-04, 5.8053061e-04,
                1.1917822e-03, 4.3062493e-04, 1.5127212e-04, 1.2611449e-04, 1.5195012e-04,
                2.7676896e-04, 1.4192387e-04, 2.7303546e-04, 2.1554529e-06, 5.0924718e-06,
                4.3514819e-04
            ])
            threshold = 0.3051467214524745
            
        elif args.model =="qwen":
            # layers = [2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
            if p0 == 0.99:
                layers = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
            elif p0 == 0.97:
                layers = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
            elif p0 == 0.95:
                layers = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
            elif p0==0.9:
                layers = [2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
            elif p0 == 0.8:
                layers = [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
            else:
                layers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
            autoencoder_loaded = SmallAutoencoder(input_dim=len(layers), bottleneck_dim=2)
            autoencoder_loaded.load_state_dict(torch.load(f"SCAV-{model_name}/pth/autoencoder-{p0}.pth"))
            autoencoder_loaded.eval()
            sim = torch.tensor([
                1.7795015e-02, 4.6132743e-02, 4.2098455e-02, 4.2640720e-02, 4.2212635e-02,
                3.8789470e-02, 2.1038610e-02, 1.0491510e-02, 9.4435867e-03, 6.7150989e-03,
                3.5425550e-03, 6.0830317e-03, 9.1507230e-03, 5.0185663e-03, 2.0167227e-03,
                3.3358939e-03, 3.1662374e-03, 2.1124729e-03, 2.8973038e-03, 1.4583316e-03,
                1.0197125e-04, 1.2604520e-05, 1.5524774e-05, 4.3310225e-05, 2.2946540e-04,
                1.9533559e-06, 7.4920434e-05, 5.9574470e-04, 1.2131371e-04, 2.6602297e-06
            ])
            threshold = 0.1716409805417061
        else:
            if p0 == 0.99:
                layers = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
            elif p0 == 0.97:
                layers = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
            elif p0 == 0.95:
                layers = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
            else:
                layers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
            autoencoder_loaded = SmallAutoencoder(input_dim=len(layers), bottleneck_dim=2)
            autoencoder_loaded.load_state_dict(torch.load(f"SCAV-{model_name}/pth/autoencoder-{p0}.pth"))
            autoencoder_loaded.eval()
            sim = torch.tensor([
                7.5133495e-02, 8.7679848e-02, 1.7810414e-02, 1.4825416e-02, 9.3930122e-03,
                8.1123617e-03, 6.9848225e-03, 5.8898367e-03, 5.5302503e-03, 4.5909202e-03,
                4.5087836e-03, 3.3759805e-03, 3.5476300e-03, 2.8716703e-03, 2.6650999e-03,
                1.8570668e-03, 1.1626951e-03, 8.6651312e-04, 2.2002305e-03, 5.0102919e-04,
                1.6152936e-03, 2.9947981e-04, 1.6926688e-03, 1.6712639e-04, 1.1767133e-03,
                2.6412841e-03, 1.3792213e-03, 5.0069750e-03, 9.2769263e-04, 3.0321009e-03,
                7.4505806e-08, 2.8565738e-03
            ])
            threshold = 0.324619482755661
        # layers = [i+1 for i in range(32)]
        start = perf_counter()

        possibilities = []
        for i in layers: 
            dim = i
            #pth path
            model_path = f'SCAV-{model_name}/pth/{pth}/{i}.pth'
            # compute score 
            scores = test_classifier(model_path, data,device=device, selected_dim=dim) 
            possibilities.append(scores)
        
        possibilities = torch.stack(possibilities,dim=1)
        #准备标签
        label1 = torch.zeros(safe.shape[0])
        # print(label1.shape)
        label2 = torch.ones(unsafe.shape[0])
        labels = torch.cat([label1,label2],dim=0).cpu()
        
        X_new = possibilities.to(torch.float32)
        # print(X_new.shape)
        # exit(0)
        if args.mode == "autoencoder":
            with torch.no_grad():
                X_new_hat = autoencoder_loaded(X_new)
                scores = ((X_new_hat - X_new) ** 2).mean(dim=1)                
        elif args.mode == "single":
            scores = X_new[:,-1].cpu().numpy()
        elif args.mode == "average":
            threshold = 0.1  # 设定阈值

            # 检查每个样本是否有任意维度超限 [B]
            binary_flags = (X_new > threshold).any(dim=1).float()  # 0/1
            # 将二值标签转为连续分数（添加微小噪声避免AUROC计算错误）
            scores = binary_flags + torch.rand_like(binary_flags) * 0.001  # [B], 范围[0,1.001]

            # print(scores)
        print(0)
        
        elapsed = perf_counter() - start
        print(f"Time elapsed: {elapsed:.6f} seconds")
        
        AUPRC,best_f1 = evaluate_AUPRC(labels, scores)
        AUROC = evaluate_AUROC(labels, scores)            
        print(dataset)
        print(f"AUPRC for: {AUPRC:.4f}")
        print(f"AUROC for: {AUROC:.4f}")
        print(f"best F1: {best_f1:.4f}")
        score_list.append(AUROC)
    print(sum(score_list)/5)
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    
    
    
        
if __name__ == "__main__":
    set_seed(42)  # 设置一个固定的种子
    main()

import torch
import sys
sys.path.append("vicuna")
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

def test_classifier(model_path, test_data, selected_dim=25,device='cuda:7',input_dim=4096,output_dim=1, dtype=torch.float16):
    model = LinearClassifier(input_dim=input_dim, output_dim=output_dim)
    model.load_state_dict(torch.load(model_path,map_location=device))
    model = model.to(dtype).to(device)
    model.eval()
    test_input = test_data[:, selected_dim, :].to(dtype).to(device)
    with torch.no_grad():
        outputs = model(test_input)

    return outputs

def evaluate_AUPRC(true_labels, scores):
    precision_arr, recall_arr, threshold_arr = precision_recall_curve(true_labels, scores)
    auprc = auc(recall_arr, precision_arr)
    return auprc


def evaluate_AUROC(true_labels, scores):
    fpr, tpr, thresholds = roc_curve(true_labels, scores)
    auroc = auc(fpr, tpr)
    return auroc      


def main():
    parser = argparse.ArgumentParser(description="Train and test a linear classifier.")
    parser.add_argument("--pth", type=str, default="advbench", help="Path to save the trained model.")
    parser.add_argument("--dataset", type=str, default="FigStep", help="Path to save the trained model.")
    parser.add_argument("--layers", type=str, default="all")
    parser.add_argument("--device", type=str, default="cpu")
    

    args = parser.parse_args()
    device = args.device
    pth = args.pth
    layer_type = args.layers
    
    datasets = ["safetybench","FigImg","vajm","umk","HADES"]
    p0=0.9     
    score_list = []
    for dataset in datasets:             
        if dataset == "safetybench":
            safe = torch.load(f"asset/HiddenStates/mm-vet_answer.pth",map_location=device)
            unsafe = torch.load(f"asset/HiddenStates/SafetyBench_answer.pth",map_location=device)
        elif dataset == "FigImg":
            safe = torch.load(f"asset/HiddenStates/mm-vet_answer.pth",map_location=device)
            unsafe = torch.load(f"asset/HiddenStates/FigImage_answer.pth",map_location=device)             
        elif dataset == "vajm":
            with open("Benchmarks/SafetyBench-vajm.json", "r") as f:
                data = json.load(f)           
            
            safe = torch.load(f"asset/HiddenStates/mm-vet_answer.pth",map_location=device) 
                       
            unsafe = torch.load(f"asset/HiddenStates/SafetyBench-vajm_answer.pth",map_location=device)

            random_numbers = random.sample(range(unsafe.shape[0]), 218)
            unsafe = unsafe[random_numbers]
            
        elif dataset == "umk":
            with open("Benchmarks/SafetyBench-umk.json", "r") as f:
                data = json.load(f)            
            
            safe = torch.load(f"asset/HiddenStates/mm-vet_answer.pth",map_location=device)            
            unsafe = torch.load(f"asset/HiddenStates/SafetyBench-umk_answer.pth",map_location=device)
            random_numbers = random.sample(range(unsafe.shape[0]), 218)
            unsafe = unsafe[random_numbers]
        elif dataset == "HADES":
            safe = torch.load(f"asset/HiddenStates/mm-vet_answer.pth",map_location=device)            
            unsafe = torch.load(f"asset/HiddenStates/HADES_answer.pth",map_location=device)
            random_numbers = random.sample(range(unsafe.shape[0]), 218)
            unsafe = unsafe[random_numbers]
            
        data = torch.cat([safe,unsafe],dim=0)
        label1 = torch.zeros(safe.shape[0])
        label2 = torch.ones(unsafe.shape[0])
        labels = torch.cat([label1,label2],dim=0).cpu()
   
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
        autoencoder_loaded.load_state_dict(torch.load(f"vicuna/pth/autoencoder-{p0}.pth"))
        autoencoder_loaded.eval()
        #MSCAV
        possibilities = []
        for i in layers: 
            dim = i
            model_path = f'vicuna/pth/{pth}/{i}.pth'
            scores = test_classifier(model_path, data,device=device, selected_dim=dim) 
            possibilities.append(scores)
        
        possibilities = torch.stack(possibilities,dim=1)

        #SAPE
        X_new = possibilities.to(torch.float32)
        with torch.no_grad():
            X_new_hat = autoencoder_loaded(X_new)
            scores = ((X_new_hat - X_new) ** 2).mean(dim=1)                
        
        AUPRC = evaluate_AUPRC(labels, scores)
        AUROC = evaluate_AUROC(labels, scores)            
        print(dataset)
        print(f"AUPRC for: {AUPRC:.4f}")
        print(f"AUROC for: {AUROC:.4f}")
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
        
if __name__ == "__main__":
    set_seed(42)
    main()

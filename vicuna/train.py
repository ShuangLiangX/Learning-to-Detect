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

        scheduler.step(total_loss)  

        if epoch > 0 and abs(total_loss - prev_loss) < 1e-4:
            print("Loss difference is less than 1e-4. Stopping training.")
            break

        prev_loss = total_loss  
    torch.save(model.state_dict(), model_path)
    print(f"Model parameters saved to: {model_path}")

    return 


def test_classifier(model_path, test_data, selected_dim=25,device='cuda:7',input_dim=4096,output_dim=1, dtype=torch.float16):
    model = LinearClassifier(input_dim=input_dim, output_dim=output_dim)
    model.load_state_dict(torch.load(model_path,map_location=device)) 
    model = model.to(device=device,dtype=dtype)
    model.eval()
    
    test_input = test_data[:, selected_dim, :].to(dtype).to(device)

    with torch.no_grad():
        outputs = model(test_input)
    return outputs



def main():
    parser = argparse.ArgumentParser(description="Train and test a linear classifier.")
    parser.add_argument("--train", action="store_true", help="Run training mode.")
    parser.add_argument("--epochs", type=int, default=5000, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for training.")
    parser.add_argument("--test", action="store_true", help="Run testing mode.")
    
    args = parser.parse_args()
    if args.train:
        safety,unsafety = train_data()
        for i in range(33):
            train_classifier(
                safety,
                unsafety,
                f'pth/advbench/{i}.pth',
                i,
                args.epochs,
                args.batch_size,
                args.lr
        )
    elif args.test:        
        device="cuda:7"
        for i in range(33): 
            print(i)
            if  i==0:
                continue
            dim = i
            model_path = f'pth/advbench/{i}.pth'
    
            safety,unsafety = train_data(device=device)
            i=test_classifier(model_path,safety,selected_dim=dim)
            print("safe train score:",(1-torch.sum(i)/i.shape[0]).item())

            i=test_classifier(model_path,unsafety,selected_dim=dim)
            print("unsafe train score:",(torch.sum(i)/i.shape[0]).item())
            #==========advbench-test================
            safety,unsafety = test_data(device=device)
            
            i=test_classifier(model_path,safety,selected_dim=dim)
            print("safe test score:",(1-torch.sum(i)/i.shape[0]).item())
            i=test_classifier(model_path,unsafety,selected_dim=dim)
            print("unsafe test score:",(torch.sum(i)/i.shape[0]).item())
    else:
        print("Please specify either --train or --test mode.")

if __name__ == "__main__":
    main()

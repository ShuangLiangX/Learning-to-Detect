import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
import os
import numpy as np
import random
import json
import sys
sys.path.append("vicuna")
from classifier import LinearClassifier
from utils import train_data,test_data
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
#AE
class SmallAutoencoder(nn.Module):
    def __init__(self, input_dim, bottleneck_dim=4):
        super(SmallAutoencoder, self).__init__()
        
        hidden1 = max(input_dim // 2, bottleneck_dim * 2)
        hidden2 = max(input_dim // 4, bottleneck_dim * 2)

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
    set_seed(42)
   
    device = "cuda:7"
    dtype = torch.float16
    p0=0.9
    layer_chosen = []
    test_pos,test_neg = test_data(device=device)
    #pos 
    test_pos = test_pos.to(dtype)
    layers = [i+1 for i in range(32)]
    classifiers = {}
    scores = []

    for i in layers:
        classifier_path = f"vicuna/pth/advbench/{i}.pth"
            
        classifier = LinearClassifier() 
        classifier.load_state_dict(torch.load(classifier_path, map_location=device))
        classifier.to(device,dtype=dtype)
        classifier.eval()
        classifiers[i] = classifier
        input1 = test_pos[:,i,:].to(device,dtype=dtype)
        input2 = test_neg[:,i,:].to(device,dtype=dtype)
        num1 = input1.shape[0]
        num2 = input2.shape[0]

        with torch.no_grad():
            output1 = classifier(input1)
            output2 = classifier(input2)
        score = (1-torch.sum(output1)/num1+torch.sum(output2)/num2)/2
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

    possibilities_safe = torch.stack(possibilities_safe, dim=1).float().cpu().numpy()
    possibilities_unsafe = torch.stack(possibilities_unsafe, dim=1).float().cpu().numpy()
    
    n_safe_total = possibilities_safe.shape[0]
    n_test_safe = 80
    indices = np.arange(n_safe_total)
    np.random.shuffle(indices)
    test_indices = indices[:n_test_safe]
    train_indices = indices[n_test_safe:]

    X_train_safe = possibilities_safe[train_indices]
    X_test_safe = possibilities_safe[test_indices]

    # ================= train and test dataset =================
    X_train = torch.tensor(X_train_safe, dtype=torch.float32)

    X_test = np.concatenate([X_test_safe, possibilities_unsafe], axis=0)
    y_test = np.concatenate([np.zeros(X_test_safe.shape[0]), np.ones(possibilities_unsafe.shape[0])])
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)

    # ============== training ==============
    X_train = X_train.to(device)
    autoencoder = SmallAutoencoder(input_dim=X_train.shape[1], bottleneck_dim=2)
    autoencoder.to(device)
    autoencoder.train()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=1e-3)

    epochs = 2000
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

    autoencoder.eval()
    with torch.no_grad():
        X_test_hat = autoencoder(X_test)
        reconstruction_errors = ((X_test_hat - X_test)**2).mean(dim=1).cpu().numpy()
    auc = roc_auc_score(y_test, reconstruction_errors)
    print(f"✅ ROC AUC: {auc:.4f}")
    
    torch.save(autoencoder.state_dict(), f"vicuna/pth/autoencoder-{p0}.pth")

if __name__ == "__main__":
    main()
    
    

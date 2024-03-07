import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import re
import os

class CustomDataset(Dataset):
    def __init__(self, features):
        self.features = features

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return self.features[idx]
    
class AE(nn.Module):
    def __init__(self, input_splits):
        super(AE, self).__init__()
        self.input_splits = input_splits
        # encoding
        self.input_layers = nn.ModuleList([
            nn.Linear(part_size, part_size//2) for part_size in input_splits
        ])
        self.encoding_layers = nn.ModuleList([
            nn.Linear(part_size//2, 1) for part_size in input_splits
        ])
        # decoding
        self.decoding_layers = nn.ModuleList([
            nn.Linear(1, part_size//2) for part_size in input_splits
        ])
        self.output_layers = nn.ModuleList([
            nn.Linear(part_size//2, part_size) for part_size in input_splits
        ])
        
    def encoding(self, x):
        encoding_parts = []
        # Starting index for each part
        start_idx = 0
        for i, part_size in enumerate(self.input_splits):
            end_idx = start_idx + part_size
            part = x[:, start_idx:end_idx]
            # input
            input_part_output = F.leaky_relu(self.input_layers[i](part))
            # hidden
            encoding_part_output = self.encoding_layers[i](input_part_output)
            encoding_parts.append(encoding_part_output)
            # Update the start index for the next part
            start_idx = end_idx
        # Concatenate the output of all parts
        features = torch.cat(encoding_parts, dim=1)
        return features
    
    def decoding(self, z):
        output_parts = []
        for i, part_size in enumerate(self.input_splits):
            # hidden
            hidden_part_output = F.leaky_relu(self.decoding_layers[i](z[:,i:i+1]))
            output_part_output = self.output_layers[i](hidden_part_output)
            output_parts.append(output_part_output)
        # Concatenate the output of all parts
        output = torch.cat(output_parts, dim=1)
        return output

    def forward(self, x):
        z = self.encoding(x)
        output = self.decoding(z)
        # Final output
        output = torch.sigmoid(output)
        return output

def ae_load(datatype, disease_mapping, X_train, X_test, code_type):
    # datatype: "binary" or "cont"
    # code_type: "short" or "full"
    
    # load csv files
    # short or full code
    file_paths = ['ae_'+code_type+'/'+datatype+'/'+name+'/' for name in ["train", "test"]]
    # create path is not exist
    for file_path in file_paths:
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

    # only keep demographic covariates at the beginning
    disease_pattern = re.compile(r'^[A-Z0-9]{3}.*')
    demo_columns = [col for col in X_test.columns if not disease_pattern.match(col)]
    X_train_demo = X_train[demo_columns].reset_index(drop=True) 
    X_test_demo = X_test[demo_columns].reset_index(drop=True)
    
    try:
        train_features = pd.read_csv(file_paths[0] + "features.csv")
        test_features = pd.read_csv(file_paths[1] + "features.csv")
        print("AE feature data loaded")
    except:
        print("AE feature data not found, training Neural Network now...")
        train_features, test_features = ae_train(datatype, disease_mapping, file_paths, X_train, X_test)
    
    X_train_ae = pd.concat([train_features, X_train_demo], axis=1)
    X_test_ae = pd.concat([test_features, X_test_demo], axis=1)
    return X_train_ae, X_test_ae

def ae_train(datatype, disease_mapping, file_paths, X_train, X_test, num_epochs=101):
 
    X_train_ae, X_test_ae = pd.DataFrame({}), pd.DataFrame({})
    full_code_lens = []
    for disease, codes in disease_mapping.items():
        regex_pattern = '|'.join(f'^{code}' for code in codes)
        X_train_single = X_train.filter(regex=regex_pattern, axis=1)
        X_test_single = X_test.filter(regex=regex_pattern, axis=1)
        # number of full code in a category
        full_code_lens.append(X_test_single.shape[1])

        X_train_ae = pd.concat(
            [X_train_ae, X_train_single], axis=1)            
        X_test_ae = pd.concat(
            [X_test_ae, X_test_single], axis=1) 

    X_train_ae = X_train_ae.reset_index(drop=True)
    X_test_ae = X_test_ae.reset_index(drop=True)
    
    # Convert DataFrame and numpy array to tensors
    X_train_tensor = torch.from_numpy(X_train_ae.values.astype(np.float32))
    X_test_tensor = torch.from_numpy(X_test_ae.values.astype(np.float32))

    # Create Dataset and DataLoader
    train_dataset = CustomDataset(X_train_tensor)
    test_dataset = CustomDataset(X_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize model, loss function, and optimizer
    model = AE(input_splits=full_code_lens)
    if datatype == "binary":
        criterion = nn.BCELoss()
    elif datatype == "cont":
        criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for inputs in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
        # Test evaluation
        model.eval()
        total = 0
        correct_or_loss = 0
        
        with torch.no_grad():
            for inputs in test_loader:
                outputs = model(inputs)
                if datatype == "binary":
                    outputs = (outputs >= 0.5).float()
                    correct_or_loss += (inputs == outputs).sum().item()
                elif datatype == "cont":
                    test_loss = criterion(outputs, inputs)
                    correct_or_loss += test_loss.item() * (inputs.shape[0] * inputs.shape[1])
                    
                total += inputs.shape[0] * inputs.shape[1]
        test_accuracy_or_loss = correct_or_loss / total
      
        if epoch%5==0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
            if datatype == "binary":
                print(f'Test Accuracy after epoch [{epoch+1}/{num_epochs}]: {test_accuracy_or_loss:.4f}')
            elif datatype == "cont":
                print(f'Test MSE after epoch [{epoch+1}/{num_epochs}]: {test_accuracy_or_loss:.4f}')
            

    print("AE training finished, collecting features...")
    # save features
    train_features = model.encoding(X_train_tensor)
    test_features = model.encoding(X_test_tensor)
    train_features = pd.DataFrame(train_features.detach()).rename(
        columns={i:list(disease_mapping.keys())[i] for i in range(len(disease_mapping))})
    test_features = pd.DataFrame(test_features.detach()).rename(
        columns={i:list(disease_mapping.keys())[i] for i in range(len(disease_mapping))})
    
    # save csv files
    train_csv = file_paths[0] + "features.csv"
    test_csv = file_paths[1] + "features.csv"
    train_features.to_csv(train_csv, index=False)
    print("AE feature training data saved as: " + train_csv)
    test_features.to_csv(test_csv, index=False)
    print("AE feature test data saved as: " + test_csv)
    
    return train_features, test_features
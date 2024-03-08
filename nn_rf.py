import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import re
import os

class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
    
class BinaryClassifier(nn.Module):
    def __init__(self, input_splits, hidden_dim):
        super(BinaryClassifier, self).__init__()
        self.input_splits = input_splits
        self.input_layers = nn.ModuleList([
            nn.Linear(part_size, hidden_dim) for part_size in input_splits
        ])
        self.hidden_layer = nn.Linear(hidden_dim, 1)
        self.hidden_layer_2 = nn.Linear(len(input_splits), len(input_splits))
        self.output_layer = nn.Linear(len(input_splits), 1)

    def forward(self, x, return_features=False):
        hidden_parts = []
        # Starting index for each part
        start_idx = 0
        for i, part_size in enumerate(self.input_splits):
            end_idx = start_idx + part_size
            part = x[:, start_idx:end_idx]
            # input
            input_part_output = F.leaky_relu(self.input_layers[i](part))
            # hidden
            # hidden_part_output = F.leaky_relu(self.hidden_layer(input_part_output))
            hidden_part_output = self.hidden_layer(input_part_output)
            hidden_parts.append(hidden_part_output)
            # Update the start index for the next part
            start_idx = end_idx
        # Concatenate the output of all parts
        features = torch.cat(hidden_parts, dim=1)
        # last hidden layer
        last_hidden_output = F.leaky_relu(self.hidden_layer_2(features))
        # Final output
        output = torch.sigmoid(self.output_layer(last_hidden_output))
        if return_features:
            return output, features
        return output
    
def nn_separate_load(datatype, disease_mapping, X_train, X_test, code_type):
    # datatype: "binary" or "cont"
    # code_type: "short" or "full"
    
    # load csv files
    # short or full code
    file_paths = ['nn_'+code_type+'/'+datatype+'/'+name+'/' for name in ["train", "test"]]
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
        print("Neural Network separation feature data loaded")
    except:
        print("Neural Network separation feature data not found, training Neural Network now...")
        train_features, test_features = nn_separate_train(datatype, disease_mapping, file_paths, X_train, X_test)
    
    X_train_nn = pd.concat([train_features, X_train_demo], axis=1)
    X_test_nn = pd.concat([test_features, X_test_demo], axis=1)
    return X_train_nn, X_test_nn

def nn_separate_train(datatype, disease_mapping, file_paths, X_train, X_test, num_epochs=201):
 
    X_train_nn, X_test_nn = pd.DataFrame({}), pd.DataFrame({})
    full_code_lens = []
    for disease, codes in disease_mapping.items():
        regex_pattern = '|'.join(f'^{code}' for code in codes)
        X_train_single = X_train.filter(regex=regex_pattern, axis=1)
        X_test_single = X_test.filter(regex=regex_pattern, axis=1)
        # number of full code in a category
        full_code_lens.append(X_test_single.shape[1])

        X_train_nn = pd.concat(
            [X_train_nn, X_train_single], axis=1)            
        X_test_nn = pd.concat(
            [X_test_nn, X_test_single], axis=1) 

    X_train_nn, y_train_nn = X_train_nn.reset_index(drop=True), X_train['recur'].values.astype(np.float32)
    X_test_nn, y_test_nn = X_test_nn.reset_index(drop=True), X_test['recur'].values.astype(np.float32)
    
    # Convert DataFrame and numpy array to tensors
    X_train_tensor = torch.from_numpy(X_train_nn.values.astype(np.float32))
    y_train_tensor = torch.from_numpy(y_train_nn.astype(np.float32))
    X_test_tensor = torch.from_numpy(X_test_nn.values.astype(np.float32))
    y_test_tensor = torch.from_numpy(y_test_nn.astype(np.float32))

    # Create Dataset and DataLoader
    train_dataset = CustomDataset(X_train_tensor, y_train_tensor)
    test_dataset = CustomDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize model, loss function, and optimizer
    model = BinaryClassifier(input_splits=full_code_lens, hidden_dim=16)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()

        # Test evaluation
        model.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                predicted = (outputs.squeeze() >= 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_accuracy = correct / total
        if epoch%5==0:
            print(f'Test Accuracy after epoch [{epoch+1}/{num_epochs}]: {test_accuracy:.4f}')

    print("Neural Network separation training finished, collecting features...")
    # save features
    _, train_features = model.forward(X_train_tensor, return_features=True)
    _, test_features = model.forward(X_test_tensor, return_features=True)
    train_features = pd.DataFrame(train_features.detach()).rename(
        columns={i:list(disease_mapping.keys())[i] for i in range(len(disease_mapping))})
    test_features = pd.DataFrame(test_features.detach()).rename(
        columns={i:list(disease_mapping.keys())[i] for i in range(len(disease_mapping))})
    
    # save csv files
    train_csv = file_paths[0] + "features.csv"
    test_csv = file_paths[1] + "features.csv"
    # train_features.to_csv(train_csv, index=False)
    # print("Neural Network feature training data saved as: " + train_csv)
    # test_features.to_csv(test_csv, index=False)
    # print("Neural Network feature test data saved as: " + test_csv)
    
    return train_features, test_features
from tqdm import tqdm
from nn_config import mlp_config, conv_config
import sys
import torch as T
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


class InsuranceDataset(Dataset):
    def __init__(self, filename='normalized_data/train.csv'):
        self.df = pd.read_csv(filename)
        self.df.set_index('id', inplace=True)
        self.transformed_data = self.transform()
        self.input_dims = self.transformed_data.shape[-1] - 1

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        vector = self.transformed_data[idx, :-1].to(self.device)
        val = self.transformed_data[idx, -1].long()
        label = T.zeros(2).to(self.device)
        label[val] = 1
        return vector, label

    def transform(self):
        self.df['Gender'] = np.where(self.df["Gender"] == "Male", 1, 0)
        for idx, cat in enumerate(pd.unique(self.df['Vehicle_Age'])):
            self.df.loc[self.df['Vehicle_Age'] == cat, 'Vehicle_Age'] = idx
        self.df['Vehicle_Damage'] = np.where(self.df["Vehicle_Damage"] == "Yes", 1, 0)
        val = T.tensor(self.df.to_numpy(dtype=np.float32), dtype=T.float32)
        lab = val[:, -1].unsqueeze(dim=-1)
        input_left = val[:, :3]
        center = val[:, 3].long()
        oh = F.one_hot(center)
        input_right = val[:, 4:-1]
        output = T.cat((input_left, oh, input_right, lab), dim=-1)
        return output
        

class BinaryClassifier(nn.Module):
    def __init__(self, input_dims, lr=1e-2, conv=False):
        super(BinaryClassifier, self).__init__()
        self.conv = conv
        self.mlp = nn.Sequential(
            nn.Linear(input_dims, mlp_config['fc1_dims']),
            nn.BatchNorm1d(num_features=mlp_config['fc1_dims']),
            nn.ReLU(),
            nn.Linear(mlp_config['fc1_dims'], mlp_config['fc2_dims']),
            nn.BatchNorm1d(num_features=mlp_config['fc2_dims']),
            nn.ReLU(),
            nn.Linear(mlp_config['fc2_dims'], mlp_config['fc3_dims']),
            nn.BatchNorm1d(num_features=mlp_config['fc3_dims']),
            nn.ReLU(),
            nn.Linear(mlp_config['fc3_dims'], mlp_config['output_dims']),
            nn.BatchNorm1d(num_features=mlp_config['output_dims']),
            nn.Softmax(dim=-1)
        )
        reduced_dims = input_dims - conv_config['kernel_size_1'] - conv_config['kernel_size_2'] - \
                       conv_config['kernel_size_3'] + 3
        self.conv_net = nn.Sequential(
            nn.Conv1d(in_channels=conv_config['in_channels_1'], \
                      out_channels=conv_config['out_channels_1'], \
                      kernel_size=conv_config['kernel_size_1']),
            nn.BatchNorm1d(num_features=conv_config['out_channels_1']),
            nn.ReLU(),
            nn.Conv1d(in_channels=conv_config['in_channels_2'], \
                      out_channels=conv_config['out_channels_2'], \
                      kernel_size=conv_config['kernel_size_2']),
            nn.BatchNorm1d(num_features=conv_config['out_channels_2']),
            nn.ReLU(),
            nn.Dropout(p=conv_config['dropout_probs']),
            nn.Conv1d(in_channels=conv_config['in_channels_3'], \
                      out_channels=conv_config['out_channels_3'], \
                      kernel_size=conv_config['kernel_size_3']),
            nn.BatchNorm1d(num_features=conv_config['out_channels_3']),
            nn.ReLU(),
            nn.Linear(reduced_dims, conv_config['fc1_dims']),
            nn.BatchNorm1d(num_features=conv_config['out_channels_3']),
            nn.ReLU(),
            nn.Dropout(p=conv_config['dropout_probs']),
            nn.Linear(conv_config['fc1_dims'], conv_config['output_dims']),
            nn.BatchNorm1d(num_features=conv_config['out_channels_3']),
            nn.Softmax(dim=-1)
        )

        self.optimizer = optim.Adam(params=self.parameters(), lr=lr, weight_decay=1e-2)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, input):
        if self.conv == True:
            input = input.unsqueeze(1)
            return self.conv_net(input).squeeze()
        return self.mlp(input)


def train(n_epochs=50, verbose=True, save=True):
    train_data = InsuranceDataset()
    trainloader = DataLoader(train_data, batch_size=4096, shuffle=True)
    classifier = BinaryClassifier(input_dims=train_data.input_dims, lr=3e-1, conv=True)
    loss_fn = nn.CrossEntropyLoss(weight=T.tensor([1.3, 8.7]).cuda())
    losses = []
    print('...Training Neural Network...')
    for epoch in tqdm(range(n_epochs)):
        if epoch % 5 == 0 and epoch != 0:
            if save:
                T.save(classifier.state_dict(), 'Trained_Models/classifier')
            test()
        for _, batch in enumerate(trainloader):
            inputs, labels = batch
            outputs = classifier.forward(inputs)
            classifier.optimizer.zero_grad()
            loss = loss_fn(outputs, labels)
            loss.backward()
            classifier.optimizer.step()
            losses.append(loss.item())
            average_loss = np.mean(losses[-2000:])
        if verbose:
            print(f'Loss after {epoch+1} epochs: {average_loss}')
    if save:
        T.save(classifier.state_dict(), 'Trained_Models/classifier')
    print('...Finished Training...')

def test():
    test_data = InsuranceDataset(filename='normalized_data/test.csv')
    testloader = DataLoader(test_data, batch_size=4096, shuffle=True)
    classifier = BinaryClassifier(input_dims=test_data.input_dims, lr=1e-2, conv=True)
    classifier.load_state_dict(T.load('Trained_Models/classifier'))
    classifier.eval()
    correct = 0
    total = 0
    tot_pred = 0
    with T.no_grad():
        print('...Testing Neural Network...')
        for data in testloader:
            inputs, labels = data
            predicted = T.argmax(classifier(inputs), dim=-1)
            tot_pred += (predicted == 1).sum().item()
            labels = T.argmax(labels, dim=-1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        tot_pred /= total

    print(f'Test Accuracy: {100 * correct // total}%')
    print(f'Percentage Ones Predicted: {100 * tot_pred}%')
    return correct / total


if __name__ == '__main__':
    train(25, verbose=False, save=False)
    test()


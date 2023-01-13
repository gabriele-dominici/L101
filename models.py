from utilities import *
import pandas as pd
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, \
    global_max_pool, GATConv, SAGEConv, SGConv, GATv2Conv

class MLP(torch.nn.Module):

    def __init__(self, vocab_mlp, hidden_size, n_classes=1):
        super(MLP, self).__init__()
        self.input_size = len(vocab_mlp)
        self.hidden_size  = hidden_size
        self.n_classes  = n_classes
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.5)
        self.fc2 = torch.nn.Linear(self.hidden_size, n_classes)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        dropout = self.dropout(relu)
        output = self.fc2(dropout)
        output = self.sigmoid(output)
        return output

class GCN2(torch.nn.Module):
    def __init__(self, vocab, n_classes, mode='mean'):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mode = mode
        self.conv1 = GCNConv(vocab, 512)
        self.conv2 = GCNConv(512, 128)
        self.relu = torch.nn.ReLU()
        self.classification = torch.nn.Linear(128, n_classes)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, edge_index, batch=""):
        if batch == "":
            batch = torch.zeros(x.shape[0]).long().to(self.device)
        # GNN layers
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        x = F.dropout(x, training=self.training)
        # Readout
        if self.mode == 'mean':
            x = global_mean_pool(x, batch)
        elif self.mode == 'max':
            x = global_max_pool(x, batch)
        # Classification
        x = self.classification(x)

        return self.sigmoid(x)

class GCN3(torch.nn.Module):
    def __init__(self, vocab, n_classes, mode='mean'):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mode = mode
        self.conv1 = GCNConv(vocab, 512)
        self.conv2 = GCNConv(512, 128)
        self.conv3 = GCNConv(128, 128)
        self.relu = torch.nn.ReLU()
        self.classification = torch.nn.Linear(128, n_classes)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, edge_index, batch=""):
        if batch == "":
            batch = torch.zeros(x.shape[0]).long().to(self.device)
        # GNN layers
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)
        x = self.relu(x)
        x = F.dropout(x, training=self.training)
        # Readout
        if self.mode == 'mean':
            x = global_mean_pool(x, batch)
        elif self.mode == 'max':
            x = global_max_pool(x, batch)
        # Classification
        x = self.classification(x)

        return self.sigmoid(x)

class GAT2(torch.nn.Module):
    def __init__(self, vocab, n_classes, mode='mean'):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mode = mode
        self.gat1 = GATv2Conv(vocab, 512, 1, concat=False)
        self.gat2 = GATv2Conv(512, 128, 1, concat=False)
        self.relu = torch.nn.ReLU()
        self.classification = torch.nn.Linear(128, n_classes)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, edge_index, batch=""):
        if batch == "":
            batch = torch.zeros(x.shape[0]).long().to(self.device)
        # GNN layers
        x = self.gat1(x, edge_index)
        x = self.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.gat2(x, edge_index)
        x = self.relu(x)
        x = F.dropout(x, training=self.training)
        # Readout
        if self.mode == 'mean':
            x = global_mean_pool(x, batch)
        elif self.mode == 'max':
            x = global_max_pool(x, batch)
        x = self.classification(x)

        return self.sigmoid(x)

class GAT3(torch.nn.Module):
    def __init__(self, vocab, n_classes, mode='mean'):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mode = mode
        self.gat1 = GATv2Conv(vocab, 512, 1, concat=False)
        self.gat2 = GATv2Conv(512, 128, 1, concat=False)
        self.gat3 = GATv2Conv(128, 128, 1, concat=False)
        self.relu = torch.nn.ReLU()
        self.classification = torch.nn.Linear(128, n_classes)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, edge_index, batch=""):
        if batch == "":
            batch = torch.zeros(x.shape[0]).long().to(self.device)
        # GNN layers
        x = self.gat1(x, edge_index)
        x = self.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.gat2(x, edge_index)
        x = self.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.gat3(x, edge_index)
        x = self.relu(x)
        x = F.dropout(x, training=self.training)
        # Readout
        if self.mode == 'mean':
            x = global_mean_pool(x, batch)
        elif self.mode == 'max':
            x = global_max_pool(x, batch)
        x = self.classification(x)

        return self.sigmoid(x)

class GraphSAGE2(torch.nn.Module):
    def __init__(self, vocab, n_classes, mode='mean'):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mode = mode
        self.l1 = SAGEConv(vocab, 512)
        self.l2 = SAGEConv(512, 128)
        self.relu = torch.nn.ReLU()
        self.classification = torch.nn.Linear(128, n_classes)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, edge_index, batch=""):
        if batch == "":
            batch = torch.zeros(x.shape[0]).long().to(self.device)
        # GNN layers
        x = self.l1(x, edge_index)
        x = self.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.l2(x, edge_index)
        x = self.relu(x)
        x = F.dropout(x, training=self.training)
        # Readout
        if self.mode == 'mean':
            x = global_mean_pool(x, batch)
        elif self.mode == 'max':
            x = global_max_pool(x, batch)
        # Classification
        x = self.classification(x)

        return self.sigmoid(x)

class GraphSAGE3(torch.nn.Module):
    def __init__(self, vocab, n_classes, mode='mean'):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mode = mode
        self.l1 = SAGEConv(vocab, 512)
        self.l2 = SAGEConv(512, 128)
        self.l3 = SAGEConv(128, 128)
        self.relu = torch.nn.ReLU()
        self.classification = torch.nn.Linear(128, n_classes)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, edge_index, batch=""):
        if batch == "":
            batch = torch.zeros(x.shape[0]).long().to(self.device)
        # GNN layers
        x = self.l1(x, edge_index)
        x = self.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.l2(x, edge_index)
        x = self.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.l3(x, edge_index)
        x = self.relu(x)
        x = F.dropout(x, training=self.training)
        # Readout
        if self.mode == 'mean':
            x = global_mean_pool(x, batch)
        elif self.mode == 'max':
            x = global_max_pool(x, batch)
        # Classification
        x = self.classification(x)

        return self.sigmoid(x)

class SimpleConv2(torch.nn.Module):
    def __init__(self, vocab, n_classes, mode='mean'):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mode = mode
        self.l1 = SGConv(vocab, 128, 2)
        self.relu = torch.nn.ReLU()
        self.classification = torch.nn.Linear(128, n_classes)
        # self.softmax = torch.nn.Softmax(dim=-1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, edge_index, batch=""):
        if batch == "":
            batch = torch.zeros(x.shape[0]).long().to(self.device)
        # GNN layers
        x = self.l1(x, edge_index)
        x = self.relu(x)
        x = F.dropout(x, training=self.training)
        # Readout
        if self.mode == 'mean':
            x = global_mean_pool(x, batch)
        elif self.mode == 'max':
            x = global_max_pool(x, batch)
        # Classification
        x = self.classification(x)

        return self.sigmoid(x)

class SimpleConv3(torch.nn.Module):
    def __init__(self, vocab, n_classes, mode='mean'):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mode = mode
        self.l1 = SGConv(vocab, 128, 3)
        self.relu = torch.nn.ReLU()
        self.classification = torch.nn.Linear(128, n_classes)
        # self.softmax = torch.nn.Softmax(dim=-1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, edge_index, batch=""):
        if batch == "":
            batch = torch.zeros(x.shape[0]).long().to(self.device)
        # GNN layers
        x = self.l1(x, edge_index)
        x = self.relu(x)
        x = F.dropout(x, training=self.training)
        # Readout
        if self.mode == 'mean':
            x = global_mean_pool(x, batch)
        elif self.mode == 'max':
            x = global_max_pool(x, batch)
        # Classification
        x = self.classification(x)

        return self.sigmoid(x)


def train_and_evaluate_mlp(X_train_mlp, Y_train_mlp,
                           X_dev_mlp, Y_dev_mlp,
                           vocab, epoch=20, lr=0.01):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    mlp = MLP(vocab, 512, 1).to(device)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=lr)

    last = 100000
    counter = 0

    for epoch in range(epoch):
        mlp.train()
        optimizer.zero_grad()
        y_pred = mlp(X_train_mlp.to(device))
        loss = criterion(y_pred.squeeze(), Y_train_mlp.to(device))
        loss.backward()
        optimizer.step()
        print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
        with torch.no_grad():
            y_pred_dev = mlp(X_dev_mlp.to(device))
            loss_dev = criterion(y_pred_dev.squeeze(), Y_dev_mlp.to(device))

            if last >= loss_dev:
                counter = 0
                last = loss_dev
            else:
                counter += 1

            print('Epoch {}: dev loss: {}'.format(epoch, loss_dev.item()))
            report_results_mlp(Y_dev_mlp.long().to(device), y_pred_dev.squeeze(), "Dev", 1)
        if counter >= 3:
            break

    return mlp

def test_mlp(mlp, X_test_mlp, Y_test_mlp, dataset='news'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mlp.eval()
    criterion = torch.nn.BCELoss()
    y_pred = mlp(X_test_mlp.to(device))
    after_train = criterion(y_pred.squeeze(), Y_test_mlp.to(device))
    print('Test loss after Training', after_train.item())
    report_results_mlp(Y_test_mlp.long().to(device), y_pred.squeeze().to(device), "Test", 1)
    df = pd.DataFrame([round(i) for i in y_pred.squeeze().tolist()])
    df.to_csv(f'./results/mlp_{dataset}.csv', index=False)

def train_and_evaluate_graph(train_loader, dev_loader, model_k, vocab, epochs=20, lr=0.01):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    model = model_k(len(vocab), 1).to(device)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    last = 10000
    counter = 0
    model.train()

    for epoch in range(epochs):
        loss_train = 0
        step = 0
        for data in train_loader:
            optimizer.zero_grad()
            out = model(data.x.to(device), data.edge_index.to(device), data.batch.to(device))
            gold = data.y.to(device)
            loss = criterion(out.squeeze(), gold)
            loss.backward()
            optimizer.step()
            del data
            gc.collect()

            loss_train += loss
            step += 1

        loss_train = loss_train / step
        print('Epoch {}: train loss: {}'.format(epoch, loss_train.item()))

        total_loss_dev = 0
        step = 0
        for data in dev_loader:
            out = model(data.x.to(device), data.edge_index.to(device), data.batch.to(device))
            gold = data.y.to(device)
            loss_dev = criterion(out.squeeze(), gold)
            if total_loss_dev == 0:
                out_total = out.squeeze()
                gold_total = gold
            else:
                out_total = torch.cat((out_total, out.squeeze()), 0)
                gold_total = torch.cat((gold_total, gold), 0)
            del data
            gc.collect()
            total_loss_dev += loss_dev
            step += 1

        total_loss_dev = total_loss_dev / step
        if last >= total_loss_dev:
            counter = 0
            last = total_loss_dev
        else:
            counter += 1

        print('Epoch {}: dev loss: {}'.format(epoch, total_loss_dev.item()))
        report_results_mlp(gold_total.long().to(device), out_total.to(device), "Dev", 2)

        if counter >= 3:
            break

    return model

def test_graph(model, test_loader, model_k, readout, dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    total_loss_test = 0
    criterion = torch.nn.BCELoss()
    for data in test_loader:
        step = 0
        out = model(data.x.to(device), data.edge_index.to(device), data.batch.to(device))
        gold = data.y.to(device)
        loss_test = criterion(out.squeeze(), gold)
        if total_loss_test == 0:
            out_total = out.squeeze()
            gold_total = gold
        else:
            out_total = torch.cat((out_total, out.squeeze()), 0)
            gold_total = torch.cat((gold_total, gold), 0)
        del data
        gc.collect()
        total_loss_test += loss_test
        step += 1

    total_loss_test = total_loss_test / step
    print('Test loss: {}'.format(total_loss_test.item()))
    report_results_mlp(gold_total.long().to(device), out_total.to(device), "Test", 2)

    df = pd.DataFrame([round(i) for i in out_total.squeeze().tolist()])
    df.to_csv(f'./results/{model_k}_{readout}_{dataset}.csv', index=False)
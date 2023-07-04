import os 
from data.datasets import *
from model.model import *

os.environ["DGLBACKEND"] = "pytorch"

class GNN_trainer_node_classification():
    '''
    node classification trainer
    '''
    def __init__(self):
        self.num_classes = Cora_dataset().num_classes
        self.dataset = Cora_dataset()[0].to('cuda') # pick the first graph in the dataset
        self.model = GCN_node_classification(self.dataset.ndata["feat"].shape[1], 16, self.num_classes).to('cuda')
        
    def train(self):
        g = self.dataset
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        best_val_acc = 0
        best_test_acc = 0

        features = g.ndata["feat"]
        labels = g.ndata["label"]
        train_mask = g.ndata["train_mask"]
        val_mask = g.ndata["val_mask"]
        test_mask = g.ndata["test_mask"]
        for e in range(1000):
            # Forward
            logits = self.model(g, features)

            # Compute prediction
            pred = logits.argmax(1)

            # Compute loss
            # Note that you should only compute the losses of the nodes in the training set.
            loss = F.cross_entropy(logits[train_mask], labels[train_mask])

            # Compute accuracy on training/validation/test
            train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
            val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
            test_acc = (pred[test_mask] == labels[test_mask]).float().mean()

            # Save the best validation accuracy and the corresponding test accuracy.
            if best_val_acc < val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if e % 5 == 0:
                print(
                    f"In epoch {e}, loss: {loss:.3f}, val acc: {val_acc:.3f} (best {best_val_acc:.3f}), test acc: {test_acc:.3f} (best {best_test_acc:.3f})"
                )
                
class GNN_trainer_graph_classification():
    '''
    graph classification trainer
    '''
    def __init__(self):
        
        self.num_classes = Protine_dataset().num_classes
        self.dataset = Protine_dataset()
        # make data loader
        self.train_dataloader, self.test_dataloader = make_data_loader(self.dataset, split_rate = 0.8)
        
        self.model = GCN_graph_classification(self.dataset.dim_nfeats, 16, self.dataset.gclasses).to('cuda')
        
    def train(self):
        
        # Create the model with given dimensions
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

        for epoch in range(100):
            epoch_loss = 0
            for iter, (batched_graph, labels) in enumerate(self.train_dataloader):
                pred = self.model(batched_graph.to('cuda'), batched_graph.ndata["attr"].float().to('cuda'))
                loss = F.cross_entropy(pred, labels.to('cuda'))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.detach().item()
            epoch_loss /= (iter + 1)
            print(
                    f"In epoch {epoch}, loss: {loss:.3f}")
        num_correct = 0
        num_tests = 0
        for batched_graph, labels in self.test_dataloader:
            pred = self.model(batched_graph.to('cuda'), batched_graph.ndata["attr"].float().to('cuda'))
            num_correct += (pred.argmax(1) == labels.to('cuda')).sum().item()
            num_tests += len(labels)

        print("Test accuracy:", num_correct / num_tests)

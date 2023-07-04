import dgl
import torch
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler

def make_data_loader(dataset, split_rate = 0.8):
    num_examples = len(dataset)
    num_train = int(num_examples * split_rate)

    train_sampler = SubsetRandomSampler(torch.arange(num_train))
    test_sampler = SubsetRandomSampler(torch.arange(num_train, num_examples))

    train_dataloader = GraphDataLoader(
        dataset, sampler=train_sampler, batch_size=5, drop_last=False
    )
    test_dataloader = GraphDataLoader(
        dataset, sampler=test_sampler, batch_size=5, drop_last=False
    )
    return train_dataloader, test_dataloader

def Cora_dataset():
    return dgl.data.CoraGraphDataset()

def Protine_dataset(split_rate = 0.8):
    return dgl.data.GINDataset("PROTEINS", self_loop=True)
    
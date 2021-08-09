import numpy as np
import gpytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import backbone
from torch.autograd import Variable
from data.qmul_loader import get_batch, train_people, test_people

class Regressor(nn.Module):
    def __init__(self):
        super(Regressor, self).__init__()
        self.layer4 = nn.Linear(2916, 1)

    def return_clones(self):
        layer4_w = self.layer4.weight.data.clone().detach()
        layer4_b = self.layer4.bias.data.clone().detach()

    def assign_clones(self, weights_list):
        self.layer4.weight.data.copy_(weights_list[0])
        self.layer4.weight.data.copy_(weights_list[1])

    def forward(self, x):
        out = self.layer4(x)
        return out

class FeatureTransfer(nn.Module):
    def __init__(self, backbone):
        super(FeatureTransfer, self).__init__()
        regressor = Regressor()
        self.feature_extractor = backbone
        self.model = Regressor()
        self.criterion = nn.MSELoss()

    def train_loop(self, epoch, n_samples, optimizer):
        batch, batch_labels = get_batch(train_people, n_samples)
        batch, batch_labels = batch.cuda(), batch_labels.cuda()

        for inputs, labels in zip(batch, batch_labels):
            optimizer.zero_grad()
            output = self.model(self.feature_extractor(inputs))
            loss = self.criterion(output.squeeze(), labels)
            loss.backward()
            optimizer.step()

            if(epoch%2==0):
                print('[%02d] - Loss: %.3f' % (
                    epoch, loss.item()
                ))
   
    def train(self, epoch, n_support, n_samples, optimizer):

        self.train_loop(epoch, n_samples, optimizer)

    def test_loop(self, n_support, n_samples, test_person, optimizer): # we need optimizer to take one gradient step
        inputs, targets = get_batch(test_people, n_samples)

        # support_ind = list(np.random.choice(list(range(19)), replace=False, size=n_support))
        # query_ind   = [i for i in range(19) if i not in support_ind]

        x_all = inputs.cuda()
        y_all = targets.cuda()

        split = np.array([True]*15 + [False]*3)
        # print(split)
        shuffled_split = []
        for _ in range(6):
            s = split.copy()
            np.random.shuffle(s)
            shuffled_split.extend(s)
        shuffled_split = np.array(shuffled_split)
        support_ind = shuffled_split
        query_ind = ~shuffled_split
        x_support = x_all[test_person, support_ind,:,:,:]
        y_support = y_all[test_person, support_ind]
        x_query   = x_all[test_person, query_ind,:,:,:]
        y_query   = y_all[test_person, query_ind]

        optimizer.zero_grad()
        z_support = self.feature_extractor(x_support).detach()
        output_support = self.model(z_support).squeeze()
        loss = self.criterion(output_support.squeeze(), y_support)
        loss.backward()
        optimizer.step()

        self.feature_extractor.eval()
        self.model.eval()
        z_query = self.feature_extractor(x_query).detach()
        output = self.model(z_query).squeeze()
        return self.criterion(output.squeeze(), y_query).item()

    def test(self, n_support, n_samples, optimizer, test_count=None):

        mse_list = []
        # choose a random test person
        test_person = np.random.choice(np.arange(len(test_people)), size=test_count, replace=False)
        for t in range(test_count):
            print(f'test #{t}')
            
            mse = self.test_loop(n_support, n_samples, test_person[t],  optimizer)
            
            mse_list.append(float(mse))
        return mse_list

    def save_checkpoint(self, checkpoint):
        torch.save({'feature_extractor': self.feature_extractor.state_dict(), 'model':self.model.state_dict()}, checkpoint)

    def load_checkpoint(self, checkpoint):
        ckpt = torch.load(checkpoint)
        self.feature_extractor.load_state_dict(ckpt['feature_extractor'])
        self.model.load_state_dict(ckpt['model'])

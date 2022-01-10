import numpy as np
import gpytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import random
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class Sine_Task():
    """
    A sine wave data distribution object with interfaces designed for MAML.
    """
    def __init__(self, amplitude, phase, xmin, xmax):
        self.amplitude = amplitude
        self.phase = phase
        self.xmin = xmin
        self.xmax = xmax

    def true_function(self, x):
        """
        Compute the true function on the given x.
        """
        return self.amplitude * np.sin(self.phase + x)

    def sample_data(self, size=1, noise=0.0, sort=False):
        """
        Sample data from this task.

        returns:
            x: the feature vector of length size
            y: the target vector of length size
        """
        x = np.random.uniform(self.xmin, self.xmax, size)
        if(sort): x = np.sort(x)
        y = self.true_function(x)
        if(noise>0): y += np.random.normal(loc=0.0, scale=noise, size=y.shape)
        x = torch.tensor(x, dtype=torch.float).unsqueeze(1)
        y = torch.tensor(y, dtype=torch.float)
        return x, y
        
        
class Cosine_Task():
    """
    A sine wave data distribution object with interfaces designed for MAML.
    """
    def __init__(self, amplitude, phase, xmin, xmax):
        self.amplitude = amplitude
        self.phase = phase
        self.xmin = xmin
        self.xmax = xmax

    def true_function(self, x):
        """
        Compute the true function on the given x.
        """
        return self.amplitude * np.cos(self.phase + x)

    def sample_data(self, size=1, noise=0.0, sort=False):
        """
        Sample data from this task.

        returns:
            x: the feature vector of length size
            y: the target vector of length size
        """
        x = np.random.uniform(self.xmin, self.xmax, size)
        if(sort): x = np.sort(x)
        y = self.true_function(x)
        if(noise>0): y += np.random.normal(loc=0.0, scale=noise, size=y.shape)
        x = torch.tensor(x, dtype=torch.float).unsqueeze(1)
        y = torch.tensor(y, dtype=torch.float)
        return x, y

class Task_Distribution():
    """
    The task distribution for sine regression tasks for MAML
    """

    def __init__(self, amplitude_min, amplitude_max, phase_min, phase_max, x_min, x_max, family="sine"):
        self.amplitude_min = amplitude_min
        self.amplitude_max = amplitude_max
        self.phase_min = phase_min
        self.phase_max = phase_max
        self.x_min = x_min
        self.x_max = x_max
        self.family = family

    def sample_task(self):
        """
        Sample from the task distribution.

        returns:
            Sine_Task object
        """
        amplitude = np.random.uniform(self.amplitude_min, self.amplitude_max)
        phase = np.random.uniform(self.phase_min, self.phase_max)
        if(self.family=="sine"):
            return Sine_Task(amplitude, phase, self.x_min, self.x_max)
        elif(self.family=="cosine"):
            return Cosine_Task(amplitude, phase, self.x_min, self.x_max)
        else:
            return None

class Feature(nn.Module):
    def __init__(self):
        super(Feature, self).__init__()
        self.layer1 = nn.Linear(1, 40)
        self.layer2 = nn.Linear(40,40)
        #self.layer3 = nn.Linear(40,1)
        
    def forward(self, x):
        out = F.relu(self.layer1(x))
        out = F.relu(self.layer2(out))
        #out = self.layer3(out)
        return out

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        #self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5))
        # self.covar_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=4, ard_num_dims=40)
        #self.feature_extractor = feature_extractor
        
    def forward(self, x):
        #z = self.feature_extractor(x)
        #z_normalized = z - z.min(0)[0]
        #z_normalized = 2 * (z_normalized / z_normalized.max(0)[0]) - 1
        # x_normalized = x - x.min(0)[0]
        # x_normalized = 2 * (x_normalized / x_normalized.max(0)[0]) - 1
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def save_checkpoint(checkpoint, gp, likelihood, feature_extractor):
        # save state
        gp_state_dict         = gp.state_dict()
        likelihood_state_dict = likelihood.state_dict()
        nn_state_dict         = feature_extractor.state_dict()
        torch.save({'gp': gp_state_dict, 'likelihood': likelihood_state_dict, 'net':nn_state_dict}, checkpoint)

def load_checkpoint(gp, likelihood, feature_extractor, checkpoint):
    
    ckpt = torch.load(checkpoint)
    if 'best' in checkpoint:
        print(f'\nBest model at epoch {ckpt["epoch"]}, MSE: {ckpt["mse"]}')
    
    gp.load_state_dict(ckpt['gp'])
    likelihood.load_state_dict(ckpt['likelihood'])
    feature_extractor.load_state_dict(ckpt['net'])

    return gp, likelihood, feature_extractor

def normalize(x, x_min, x_max):
    x = (x - x_min) / x_max
    return x
def denormalize(x, x_min, x_max):
    x = x * x_max + x_min
    return x
seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False 
def main():
    ## Defining model
    device = 'cpu'
    do_train = True

    family = 'sine'
    n_shot_train = 200
    n_shot_test = 180 #n_support for test time 
    t = 150.0
    train_range=(-t, t)
    test_range=(-t, t) # This must be (-5, +10) for the out-of-range condition
    y_range = (-6, 6)
    sample_size = n_shot_train
    checkpoint = './sines/save_model'
    if not os.path.isdir(checkpoint):
        os.makedirs(checkpoint)
    criterion = nn.MSELoss()
    tasks_loader     = [Task_Distribution(amplitude_min=0.1, amplitude_max=5.0, 
                                  phase_min=0.0, phase_max=np.pi, 
                                  x_min=train_range[0], x_max=train_range[1], 
                                  family=family) for _ in range(100)]
    net       = Feature().to(device)
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    dummy_inputs = torch.zeros([n_shot_train,40]).to(device)
    dummy_labels = torch.zeros([n_shot_train]).to(device)
    gp = ExactGPModel(dummy_inputs, dummy_labels, likelihood).to(device)
    gp.likelihood.noise = 0.1
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp).to(device)
    optimizer = torch.optim.Adam([{'params': gp.parameters(), 'lr': 1e-3},
                                  {'params': net.parameters(), 'lr': 1e-3}])

    ## Training
    likelihood.train()
    gp.train()
    net.train()

    tot_epoch=100 #50000
    if do_train:
        for epoch in range(tot_epoch):
            for itr, task in enumerate(tasks_loader):
                optimizer.zero_grad()
                xx, yy = task.sample_task().sample_data(n_shot_train, noise=0.1)
                # x = inputs.to(device)
                # y = labels.to(device)
                
                xx = normalize(xx, xx.min(), xx.max())
                yy = normalize(yy, yy.min(), yy.max())
                # x = x - xx.min(0)[0]
                # x = 2 * (x / xx.max(0)[0]) - 1
                # y = y - yy.min(0)[0]
                # y = 2 * (y / yy.max(0)[0]) - 1
                z = net(xx)
                gp.set_train_data(inputs=z, targets=yy, strict=False)  
                predictions = gp(z)
                loss = -mll(predictions, gp.train_targets)
                loss.backward()
                optimizer.step()
                mse = criterion(predictions.mean, yy)
                #---- print some stuff ----
                if(itr%20==0):
                    print('[%d/%d] - Loss: %.3f  MSE: %.3f  lengthscale: %.3f   noise: %.3f' % (
                        itr, epoch, loss.item(), mse.item(),
                        gp.covar_module.base_kernel.lengthscale.item(),
                        gp.likelihood.noise.item()
                    ))
            
            
        save_checkpoint(f'{checkpoint}/DKT.pth', gp, likelihood, net)
    
    
    ## Test phase on a new sine/cosine wave
    checkpoint = f'{checkpoint}/DKT.pth'
    gp, likelihood, net = load_checkpoint(gp, likelihood, net, checkpoint)
    tasks_test_loader = [Task_Distribution(amplitude_min=0.1, amplitude_max=5.0, 
                                  phase_min=0.0, phase_max=np.pi, 
                                  x_min=train_range[0], x_max=train_range[1], 
                                  family=family) for _ in range(500)]
    print("Test, please wait...")

    likelihood.eval()    
    net.eval()
    gp.eval()
    tot_iterations=100
    mse_list = list()
    for epoch, task_test in enumerate(tasks_test_loader):
        sample_task = task_test.sample_task()
        
        xx, yy = sample_task.sample_data(sample_size, noise=0.1, sort=True)
        xx = normalize(xx, xx.min(), xx.max())
        yy = normalize(yy, yy.min(), yy.max())
        indices = np.arange(sample_size)
        np.random.shuffle(indices)
        support_indices = np.sort(indices[0:n_shot_test])

        query_indices = np.sort(indices[n_shot_test:])
        x_support = xx[support_indices]
        y_support = yy[support_indices]
        x_query = xx[query_indices]
        y_query = yy[query_indices]

        #Feed the support set
        z_support = net(x_support).detach()
        gp.train()
        gp.set_train_data(inputs=z_support, targets=y_support, strict=False)  
        gp.eval()

        #Evaluation on query set
        z_query = net(x_query).detach()
        mean = likelihood(gp(z_query)).mean

        mse = criterion(mean, y_query)
        mse_list.append(mse.item())

    print("-------------------")
    print("Average MSE: " + str(np.mean(mse_list)) + " +- " + str(np.std(mse_list)))
    print("-------------------")
    tasks_test_loader = [Task_Distribution(amplitude_min=0.1, amplitude_max=5.0, 
                                  phase_min=0.0, phase_max=np.pi, 
                                  x_min=train_range[0], x_max=train_range[1], 
                                  family=family) for _ in range(10)]
    for i, task_test in enumerate(tasks_test_loader):
        sample_task = task_test.sample_task()
        x_all, y_all = sample_task.sample_data(sample_size, noise=0.1, sort=True)
        # x_min, x_max =  x_all.min(), x_all.max()
        # y_min, y_max =  y_all.min(), y_all.max()
        # xx = normalize(x_all, x_min, x_max)
        # yy = normalize(y_all, y_min, y_max)
        query_indices = np.sort(indices[n_shot_test:])
        x_support = x_all[support_indices]
        y_support = y_all[support_indices]
        x_query = x_all[query_indices]
        y_query = y_all[query_indices]
    
        z_support = net(x_support).detach()
        gp.train()
        gp.set_train_data(inputs=z_support, targets=y_support, strict=False)  
        gp.eval()
            
        #Evaluation on all data
        z_all = net(x_all).detach()
        mean = likelihood(gp(z_all)).mean
        lower, upper = likelihood(gp(z_all)).confidence_region() #2 standard deviations above and below the mean
        # mean = denormalize(mean, y_min, y_max)
        # lower = denormalize(lower, y_min, y_max)
        # upper = denormalize(upper, y_min, y_max)
        #Plot
        fig, ax = plt.subplots(figsize=(16, 2))
        #true-curve
        true_curve = np.linspace(train_range[0], train_range[1], 1000)
        true_curve = [sample_task.true_function(x) for x in true_curve]
        ax.plot(np.linspace(train_range[0], train_range[1], 1000), true_curve, color='blue', linewidth=2.0)
        if(train_range[1]<test_range[1]):
            dotted_curve = np.linspace(train_range[1], test_range[1], 1000)
            dotted_curve = [sample_task.true_function(x) for x in dotted_curve]
            ax.plot(np.linspace(train_range[1], test_range[1], 1000), dotted_curve, color='blue', linestyle="--", linewidth=2.0)
        #query points (ground-truth)
        #ax.scatter(x_query, y_query, color='blue')
        #query points (predicted)

        ax.plot(np.squeeze(x_all), mean.detach().numpy(), color='red', linewidth=2.0)
        ax.fill_between(np.squeeze(x_all),
                        lower.detach().numpy(), upper.detach().numpy(),
                        alpha=.1, color='red')
        #support points
        x_support = x_all[support_indices]
        y_support = y_all[support_indices]
        ax.scatter(x_support, y_support, color='darkblue', marker='*', s=50, zorder=10)
                    
        #all points
        #ax.scatter(x_all.numpy(), y_all.numpy())
        #plt.show()
        plt.tight_layout()
        plt.ylim(y_range[0], y_range[1])
        plt.xlim(test_range[0], test_range[1])
        plt.savefig('./sines/image/plot_DKT_' + str(i) + '.png', dpi=300)


if __name__ == "__main__":
    main()

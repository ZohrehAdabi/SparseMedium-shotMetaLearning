import numpy as np
import gpytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from collections import namedtuple
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from colorama import Fore
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# import methods
from methods.Fast_RVM_regression import Fast_RVM_regression 

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
    def __init__(self, train_x, train_y, inducing_point, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.base_covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        #self.base_covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5))
        # self.base_covar_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=4, ard_num_dims=40)
        #self.feature_extractor = feature_extractor
        self.covar_module = gpytorch.kernels.InducingPointKernel(self.base_covar_module, inducing_points=inducing_point , likelihood=likelihood)
    def forward(self, x):
        #z = self.feature_extractor(x)
        #z_normalized = z - z.min(0)[0]
        #z_normalized = 2 * (z_normalized / z_normalized.max(0)[0]) - 1
        #x_normalized = x - x.min(0)[0]
        #x_normalized = 2 * (x_normalized / x_normalized.max(0)[0]) - 1
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

IP = namedtuple("inducing_points", "z_values index count alpha gamma x y i_idx j_idx")
def get_inducing_points(gp, likelihood, inputs, targets,  config='1011', align_threshold=1e-3, verbose=True, device='cpu'):

        criterion = nn.MSELoss()
        IP_index = np.array([])

        if True:
            # with sigma and updating sigma converges to more sparse solution
            N   = inputs.shape[0]
            tol = 1e-6
            eps = torch.finfo(torch.float32).eps
            max_itr = 1000
            sigma = likelihood.noise[0].clone()
            # sigma = torch.tensor([0.0000001])
            # sigma = torch.tensor([torch.var(targets) * 0.1]) #sigma^2
            sigma = sigma.to(device)
            beta = 1 /(sigma + eps)
            covar_module = gp.base_covar_module
            kernel_matrix = covar_module(inputs).evaluate()
            # normalize kernel
            scales = torch.ones(kernel_matrix.shape[0]).to(device)
            if True:
                scales	= torch.sqrt(torch.sum(kernel_matrix**2, axis=0))
                # print(f'scale: {Scales}')
                scales[scales==0] = 1
                kernel_matrix = kernel_matrix / scales

            kernel_matrix = kernel_matrix.to(torch.float64)
            target = targets.clone().to(torch.float64)
            active, alpha, gamma, beta, mu_m, U = Fast_RVM_regression(kernel_matrix, target, beta, N, config, align_threshold,
                                                    False, eps, tol, max_itr, device, verbose)
            
            # index = np.argsort(active)
            # active = active[index]
            # gamma = gamma[index].to(torch.float)
            ss = scales[active]
            # alpha = alpha[index] #/ ss**2
            # mu_m = mu_m[index] #/ ss
            inducing_points = inputs[active]
            
            num_IP = active.shape[0]
            IP_index = active
            with torch.no_grad():
                if True:
                    
                    K = covar_module(inputs, inducing_points).evaluate()
                    # K = covar_module(X, X[active]).evaluate()
                    
                    mu_r = mu_m.to(torch.float) /ss
                    y_pred_r = K @ mu_r
                    
                    mse_r = criterion(y_pred_r, target)
                    print(f'FRVM MSE: {mse_r:0.4f}')
            

        return IP(inducing_points, IP_index, num_IP, alpha.to(torch.float64), gamma, None, None, None, None), beta, mu_m.to(torch.float64), U
 
def rvm_ML(K_m, targets, alpha_m, mu_m, U, beta):
        
        N = targets.shape[0]
        # targets = targets.to(torch.float64)
        # K_mt = targets @ K_m
        # A_m = torch.diag(alpha_m)
        # H = A_m + beta * K_m.T @ K_m
        # U, info =  torch.linalg.cholesky_ex(H, upper=True)
        # # if info>0:
        # #     print('pd_err of Hessian')
        # U_inv = torch.linalg.inv(U)
        # Sigma_m = U_inv @ U_inv.T      
        # mu_m = beta * (Sigma_m @ K_mt)
        y_ = K_m @ mu_m  
        e = (targets - y_)
        ED = e.T @ e
        # DiagC	= torch.sum(U_inv**2, axis=1)
        # Gamma	= 1 - alpha_m * DiagC
        # beta	= (N - torch.sum(Gamma))/ED
        # dataLikely	= (N * torch.log(beta) - beta * ED)/2
        # logdetHOver2	= torch.sum(torch.log(torch.diag(U)))
        
        # 2001-JMLR-SparseBayesianLearningandtheRelevanceVectorMachine in Appendix:
        # C = sigma * I + K_m @ A_m @ K_m.T  ,  log|C| = - log|Sigma_m| - N * log(beta) - log|A_m|
        # t.T @ C^-1 @ t = beta * ||t - K_m @ mu_m||**2 + mu_m.T @ A_m @ mu_m 
        # log p(t) = -1/2 (log|C| + t.T @ C^-1 @ t ) + const 
        logML = -1/2 * (beta * ED)  #+ (mu_m**2) @ alpha_m  #+ N * torch.log(beta) + 2*logdetHOver2
        # logML			= dataLikely - (mu_m**2) @ alpha_m /2 + torch.sum(torch.log(alpha_m))/2 - logdetHOver2
        # logML = -1/2 * beta * ED
    
        # NOTE new loss for rvm
        # S = torch.ones(N).to(self.device) *1/beta
        # K_star_Sigma = torch.diag(K_star_m @ Sigma_m @ K_star_m.T)
        # Sigma_star = torch.diag(S) + torch.diag(K_star_Sigma)
        # K_star_Sigma = K_star_m @ Sigma_m @ K_star_m.T
        # Sigma_star = torch.diag(S) + K_star_Sigma

        # new_loss =-1/2 *((e) @ torch.linalg.inv(Sigma_star) @ (e) + torch.log(torch.linalg.det(Sigma_star)+1e-10))

        # return logML/N
        return logML/N, ED/N

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
    IP = torch.ones(gp.covar_module.inducing_points.shape[0], 40).cuda()
    ckpt['gp']['covar_module.inducing_points'] = IP
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
def main():
    ## Defining model
    device = 'cpu'
    do_train = False
    n_shot_train = 100
    n_shot_test = 80
    t = 20.0
    train_range=(-t, t)
    test_range=(-t, t) # This must be (-5, +10) for the out-of-range condition
    y_range = (-6, 6)
    sample_size = 100 
    checkpoint = './sines/save_model'
    if not os.path.isdir(checkpoint):
        os.makedirs(checkpoint)
    criterion = nn.MSELoss()
    tasks     = Task_Distribution(amplitude_min=0.1, amplitude_max=5.0, 
                                  phase_min=0.0, phase_max=np.pi, 
                                  x_min=train_range[0], x_max=train_range[1], 
                                  family="sine")
    net       = Feature()
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    dummy_inputs = torch.zeros([n_shot_train,40])
    dummy_labels = torch.zeros([n_shot_train])
    gp = ExactGPModel(dummy_inputs, dummy_labels, dummy_inputs, likelihood)
    mll_loss = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp)
    optimizer = torch.optim.Adam([{'params': gp.parameters(), 'lr': 1e-3},
                                  {'params': net.parameters(), 'lr': 1e-3}])

    ## Training
    if do_train:
        likelihood.train()
        gp.train()
        net.train()

        tot_iterations=5000 #50000
        l = 100
        for epoch in range(tot_iterations):
            optimizer.zero_grad()
            inputs, labels = tasks.sample_task().sample_data(n_shot_train, noise=0.1)
            xx = normalize(inputs, inputs.min(), inputs.max())
            yy = normalize(labels, labels.min(), labels.max())
            z = net(xx)
            with torch.no_grad():
                inducing_points, beta, mu_m, U = get_inducing_points(gp, likelihood, z, yy, verbose=False, device=device)
            
            ip_index = inducing_points.index
            ip_values = z[ip_index]
            gp.covar_module.inducing_points = nn.Parameter(ip_values, requires_grad=False)

            alpha_m = inducing_points.alpha
            K_m = gp.base_covar_module(z, ip_values).evaluate()
            K_m = K_m.to(torch.float64)
            scales	= torch.sqrt(torch.sum(K_m**2, axis=0))
            K_m = K_m / scales
            # alpha_m = alpha_m / (scales**2)
            # mu_m = mu_m / scales
            rvm_mll, rvm_mse = rvm_ML(K_m, yy, alpha_m, mu_m, U, beta)

            gp.set_train_data(inputs=z, targets=yy, strict=False)  
            predictions = gp(z)
            mll = mll_loss(predictions, gp.train_targets)

            loss = -mll + l * rvm_mse
            # loss = -mll - l * rvm_mll

            loss.backward()
            optimizer.step()
            mse = criterion(predictions.mean, labels)
            #---- print some stuff ----
            if(epoch%100==0):
                print(Fore.LIGHTRED_EX, '[%d] - Loss: %.3f  MSE: %.3f MLL: %.3f  RVM MLL: %.3f  lengthscale: %.3f   noise: %.3f' % (
                    epoch, loss.item(), mse.item(), -mll.item(), -rvm_mll.item(),
                    gp.base_covar_module.base_kernel.lengthscale.item(),
                    gp.likelihood.noise.item()
                ), Fore.RESET)

        
        save_checkpoint(f'{checkpoint}/Sparse_DKT.pth', gp, likelihood, net)
    
    
    ## Test phase on a new sine/cosine wave
    model_path = f'{checkpoint}/Sparse_DKT.pth'
    gp, likelihood, net = load_checkpoint(gp, likelihood, net, model_path)
    tasks_test = Task_Distribution(amplitude_min=0.1, amplitude_max=5.0, 
                                   phase_min=0.0, phase_max=np.pi, 
                                   x_min=test_range[0], x_max=test_range[1], 
                                   family="sine")
    print("Test, please wait...")

    likelihood.eval()    
    net.eval()
    tot_iterations=500
    mse_list = list()
    for epoch in range(tot_iterations):
        sample_task = tasks_test.sample_task()
        
        x_all, y_all = sample_task.sample_data(sample_size, noise=0.1, sort=True)
        xx = normalize(x_all, x_all.min(), x_all.max())
        yy = normalize(y_all, y_all.min(), y_all.max())
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
        with torch.no_grad():
            inducing_points, beta, mu_m, U = get_inducing_points(gp, likelihood, z_support, y_support, verbose=False, device=device)
            
        ip_index = inducing_points.index
        ip_values = z_support[ip_index]
        gp.covar_module.inducing_points = nn.Parameter(ip_values, requires_grad=False)
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

    for i in range(10):
        x_all, y_all = sample_task.sample_data(sample_size, noise=0.1, sort=True)
        x_min, x_max =  x_all.min(), x_all.max()
        y_min, y_max =  y_all.min(), y_all.max()
        xx = normalize(x_all, x_min, x_max)
        yy = normalize(y_all, y_min, y_max)
        query_indices = np.sort(indices[n_shot_test:])
        x_support = xx[support_indices]
        y_support = yy[support_indices]
        x_query = xx[query_indices]
        y_query = yy[query_indices]
    
        z_support = net(x_support).detach()
        gp.train()
        with torch.no_grad():
            inducing_points, beta, mu_m, U = get_inducing_points(gp, likelihood, z_support, y_support, verbose=False, device=device)
            
        ip_index = inducing_points.index
        ip_values = z_support[ip_index]
        gp.covar_module.inducing_points = nn.Parameter(ip_values, requires_grad=False)
        gp.set_train_data(inputs=z_support, targets=y_support, strict=False)  
        gp.eval()
            
        #Evaluation on all data
        z_all = net(xx).detach()
        mean = likelihood(gp(z_all)).mean
        lower, upper = likelihood(gp(z_all)).confidence_region() #2 standard deviations above and below the mean
        mean = denormalize(mean, y_min, y_max)
        lower = denormalize(lower, y_min, y_max)
        upper = denormalize(upper, y_min, y_max)
        #Plot
        fig, ax = plt.subplots()
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
        ax.scatter(x_support[ip_index], y_support[ip_index], color='darkgreen', marker='*', s=50, zorder=11)
                    
        #all points
        #ax.scatter(x_all.numpy(), y_all.numpy())
        #plt.show()
        plt.ylim(y_range[0], y_range[1])
        plt.xlim(test_range[0], test_range[1])
        plt.savefig('./sines/image/plot_Sparse_DKT_' + str(i) + '.png', dpi=300)

    print("-------------------")
    print("Average MSE: " + str(np.mean(mse_list)) + " +- " + str(np.std(mse_list)))
    print("-------------------")

if __name__ == "__main__":
    main()

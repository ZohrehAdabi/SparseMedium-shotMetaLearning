

import math
from gpytorch import likelihoods
import torch
import gpytorch
from matplotlib import pyplot as plt

def one():
    train_x = torch.linspace(0, 1, 50)
    train_y = torch.sign(torch.cos(train_x * (4 * math.pi))).add(1).div(2)

    class ExactGPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood, kernel='rbf', inducing_points=None):
            super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
            
            self.mean_module  = gpytorch.means.ConstantMean(batch_shape=torch.Size((2,)))

            ## RBF kernel
            if(kernel=='rbf' or kernel=='RBF'):
                # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
                self.base_covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
                self.covar_module = gpytorch.kernels.InducingPointKernel(
                    self.base_covar_module, inducing_points=inducing_points , likelihood=likelihood
                )
    
        def forward(self, x):
            mean_x  = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


    # Initialize model and likelihood
    inducing_point = train_x[:10].clone()
    train_y = torch.round(train_y).long()
    likelihood = gpytorch.likelihoods.DirichletClassificationLikelihood(targets=train_y, learn_additional_noise=False)
    # NOTE THE TRANSFORM HERE
    model = ExactGPModel(train_x, likelihood.transformed_targets, likelihood, 'rbf', inducing_point)
    training_iterations = 200

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()
    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iterations):
        # Zero backpropped gradients from previous iteration
        optimizer.zero_grad()
        # Get predictive output
        output = model(train_x)
        # Calc loss and backprop gradients
        # THE responses we train with are the transformed_targets here.
        loss = -mll(output, likelihood.transformed_targets).sum()
        loss.backward()
        # print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
        if (i+1)%50==0:
            print(f'Iter {i + 1:02}/{training_iterations} - Loss: {loss.item():.4f}')
        optimizer.step()

    # Go into eval mode
    model.eval()
    likelihood.eval()
    with torch.no_grad():
        # Test x are regularly spaced by 0.01 0,1 inclusive
        test_x = torch.linspace(0, 1, 100)
        # test_labels = torch.round(test_x).long()
        # Get classification predictions
        output = model(test_x)
        observed_pred = likelihood(output) 

        # Initialize fig and axes for plot
        f, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.plot(train_x.numpy(), train_y.numpy(), 'k*')
        # Get the predicted labels (probabilites of belonging to the positive class)
        # Transform these probabilities to be 0/1 labels
        print(observed_pred)
        pred_labels = observed_pred.mean.ge(0.5).float()
        pred_labels = (observed_pred.mean[0] < observed_pred.mean[1]).to(int)
        ax.plot(test_x.numpy(), pred_labels.numpy(), 'b')
        ax.set_ylim([-1, 2])
        ax.legend(['Observed Data', 'Mean'])
        plt.show()



class ExactGPLayer(gpytorch.models.ExactGP):
    '''
    Parameters learned by the model:
        likelihood.noise_covar.raw_noise (Gaussian)
        base_covar_module.raw_outputscale
        base_covar_module.base_kernel.raw_lengthscale
    '''
    def __init__(self, train_x, train_y, likelihood, dirichlet, inducing_points, kernel='linear'):
        #Set the likelihood noise and enable/disable learning
        if not dirichlet:
            likelihood.noise_covar.raw_noise.requires_grad = False
            likelihood.noise_covar.noise = torch.tensor(0.1)
        super().__init__(train_x, train_y, likelihood)

        if dirichlet:
            self.mean_module  = gpytorch.means.ConstantMean(batch_shape=torch.Size((2,)))
        else:
            self.mean_module = gpytorch.means.ConstantMean()

        ## Linear kernel
        if(kernel=='linear'):
            self.base_covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.LinearKernel())
        ## RBF kernel
        elif(kernel=='rbf' or kernel=='RBF'):
            self.base_covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        ## Matern kernel
        elif(kernel=='matern'):
            self.base_covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel())
        ## Polynomial (p=1)
        elif(kernel=='poli1'):
            self.base_covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PolynomialKernel(power=1))
        ## Polynomial (p=2)
        elif(kernel=='poli2'):
            self.base_covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PolynomialKernel(power=2))
        elif(kernel=='cossim' or kernel=='bncossim'):
        ## Cosine distance and BatchNorm Cosine distance
            self.base_covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.LinearKernel())
            self.base_covar_module.base_kernel.variance = 1.0
            self.base_covar_module.base_kernel.raw_variance.requires_grad = False
        else:
            raise ValueError("[ERROR] the kernel '" + str(kernel) + "' is not supported!")

        self.covar_module = gpytorch.kernels.InducingPointKernel(self.base_covar_module,
                                         inducing_points=inducing_points, likelihood=likelihood)


    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


dirichlet = True
train_x_list=[torch.ones(10, 3).cuda()]*5
train_y_list=[torch.ones(10).cuda()]*5
model_list = list()
likelihood_list = list()
for train_x, train_y in zip(train_x_list, train_y_list):

    if dirichlet:
        print('dirichlet')
        likelihood = gpytorch.likelihoods.DirichletClassificationLikelihood(targets=train_y.long(), learn_additional_noise=False)
    else:
        likelihood = gpytorch.likelihoods.GaussianLikelihood()

    model = ExactGPLayer(train_x=train_x, train_y= likelihood.transformed_targets, likelihood=likelihood, dirichlet=dirichlet,
                                 inducing_points=train_x[:5], kernel='rbf')
    model_list.append(model)
    likelihood_list.append(model.likelihood)
                                 
model = gpytorch.models.IndependentModelList(*model_list).cuda()
likelihood = gpytorch.likelihoods.LikelihoodList(*likelihood_list).cuda()
mll = gpytorch.mlls.SumMarginalLogLikelihood(likelihood, model).cuda()


output = model(*model.train_inputs)
if dirichlet:
    transformed_targets = [model.likelihood.transformed_targets for model in model.models]
    loss = -mll(output, transformed_targets).sum()
    # loss = 0
    # for (mll_loss, tt, out) in zip(mll.mlls, transformed_targets, output):
    #     # print(f'{mll_loss}, {tt}, {out}')
    #     print(f'loss {-mll_loss(out, tt).sum()}')
        # loss += -mll_loss(out, tt).sum()
loss
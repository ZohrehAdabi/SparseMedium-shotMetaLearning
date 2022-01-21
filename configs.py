save_dir                    = './save/'
data_dir = {}
data_dir['CUB']             = './filelists/CUB/'
data_dir['miniImagenet']    = './filelists/miniImagenet/'
data_dir['omniglot']        = './filelists/omniglot/'
data_dir['emnist']          = './filelists/emnist/'
data_dir['QMUL']          = './filelists/QMUL/'
data_dir['FSC147']          = './filelists/FSC147/'
data_dir['MSC44']          = './filelists/MSC44/'
kernel_type                 = 'linear' #linear, rbf, 2rbf, spectral (regression only), matern, poli1, poli2, cossim, bncossim 
run_float64 = True

init_noise = 0.05
init_outputscale = 0.1
init_lengthscale = 0.5
from . import meta_template
from . import baselinetrain
from . import baselinefinetune 
from . import feature_transfer
from . import matchingnet
from . import protonet
from . import relationnet
from . import MAML, MAML_regression
from . import DKT, DKT_binary, DKT_regression
from . import Sparse_DKT_Nystrom, Sparse_DKT_Exact, Sparse_DKT_binary_RVM
from . import Sparse_DKT_binary_Nystrom, Sparse_DKT_binary_Exact, Sparse_DKT_binary_RVM
from . import Sparse_DKT_regression_Nystrom, Sparse_DKT_regression_Exact, Sparse_DKT_regression_RVM
import sys
sys.path.append("..")
import configs



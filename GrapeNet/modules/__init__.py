from .mlp import MLP
from .perceiver import GraphMultisetAggregation as perceiver

from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
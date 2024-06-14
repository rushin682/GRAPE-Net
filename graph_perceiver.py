import torch
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch.nn.init import kaiming_normal_

from modules import GNN
from aggr import PerceiverPool

class GraPeNet(torch.nn.Module):
    def __init__(self, config):
        '''
            num_tasks (int): number of labels to be predicted
            virtual_node (bool): whether to add virtual node or not
        '''

        super(GraPeNet, self).__init__()
        self.num_layer = config.num_layer
        self.input_dim = config.input_dim
        self.emb_dim = config.emb_dim
        self.gnn_type = config.gnn_type
        self.drop_ratio = config.drop_ratio
        self.jk = config.jk # jumping knowledge connections (i.e skip connections) | options: 'last', 'sum' 
        self.graph_pooling = config.graph_pooling

        if self.num_layer < 0:
            raise ValueError("Number of GNN layers must be greater than 0.")
        
        ### GNN to generate node embeddings
        self.gnn = torch.nn.Sequential(GNN(self.num_layer,
                                           self.input_dim,
                                           self.emb_dim,
                                           jk = self.jk,
                                           drop_ratio = self.drop_ratio,
                                           gnn_type = self.gnn_type))
        
        ### Pooling function to generate whole-graph embeddings
        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool
        elif self.graph_pooling == "perceiver":
            self.pool = PerceiverPool(dim=self.emb_dim, 
                                      k=200, 
                                      num_pma_blocks=1, 
                                      num_encoder_blocks=3, 
                                      heads=8, 
                                      dropout=self.drop_ratio)
        else:
            raise ValueError("Invalid graph pooling type.")

        # classification layers
        self.pred_head = torch.nn.Sequential(torch.nn.Linear(self.emb_dim, self.emb_dim, bias=False),
                                             torch.nn.Linear(self.emb_dim, self.num_class, bias=False))
        
        # Initialize classification prediction head layers with Kaiming normal initialization
        for m in self.pred_head.modules():
            if isinstance(m, torch.nn.Linear):
                kaiming_normal_(m.weight)
                if isinstance(m, torch.nn.Linear) and m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

        self.layer_grads = {}
        self.layer_acts = {}

    def forward(self, data, register_hook=False):
        self.graph_embeddings = self.gnn(data)
        _, batch, node_coords = data.edge_index, data.batch, data.node_coords

        if self.graph_pooling == "perceiver":
            h_graph= self.pool(x=self.graph_embeddings, batch=batch, node_coords=node_coords, register_hook=register_hook)
        else:
            h_graph = self.pool(x=self.graph_embeddings, batch=batch)

        return self.pred_head(h_graph)
    
if __name__ == '__main__':

    import yaml
    from pathlib import Path

    config_file = Path("configs/config.yaml")
    with open(config_file, 'r') as stream:
        config = yaml.safe_load(stream)

    GraPeNet(config)
    print('GraPeNet initialized successfully!')



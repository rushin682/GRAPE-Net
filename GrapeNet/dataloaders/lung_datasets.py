from torch_geometric.data import Data

from .base_dataset import TissueDataset

class TcgaDataset(TissueDataset):
    def __init__(self, root, ids, fdim, n_classes, isTrain=False, transform=None):
        TissueDataset.__init__(self, root, ids, fdim, n_classes, isTrain)

        self.transform = transform

        self.classdict = {'normal': 0, 
                          'lusc': 1, 
                          'luad': 2}
            
        self.to_be_predicted_classes = self.classdict
        
    def fetch_label_from_code(self, label): # Rushin: need to deprecate

        # Label Conversion for 3-Label
        if label == 'TCGA-LUSC': label = 'lusc'
        elif label == 'TCGA-LUAD': label = 'luad'
        elif label == 'Normal': label = 'normal'

        return label

    def __getitem__(self, index):

        info = self.ids[index].replace('\n', '')
        slide_name, label = info.split('\t')[0], info.split('\t')[1]
#        print(slide_name)

        features, adj_s, node_coords = TissueDataset.get_slide_attributes(self, slide_name)

        label = self.fetch_label_from_code(label)

        # Custom Data Object with slide_name & node_coordinates
        geometric_graph = Data(x=features,
                               edge_index=adj_s,
                               edge_attr=None,
                               y=self.classdict[label],
                               slide_path=slide_name,
                               node_coords=node_coords)

        if self.transform:
            geometric_graph = self.transform(geometric_graph)

        return geometric_graph
    
class CptacDataset(TissueDataset):
    def __init__(self, root, ids, fdim, n_classes, isTrain=False, transform=None):
        TissueDataset.__init__(self, root, ids, fdim, n_classes, isTrain)

        self.transform = transform

        self.classdict = {'normal': 0, 
                          'lusc': 1, 
                          'luad': 2}

        self.to_be_predicted_classes = self.classdict

    def __getitem__(self, index):

        info = self.ids[index].replace('\n', '')
        slide_name, label = info.split('\t')[0], info.split('\t')[1]

        features, adj_s, node_coords = TissueDataset.get_slide_attributes(self, slide_name)

        # Label Conversion for 3-Label / 4-Label classification
        # Rushin: Need to deprecate
        if self.n_classes == 3:
            if label == 'lscc': label = 'lusc'
            elif label == 'luad': label = 'luad'

        # Custom Data Object with slide_name & node_coordinates
        geometric_graph = Data(x=features,
                               edge_index=adj_s,
                               edge_attr=None,
                               y=self.classdict[label],
                               slide_path=slide_name,
                               node_coords=node_coords)

        if self.transform:
            geometric_graph = self.transform(geometric_graph)

        return geometric_graph
    
class PcgaDataset(TissueDataset):
    def __init__(self, root, ids, fdim, n_classes, isTrain=False, transform=None):
        TissueDataset.__init__(self, root, ids, fdim, n_classes, isTrain)

        self.transform = transform

        # self.classdict = {'pml_normal': 0, 'hyperplasia': 1, 'metaplasia': 2, 'mild_dysplasia': 3, 'moderate_dysplasia': 4, 'severe_dysplasia': 5, 'cis': 6, 'unknown': 7, 'tumor': 8}
        # self.classdict = {'premalignant': 0}
        self.classdict = {'pml_normal': 0, 'hyperplasia':1, 'metaplasia': 2, 'dysplasia': 3, 'cis': 4}
        
        if self.n_classes == 3:
            self.to_be_predicted_classes = {'normal': 0, 'lusc': 1, 'luad': 2}
        else:
            raise ValueError("Invalid classification type.")

        # self.meta_feats = pd.read_csv(os.path.join('dataset/PCGA/', 'clinical_metadata.csv')) 

    def fetch_label_from_code(self, label): # Rushin: need to deprecate

        if label == 'CIS':
            label = 'cis'

        if label == 'normal':
            label = 'pml_normal'

        return label


    def __getitem__(self, index):

        info = self.ids[index].replace('\n', '')
        slide_name, label = info.split('\t')[0], info.split('\t')[1] 

        features, adj_s, node_coords = TissueDataset.get_slide_attributes(self, slide_name)

        label = self.fetch_label_from_code(label)

        geometric_graph = Data(x=features,
                               edge_index=adj_s,
                               edge_attr=None,
                               y=self.classdict[label],
                               slide_path=slide_name,
                               node_coords=node_coords)

        if self.transform:
            geometric_graph = self.transform(geometric_graph)

        return geometric_graph

class UclDataset(TissueDataset):
    def __init__(self, root, ids, fdim, n_classes, isTrain=False, transform=None):
        TissueDataset.__init__(self, root, ids, fdim, n_classes, isTrain=False)

        self.transform = transform

        # self.classdict = {'pml_normal': 0, 'hyperplasia': 1, 'metaplasia': 2, 'mild_dysplasia': 3, 'moderate_dysplasia': 4, 'severe_dysplasia': 5, 'cis': 6, 'unknown': 7, 'tumor': 8}
        self.classdict = {'cis': 0}

        if self.n_classes == 3:
            self.to_be_predicted_classes = {'normal': 0, 
                                            'lusc': 1,
                                             'luad': 2}

        else:
            raise ValueError("Invalid classification type.")

        # self.meta_feats = pd.read_csv(os.path.join('dataset/CIS', 'clinical_metadata.csv')) # contains metadata for common samples.

    def __getitem__(self, index):

        info = self.ids[index].replace('\n', '')
        slide_name, label = info.split('\t')[0], info.split('\t')[1] 

        features, adj_s, node_coords = TissueDataset.get_slide_attributes(self, slide_name)

        if label == 'CIS': # Rushin: need to deprecate
            label = 'cis'

        geometric_graph = Data(x=features,
                               edge_index=adj_s,
                               edge_attr=None,
                               y=self.classdict[label],
                               slide_path=slide_name,
                               node_coords=node_coords)
        
        if self.transform:
            geometric_graph = self.transform(geometric_graph)

        return geometric_graph
    
class NlstDataset(TissueDataset):
    def __init__(self, root, ids, fdim, c_type, isTrain=False):
        TissueDataset.__init__(self, root, ids, fdim, c_type, isTrain)

        self.classdict = {'normal': 0, 
                          'lusc': 1, 
                          'luad': 2}

        self.to_be_predicted_classes = self.classdict

    def __getitem__(self, index):

        info = self.ids[index].replace('\n', '')
        slide_name, label = info.split('\t')[0], info.split('\t')[1]

        features, adj_s, node_coords = TissueDataset.get_slide_attributes(self, slide_name)

        # Label Conversion for 3-Label / 4-Label classification
        # Rushin: Need to deprecate
        if self.n_classes == 3:
            if label == 'lscc': label = 'lusc'
            elif label == 'luad': label = 'luad'

        # Custom Data Object with slide_name & node_coordinates
        geometric_graph = Data(x=features,
                               edge_index=adj_s,
                               edge_attr=None,
                               y=self.classdict[label],
                               slide_path=slide_name,
                               node_coords=node_coords)

        if self.transform:
            geometric_graph = self.transform(geometric_graph)

        return geometric_graph
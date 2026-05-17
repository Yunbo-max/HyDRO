import json
import os.path as osp
import os
import pickle

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from ogb.nodeproppred import PygNodePropPredDataset
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from torch_geometric.datasets import Planetoid, Coauthor, CitationFull, Amazon, Flickr, Reddit2,SNAPDataset
from torch_geometric.loader import NeighborSampler
from torch_geometric.utils import to_undirected, add_self_loops
from torch_sparse import SparseTensor
from dgl.data import FraudDataset
import shutil

from graphslim.dataset.convertor import ei2csr, csr2ei, from_dgl
from graphslim.dataset.utils import splits
from graphslim.utils import index_to_mask, to_tensor


def get_dataset(name='cora', args=None, load_path='../../data'):
    path = osp.join(load_path)
    # Create a dictionary that maps standard names to normalized names
    standard_names = ['flickr', 'reddit', 'dblp', 'cora_ml', 'physics', 'cs', 'cora', 'citeseer', 'pubmed', 'photo',
                      'computers', 'ogbn-products', 'ogbn-proteins', 'ogbn-papers100m', 'ogbn-arxiv', 'yelp', 'amazon','amazon0302','roadnet-ca','ego-facebook','SCNp','SCNn']
    # normalized_names = [name.lower().replace('-', '').replace('_', '') for name in standard_names]
    # name_dict = dict(zip(normalized_names, standard_names))

    # # Normalize the name input
    # normalized_name = name.lower().replace('-', '').replace('_', '')

    if name in standard_names:
        print("yes,it is ")
        # name = name_dict[normalized_name]  # Transfer to standard name
        if name in ['flickr']:
            dataset = Flickr(root=path + '/flickr')
        elif name in ['reddit']:
            dataset = Reddit2(root=path + '/reddit')
        elif name in ['roadnet-ca','amazon0302','ego-facebook']:
            dataset = SNAPDataset(root=path, name=name)
        elif name in ['dblp', 'cora_ml', 'cora_full', 'citeseer_full']:
            dataset = CitationFull(root=path, name=name)
        elif name in ['physics', 'cs']:
            dataset = Coauthor(root=path, name=name)
        elif name in ['cora', 'citeseer', 'pubmed']:
            dataset = Planetoid(root=path, name=name)
        elif name in ['photo', 'computers']:
            dataset = Amazon(root=path, name=name)
        elif name in ['SCNn']:
            dataset = SCN(root=path,name=name)

        elif name in ['ogbn-arxiv']:
            dataset = DataGraphSAINT(root=path, dataset=name)
            dataset.num_classes = 40
        elif name in ['ogbn-products', 'ogbn-proteins', 'ogbn-papers100m']:
            dataset = PygNodePropPredDataset(name, root=path)
        elif name in ['yelp', 'amazon']:
            # dataset = pickle.load(open(f'{path}/{args.dataset}.dat', 'rb'))
            # dataset.num_classes = 2
            dataset = FraudDataset(name, raw_dir=path)
            dataset = from_dgl(dataset[0], name=name, hetero=False)  # dgl2pyg
    else:
        raise ValueError("Dataset name not recognized.")

    try:
        data = dataset[0]

    except:
        data = dataset

    # pyg2TransAndInd: add splits
    if name in 'SCNn':
        data = data
    
    else:
        data = splits(data, args.split)

    data = TransAndInd(data, name, args.pre_norm)

    try:
        data.nclass = dataset.num_classes
    except:
        data.nclass = data.num_classes

    print("train nodes num:", sum(data.train_mask).item())
    print("val nodes num:", sum(data.val_mask).item())
    print("test nodes num:", sum(data.test_mask).item())
    print("total nodes num:", data.x.shape[0])
    return data


class TransAndInd:

    def __init__(self, data, dataset, norm=True):
        self.class_dict = None  # sample the training data per class when initializing synthetic graph
        self.samplers = None
        self.class_dict2 = None  # sample from the same class when training
        self.sparse_adj = None
        self.adj_full = None
        self.feat_full = None
        self.labels_full = None
        self.num_nodes = data.num_nodes
        self.train_mask, self.val_mask, self.test_mask = data.train_mask, data.val_mask, data.test_mask
        self.pyg_saint(data)
        if dataset in ['flickr', 'reddit', 'ogbn-arxiv']:
            self.edge_index = to_undirected(self.edge_index, self.num_nodes)
            feat_train = self.x[data.idx_train]
            scaler = StandardScaler()
            scaler.fit(feat_train)
            self.feat_full = scaler.transform(self.x)
            self.feat_full = torch.from_numpy(self.feat_full).float()
        if norm and dataset in ['cora', 'citeseer', 'pubmed']:
            self.feat_full = F.normalize(self.feat_full, p=1, dim=1)
        self.idx_train, self.idx_val, self.idx_test = data.idx_train, data.idx_val, data.idx_test
        # self.nclass = max(self.labels_full).item() + 1

        self.adj_train = self.adj_full[np.ix_(self.idx_train, self.idx_train)]
        self.adj_val = self.adj_full[np.ix_(self.idx_val, self.idx_val)]
        self.adj_test = self.adj_full[np.ix_(self.idx_test, self.idx_test)]

        self.labels_train = self.labels_full[self.idx_train]
        self.labels_val = self.labels_full[self.idx_val]
        self.labels_test = self.labels_full[self.idx_test]

        self.feat_train = self.feat_full[self.idx_train]
        self.feat_val = self.feat_full[self.idx_val]
        self.feat_test = self.feat_full[self.idx_test]

    def to(self, device):
        """Move data to the specified device."""
        self.feat_full = self.feat_full.to(device)
        self.labels_full = self.labels_full.to(device)
        self.x = self.x.to(device)
        self.y = self.y.to(device)
        self.edge_index = self.edge_index.to(device)

        self.feat_train = self.feat_train.to(device)
        self.feat_val = self.feat_val.to(device)
        self.feat_test = self.feat_test.to(device)

        # self.labels_train = self.labels_train.to(device)
        # self.labels_val = self.labels_val.to(device)
        # self.labels_test = self.labels_test.to(device)

        return self

    def pyg_saint(self, data):
        # reference type
        # pyg format use x,y,edge_index
        if hasattr(data, 'x'):
            self.x = data.x
            self.y = data.y
            self.feat_full = data.x
            self.labels_full = data.y
            self.adj_full = ei2csr(data.edge_index, data.x.shape[0])
            self.edge_index = data.edge_index
            self.sparse_adj = SparseTensor.from_edge_index(data.edge_index)
        # saint format use feat,labels,adj
        elif hasattr(data, 'feat_full'):
            self.adj_full = data.adj_full
            self.feat_full = data.feat_full
            self.labels_full = data.labels_full
            self.edge_index = csr2ei(data.adj_full)
            self.sparse_adj = SparseTensor.from_edge_index(self.edge_index)
            self.x = data.feat_full
            self.y = data.labels_full
        return data

    def retrieve_class(self, c, num=256):
        # change the initialization strategy here
        if self.class_dict is None:
            self.class_dict = {}
            for i in range(self.nclass):
                self.class_dict['class_%s' % i] = (self.labels_train == i)
        idx = np.arange(len(self.labels_train))
        idx = idx[self.class_dict['class_%s' % c]]
        return np.random.permutation(idx)[:num]

    def retrieve_class_sampler(self, c, adj, args, num=256):
        if self.class_dict2 is None:
            self.class_dict2 = {}
            for i in range(self.nclass):
                if args.setting == 'trans':
                    idx = self.idx_train[self.labels_train == i]
                else:
                    idx = np.arange(len(self.labels_train))[self.labels_train == i]
                self.class_dict2[i] = idx

        if args.nlayers == 1:
            sizes = [15]
        elif args.nlayers == 2:
            sizes = [15, 8] if args.dataset in ['reddit', 'flickr'] else [10, 5]
        elif args.nlayers == 3:
            sizes = [15, 10, 5]
        elif args.nlayers == 4:
            sizes = [15, 10, 5, 5]
        elif args.nlayers == 5:
            sizes = [15, 10, 5, 5, 5]

        if self.samplers is None:
            self.samplers = []
            for i in range(self.nclass):
                node_idx = torch.LongTensor(self.class_dict2[i])
                self.samplers.append(
                    NeighborSampler(
                        adj,
                        node_idx=node_idx,
                        sizes=sizes,
                        batch_size=num,
                        num_workers=8,
                        return_e_id=False,
                        num_nodes=adj.size(0),
                        shuffle=True,
                    )
                )
        batch = np.random.permutation(self.class_dict2[c])[:num]
        out = self.samplers[c].sample(batch.astype(np.int64))
        print(type(out), len(out))  # Check the number of items in the tuple/list
        print("Sampler output:", out)

        
        # Ensure the returned object is unpackable into three components
        n_id, adjs = out.node_id, out.adjs
        batch_size = len(batch)  # Infer batch size
        return batch_size, n_id, adjs


    def reset(self):
        self.samplers = None
        self.class_dict2 = None
        self.labels_syn, self.feat_syn, self.adj_syn = None, None, None


class LargeDataLoader(nn.Module):
    def __init__(self, name='Flickr', split='train', batch_size=200, split_method='kmeans'):
        super(LargeDataLoader, self).__init__()
        path = osp.join('../../data')
        if name in ['ogbn-arxiv']:
            dataset = DataGraphSAINT(root=path, dataset=name)
            dataset.num_classes = 40
            data = dataset[0]
            self.n, self.dim = data.feat_full.shape
            labels = data.labels_full
            features = to_tensor(data.feat_full)
            edge_index = csr2ei(data.adj_full)
            values = torch.ones(edge_index.shape[1])
            Adj = torch.sparse_coo_tensor(edge_index, values, torch.Size([self.n, self.n]))
            sparse_eye = torch.sparse_coo_tensor(torch.arange(self.n).repeat(2, 1), torch.ones(self.n),
                                                 (self.n, self.n))
            self.Adj = Adj + sparse_eye

            features = self.normalize_data(features)
            features = self.GCF(self.Adj, features, k=1)

            self.split_idx = torch.tensor(data.idx_train)
            self.n_split = len(self.split_idx)
            self.k = torch.round(torch.tensor(self.n_split / batch_size)).to(torch.int)
            self.split_feat = features[self.split_idx]
            self.split_label = labels[self.split_idx]

            self.split_method = split_method
            self.n_classes = dataset.num_classes
        else:
            if name == 'flickr':
                from torch_geometric.datasets import Flickr as DataSet
            elif name == 'reddit':
                from torch_geometric.datasets import Reddit2 as DataSet

            Dataset = DataSet(root=path + f'/{name}')
            self.n, self.dim = Dataset[0].x.shape
            mask = split + '_mask'
            features = Dataset[0].x
            labels = Dataset[0].y
            edge_index = Dataset[0].edge_index

            values = torch.ones(edge_index.shape[1])
            Adj = torch.sparse_coo_tensor(edge_index, values, torch.Size([self.n, self.n]))
            sparse_eye = torch.sparse_coo_tensor(torch.arange(self.n).repeat(2, 1), torch.ones(self.n),
                                                 (self.n, self.n))
            self.Adj = Adj + sparse_eye
            features = self.normalize_data(features)
            # features      = self.GCF(self.Adj, features, k=2)
            self.split_idx = torch.where(Dataset[0][mask])[0]
            self.n_split = len(self.split_idx)
            self.k = torch.round(torch.tensor(self.n_split / batch_size)).to(torch.int)

            # Masked Adjacency Matrix
            optor_index = torch.cat(
                (self.split_idx.reshape(1, self.n_split), torch.tensor(range(self.n_split)).reshape(1, self.n_split)),
                dim=0)
            optor_value = torch.ones(self.n_split)
            optor_shape = torch.Size([self.n, self.n_split])
            optor = torch.sparse_coo_tensor(optor_index, optor_value, optor_shape)
            self.Adj_mask = torch.sparse.mm(torch.sparse.mm(optor.t(), self.Adj), optor)
            self.split_feat = features[self.split_idx]
            # self.split_feat   = self.GCF(self.Adj_mask, self.split_feat, k = 2)

            self.split_label = labels[self.split_idx]
            self.split_method = split_method
            self.n_classes = Dataset.num_classes

    def normalize_data(self, data):
        """
        normalize data
        parameters:
            data: torch.Tensor, data need to be normalized
        return:
            torch.Tensor, normalized data
        """
        mean = data.mean(dim=0)  
        std = data.std(dim=0)  
        std[std == 0] = 1  
        normalized_data = (data - mean) / std
        return normalized_data

    def GCF(self, adj, x, k=2):
        """
        Graph convolution filter
        parameters:
            adj: torch.Tensor, adjacency matrix, must be self-looped
            x: torch.Tensor, features
            k: int, number of hops
        return:
            torch.Tensor, filtered features
        """
        n = adj.shape[0]
        ind = torch.tensor(range(n)).repeat(2, 1)
        adj = adj + torch.sparse_coo_tensor(ind, torch.ones(n), (n, n))

        D = torch.pow(torch.sparse.sum(adj, 1).to_dense(), -0.5)
        D = torch.sparse_coo_tensor(ind, D, (n, n))

        filter = torch.sparse.mm(torch.sparse.mm(D, adj), D)
        for i in range(k):
            x = torch.sparse.mm(filter, x)
        return x

    def properties(self):
        return self.k, self.n_split, self.n_classes, self.dim, self.n

    def split_batch(self):
        """
        split data into batches
        parameters:
            split_method: str, method to split data, default is 'kmeans'
        """
        if self.split_method == 'kmeans':
            kmeans = KMeans(n_clusters=self.k.item(), n_init=10)
            kmeans.fit(self.split_feat.numpy())
            self.batch_labels = kmeans.predict(self.split_feat.numpy())

    def getitem(self, idx):
        """
        对于给定的 idx 输出对应的 node_features, labels, sub Ajacency matrix
        """
        # idx   = [idx]
        n_idx = len(idx)
        idx_raw = self.split_idx[idx]
        feat = self.split_feat[idx]
        label = self.split_label[idx]
        # idx   = idx.tolist()

        optor_index = torch.cat((idx_raw.reshape(1, n_idx), torch.tensor(range(n_idx)).reshape(1, n_idx)), dim=0)
        optor_value = torch.ones(n_idx)
        optor_shape = torch.Size([self.n, n_idx])
        optor = torch.sparse_coo_tensor(optor_index, optor_value, optor_shape)
        sub_A = torch.sparse.mm(torch.sparse.mm(optor.t(), self.Adj), optor)

        return (feat, label, sub_A)

    def get_batch(self, i):
        idx = torch.where(torch.tensor(self.batch_labels) == i)[0]
        batch_i = self.getitem(idx)
        return batch_i


class OgbDataLoader(nn.Module):
    def __init__(self, dataset_name='ogbn-arxiv', split='train', batch_size=5000, split_method='kmeans'):
        super(OgbDataLoader, self).__init__()









import torch
import pandas as pd
import numpy as np
import gensim
from sklearn.model_selection import train_test_split
from torch_geometric.data import InMemoryDataset, Data
import torch_geometric.transforms as T

class SCN(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None):
        self.name = name
        super(SCN, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass
    
    # Function to process the dataframe and create the graph
    def process(self):
        data_list = []
        
        # Load the dataframe
        data_path = osp.join(osp.dirname(osp.abspath(__file__)), '..', '..', 'data', 'Final_markline.csv')
        df = pd.read_csv(data_path)

        # Step 1: Train Word2Vec model on company names and suppliers (tokenize by space)
        model = gensim.models.Word2Vec(
            [name.split() for name in df['CompanyName']] + [name.split() for name in df['Suppliers']],
            vector_size=100,  # Vector dimension for each word embedding
            window=5,         # Context window size
            min_count=1,      # Minimum word frequency
            workers=4         # Number of CPU cores for training
        )

        MAX_TOKENS = 10  # Maximum number of tokens

        # Function to pad or truncate embeddings
        def pad_or_truncate(embeddings, max_tokens=MAX_TOKENS, embedding_size=100):
            while len(embeddings) < max_tokens:
                embeddings.append(np.zeros(embedding_size))
            return embeddings[:max_tokens]

        # Function to get embeddings for company and supplier names
        def get_name_embeddings(name):
            words = name.split()
            word_embeddings = [model.wv[word] for word in words if word in model.wv]
            return pad_or_truncate(word_embeddings)

        # Get embeddings for companies and suppliers
        df['Company_Embeddings'] = df['CompanyName'].apply(get_name_embeddings)
        df['Supplier_Embeddings'] = df['Suppliers'].apply(get_name_embeddings)

        # 1. Prepare Node Features and Categories (with unique encoding for categories)
        node_features = []
        node_categories = []
        node_mapping = {}
        
        # Create a mapping for categories (unique values for countries)
        company_countries = list(set(df['company_country'].values))
        supplier_countries = list(set(df['supplier_country'].values))
        all_categories = company_countries + supplier_countries
        
        # Map categories to unique indices
        category_mapping = {category: idx for idx, category in enumerate(all_categories)}

        # Count how many nodes belong to each category
        category_counts = df['company_country'].value_counts().to_dict()
        category_counts.update(df['supplier_country'].value_counts().to_dict())

        # Step 2: Filter out categories with fewer than 100 nodes
        valid_categories = {category for category, count in category_counts.items() if count > 100}

        # Filter out nodes based on valid categories
        df_filtered = df[df['company_country'].isin(valid_categories) | df['supplier_country'].isin(valid_categories)]

        for idx, row in df_filtered.iterrows():
            # Company Node: Use embeddings and map category to unique integer
            company_embedding = row['Company_Embeddings']
            company_category = row['company_country']
            node_features.append(company_embedding)
            node_categories.append(category_mapping[company_category])
            node_mapping[row['CompanyName']] = len(node_mapping)

            # Supplier Node: Use embeddings and map category to unique integer
            supplier_embedding = row['Supplier_Embeddings']
            supplier_category = row['supplier_country']
            node_features.append(supplier_embedding)
            node_categories.append(category_mapping[supplier_category])
            node_mapping[row['Suppliers']] = len(node_mapping)

        # Convert node features to tensor
        node_features = torch.tensor(node_features, dtype=torch.float)
        node_categories = torch.tensor(node_categories, dtype=torch.long)

        # 2. Create Edge List (Edge Index) directly from the filtered dataframe
        edges = []
        for idx, row in df_filtered.iterrows():
            company_idx = node_mapping[row['CompanyName']]
            supplier_idx = node_mapping[row['Suppliers']]
            edges.append([company_idx, supplier_idx])

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        # 3. Split data into train and test sets
        train_mask, test_mask = train_test_split(range(len(node_categories)), test_size=0.7, random_state=42)

        # 4. Create PyTorch Geometric Data Object
        data = Data(x=node_features, edge_index=edge_index, y=node_categories)
        data.x = node_features.view(node_features.shape[0], -1)

        # 5. Randomly split nodes into train, validation, and test sets
        split = T.RandomNodeSplit(num_val=0.1, num_test=0.5)
        data = split(data)
        print("Node features shape:", data.x.shape)
        print("Edge index shape:", data.edge_index.shape)
        print("Node categories:", data.y)

        # Add the data to the data list
        data_list.append(data)

        # Save processed data
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])



class DataGraphSAINT:
    '''datasets used in GraphSAINT paper'''

    def __init__(self, root, dataset, **kwargs):
        import gdown
        dataset = dataset.replace('-', '_')
        dataset_str = root + '/' + dataset + '/raw/'
        if not osp.exists(dataset_str):
            os.makedirs(dataset_str)
            print('Downloading dataset')
            url = 'https://drive.google.com/drive/folders/1VDobXR5KqKoov6WhYXFMwH4rN0FMnVOa'  # Change this to your actual file ID
            downloaded_folder=gdown.download_folder(url=url, output=dataset_str, quiet=False)
            if downloaded_folder and osp.isdir(downloaded_folder):
                for filename in os.listdir(downloaded_folder):
                    file_path = osp.join(downloaded_folder, filename)
                    shutil.move(file_path, dataset_str)

                shutil.rmtree(downloaded_folder)

        if dataset == 'ogbn_arxiv':
            self.adj_full = sp.load_npz(dataset_str + 'adj_full.npz')
            self.adj_full = self.adj_full + self.adj_full.T
            self.adj_full[self.adj_full > 1] = 1

        self.num_nodes = self.adj_full.shape[0]

        role = json.load(open(dataset_str + 'role.json', 'r'))
        self.idx_train = role['tr']
        self.idx_test = role['te']
        self.idx_val = role['va']
        self.train_mask = index_to_mask(self.idx_train, self.num_nodes)
        self.test_mask = index_to_mask(self.idx_test, self.num_nodes)
        self.val_mask = index_to_mask(self.idx_val, self.num_nodes)

        self.feat_full = np.load(dataset_str + 'feats.npy')
        # ---- normalize feat ----

        class_map = json.load(open(dataset_str + 'class_map.json', 'r'))
        self.labels_full = to_tensor(label=self.process_labels(class_map))

    def process_labels(self, class_map):
        """
        setup vertex property map for output classests
        """
        num_vertices = self.num_nodes
        if isinstance(list(class_map.values())[0], list):
            num_classes = len(list(class_map.values())[0])
            self.nclass = num_classes
            class_arr = np.zeros((num_vertices, num_classes))
            for k, v in class_map.items():
                class_arr[int(k)] = v
        else:
            class_arr = np.zeros(num_vertices, dtype=np.int64)
            for k, v in class_map.items():
                class_arr[int(k)] = v
            class_arr = class_arr - class_arr.min()
            self.nclass = max(class_arr) + 1
        return class_arr

    def get(self, idx):
        return self

    def __getitem__(self, idx):
        return self.get(idx)

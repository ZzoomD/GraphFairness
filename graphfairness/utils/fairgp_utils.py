import torch
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import networkx as nx
import heapq
from torch_geometric.utils import to_scipy_sparse_matrix

# Optional imports for partitioning methods
try:
    import metis
except ImportError:
    metis = None

try:
    from community import community_louvain
except ImportError:
    community_louvain = None

try:
    import leidenalg
    import igraph as ig
except ImportError:
    leidenalg = None
    ig = None

def edge_index_2_sparse_mx(edge_index, num_nodes=None):
    if num_nodes is None:
        num_nodes = edge_index.max().item() + 1
    
    if edge_index.is_cuda:
        edge_index = edge_index.cpu()
        
    row = edge_index[0].numpy()
    col = edge_index[1].numpy()
    data = np.ones(len(row))
    
    # Use scipy.sparse.coo_matrix
    sparse_adj = sp.coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
    return sparse_adj

def adjacency_positional_encoding(g, pos_enc_dim):
    # g is scipy sparse matrix
    eignvalue, eignvector = sp.linalg.eigsh(g, which='LM', k=pos_enc_dim)
    eignvalue = torch.from_numpy(eignvalue).float()
    eignvector = torch.from_numpy(eignvector).float()
    return eignvalue, eignvector

def laplacian_positional_encoding(g, pos_enc_dim):
    # g is scipy sparse matrix
    laplacian = sp.csgraph.laplacian(g, normed=False)
    eignvalue, eignvector = sp.linalg.eigsh(laplacian, which='LM', k=pos_enc_dim)
    eignvalue = torch.from_numpy(eignvalue).float()
    eignvector = torch.from_numpy(eignvector).float()
    return eignvalue, eignvector

# Partitioning functions
def random_partition(num_nodes, n_patches=50, seed=None):
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    if num_nodes < n_patches:
        membership = torch.randperm(num_nodes)
    else:
        membership = torch.randint(0, n_patches, (num_nodes,))

    patch = []
    max_patch_size = -1
    for i in range(n_patches):
        patch.append(list())
        patch[-1] = torch.where(membership == i)[0].tolist()
        max_patch_size = max(max_patch_size, len(patch[-1]))

    for i in range(len(patch)):
        l = len(patch[i])
        if l < max_patch_size:
            patch[i] += [num_nodes] * (max_patch_size - l)

    patch = torch.tensor(patch)
    return patch

def metis_partition(edge_index, num_nodes, n_patches=50, seed=None):
    if metis is None:
        raise ImportError("metis is not installed. Please install it to use 'metis' partition.")
        
    if num_nodes < n_patches:
        membership = torch.randperm(n_patches)
    else:
        adjlist = edge_index.t()
        G = nx.Graph()
        G.add_nodes_from(np.arange(num_nodes))
        G.add_edges_from(adjlist.tolist())
        cuts, membership = metis.part_graph(G, n_patches, recursive=True)

    membership = torch.tensor(membership[:num_nodes])

    patch = []
    max_patch_size = -1
    for i in range(n_patches):
        patch.append(list())
        patch[-1] = torch.where(membership == i)[0].tolist()
        max_patch_size = max(max_patch_size, len(patch[-1]))

    for i in range(len(patch)):
        l = len(patch[i])
        if l < max_patch_size:
            patch[i] += [num_nodes] * (max_patch_size - l)

    patch = torch.tensor(patch)
    return patch

def louvain_partition(edge_index, num_nodes, n_patches=50, seed=None):
    if community_louvain is None:
        raise ImportError("python-louvain is not installed. Please install it to use 'louvain' partition.")

    adjlist = edge_index.t()
    G = nx.Graph()
    G.add_nodes_from(np.arange(num_nodes))
    G.add_edges_from(adjlist.tolist())

    partition = community_louvain.best_partition(G)
    membership = np.array([partition[i] for i in range(num_nodes)])

    unique_labels = np.unique(membership)
    label_map = {label: i for i, label in enumerate(unique_labels)}
    membership = np.array([label_map[label] for label in membership])
    
    # Simple logic to ensure n_patches (simplified from original for brevity, can be expanded if needed)
    # The original logic is complex regarding splitting/merging. 
    # For now I will include the core logic from the original file.
    
    unique_labels, counts = np.unique(membership, return_counts=True)
    max_nodes_per_patch = num_nodes // n_patches
    partition_groups = {label: np.where(membership == label)[0].tolist() for label in unique_labels}
    
    new_groups = []
    items_to_modify = []

    for group_i, nodes in partition_groups.items():
        while len(nodes) > max_nodes_per_patch:
            long_group = list.copy(nodes)
            partition_groups[group_i] = list.copy(long_group[:max_nodes_per_patch])
            new_grp_i = max(partition_groups.keys()) + 1
            new_groups.append(new_grp_i)
            items_to_modify.append((new_grp_i, long_group[max_nodes_per_patch:]))
            nodes = long_group[max_nodes_per_patch:]

    for new_grp_i, new_nodes in items_to_modify:
        partition_groups[new_grp_i] = new_nodes

    unique_labels = list(partition_groups.keys())

    if len(unique_labels) > n_patches:
        community_sizes = [(len(partition_groups[label]), label) for label in unique_labels]
        heapq.heapify(community_sizes)
        
        while len(unique_labels) > n_patches:
            smallest_community_size, smallest_community = heapq.heappop(community_sizes)
            smallest_community_members = partition_groups.pop(smallest_community)
            unique_labels.remove(smallest_community)
            
            closest_community = min(
                unique_labels,
                key=lambda x: (
                    len(set(smallest_community_members) & set(partition_groups[x])) + len(partition_groups[x])
                )
            )
            
            partition_groups[closest_community].extend(smallest_community_members)
            unique_labels = list(partition_groups.keys())
            community_sizes = [(len(partition_groups[label]), label) for label in unique_labels]
            heapq.heapify(community_sizes)
    else:
        n_patches = len(unique_labels)

    patch = []
    max_patch_size = -1
    for label in unique_labels:
        patch_i = partition_groups[label]
        patch.append(patch_i)
        max_patch_size = max(max_patch_size, len(patch_i))

    for i in range(len(patch)):
        l = len(patch[i])
        if l < max_patch_size:
            patch[i].extend([num_nodes] * (max_patch_size - l))

    patch = torch.tensor(patch)
    return patch, n_patches

def partition_patch(node_feat, edge_index, labels, n_patches, num_nodes, method='random', seed=None):
    if num_nodes is None:
        num_nodes = node_feat.shape[0]

    if n_patches == 1:
        patch = torch.tensor(range(num_nodes + 1)).unsqueeze(dim=0)
    else:
        if method == 'metis':
            patch = metis_partition(edge_index, num_nodes=num_nodes, n_patches=n_patches)
        elif method == 'louvain':
            patch, n_patches = louvain_partition(edge_index, num_nodes=num_nodes, n_patches=n_patches)
        elif method == 'random':
            patch = random_partition(num_nodes=num_nodes, n_patches=n_patches)
        else:
            # Fallback to random if method not supported or unimplemented
            print(f"partition method {method} not implemented in this port, using random")
            patch = random_partition(num_nodes=num_nodes, n_patches=n_patches)

    # Pad data for the virtual node (index = num_nodes)
    num_nodes += 1
    node_feat = F.pad(node_feat, [0, 0, 0, 1]) # Pad feat dim 0 at bottom
    labels = F.pad(labels, [0, 1])
    
    return patch, node_feat, labels, num_nodes

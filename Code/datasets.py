from torch.utils.data import Dataset
import torch
import numpy as np
from einops import rearrange
import scipy.sparse as sp
from sklearn.decomposition import PCA

from utils import plot_ST, read_anndata

class ST2D(Dataset):
    def __init__(self, data_file, require_coordnorm=True, keep_ratio=True, **kwargs):
        super().__init__()
        adata = read_anndata(data_file)
        
        self.coordinates = adata.obsm["spatial"].astype(float)
        
        if sp.issparse(adata.X):
            self.raw_representations = adata.X.toarray()
        else:
            self.raw_representations = adata.X
        
        # take statistics
        self.n_cell = self.raw_representations.shape[0]
        self.n_gene = self.raw_representations.shape[1]
        self.n_channels = self.n_gene
        
        if "embeddings" in adata.obsm:
            self.embeddings = adata.obsm["embeddings"]
            assert self.raw_representations.shape[0] == self.embeddings.shape[0]
            self.n_embd = self.embeddings.shape[1]
        else: 
            self.embeddings = None

        # Fixed train/val split from preprocessing
        self.split = None
        if "split" in adata.obs:
            self.split = np.asarray(adata.obs["split"].astype(str))
            assert len(self.split) == self.n_cell, (
                f"split length {len(self.split)} != number of cells/spots {self.n_cell}"
            )

        # Optional local image features, e.g. adata.obsm["z_local"]
        self.local_features = None
        self.n_local = None
        self.local_key = kwargs.get("local_key", None)

        if self.local_key is not None:
            if self.local_key not in adata.obsm:
                raise KeyError(
                    f"local_key={self.local_key} is not found in adata.obsm. "
                    f"Available obsm keys are: {list(adata.obsm.keys())}"
                )

            self.local_features = np.asarray(adata.obsm[self.local_key]).astype(np.float32)

            assert self.local_features.shape[0] == self.n_cell, (
                f"local_features has {self.local_features.shape[0]} rows, "
                f"but expression matrix has {self.n_cell} rows."
            )

            self.n_local = self.local_features.shape[1]

        if require_coordnorm:
            self._normalize_coordinates(keep_ratio=keep_ratio)

        self.raw_pca = PCA(n_components=3, random_state=0) # map raw representation dimension to 3 for visualization
        self.raw_pca.fit(self.raw_representations)

        if self.has_embeddings():
            self.embd_pca = PCA(n_components=3, random_state=0) # map embedding dimension to 3 for visualization
            self.embd_pca.fit(self.embeddings)
        
        # Optional image features for alignment, e.g. adata.obsm["phikon_cls"]
        self.image_features = None
        self.image_key = kwargs.get("image_key", None)

        if self.image_key is not None:
            if self.image_key not in adata.obsm:
                raise KeyError(
                    f"image_key={self.image_key} is not found in adata.obsm. "
                    f"Available obsm keys are: {list(adata.obsm.keys())}"
                )

            self.image_features = np.asarray(adata.obsm[self.image_key]).astype(np.float32)

            assert self.image_features.shape[0] == self.n_cell, (
                f"image_features has {self.image_features.shape[0]} rows, "
                f"but expression matrix has {self.n_cell} rows."
            )
        
    def has_image_features(self):
        return self.image_features is not None
    
    def has_embeddings(self):
        return False if self.embeddings is None else True

    def has_split(self):
        return self.split is not None

    def get_split_indices(self, split_name):
        if not self.has_split():
            raise ValueError("This dataset has no split information in adata.obs['split'].")

        indices = np.where(self.split == split_name)[0].tolist()

        if len(indices) == 0:
            raise ValueError(
                f"No samples found for split='{split_name}'. "
                f"Available split values are: {np.unique(self.split)}"
            )

        return indices

    def get_train_val_indices(self):
        train_idx = self.get_split_indices("train")
        val_idx = self.get_split_indices("val")
        return train_idx, val_idx

    def has_local_features(self):
        return False if self.local_features is None else True


    def get_local_dim(self):
        if self.has_local_features():
            return self.n_local
        else:
            return None

    def plot_raw_representations(self, spot_size=2, train_indices=None, val_indices=None):
        if train_indices and val_indices:
            train_fig = plot_ST(self.coordinates[train_indices,:], self.raw_pca.transform(self.raw_representations[train_indices,:]), spot_size)
            val_fig = plot_ST(self.coordinates[val_indices,:], self.raw_pca.transform(self.raw_representations[val_indices,:]), spot_size)
            return train_fig, val_fig
        else:
            fig = plot_ST(self.coordinates, self.raw_pca.transform(self.raw_representations), spot_size)
        return fig
    
    def plot_embeddings(self, spot_size=2, train_indices=None, val_indices=None):
        assert self.has_embeddings(), "The current adata file has NO embeddings!"
        if train_indices and val_indices:
            train_fig = plot_ST(self.coordinates[train_indices,:], self.embd_pca.transform(self.embeddings[train_indices,:]), spot_size)
            val_fig = plot_ST(self.coordinates[val_indices,:], self.embd_pca.transform(self.embeddings[val_indices,:]), spot_size)
            return train_fig, val_fig
        else:
            fig = plot_ST(self.coordinates, self.embd_pca.transform(self.embeddings), spot_size)
            return fig
    

    # normalize coordinates to [-1.0, +1.0]
    def _normalize_coordinates(self, keep_ratio):
        x_min, y_min = list(self.coordinates.min(axis=0))
        x_max, y_max = list(self.coordinates.max(axis=0))
        x_range, y_range = x_max - x_min, y_max - y_min

        self.coordinates[:,0] = (self.coordinates[:,0] - x_min) / x_range
        self.coordinates[:,1] = (self.coordinates[:,1] - y_min) / y_range

        self.coordinates -= 0.5
        self.coordinates *= 2.0

        if keep_ratio: # may cause waste of space in the short side
            max_range = max(x_range, y_range)
            scale_x, scale_y = x_range / max_range, y_range / max_range
            self.coordinates[:,0] *= scale_x
            self.coordinates[:,1] *= scale_y
    
    def __len__(self):
        return self.n_cell
    
    def get_raw_dim(self):
        return self.n_gene
    
    def get_embd_dim(self):
        if self.has_embeddings():
            return self.n_embd
        else:
            return None
    
    def __getitem__(self, idx):
        item = {
            "idx": idx,
            "coordinates": torch.Tensor(self.coordinates[idx, :].copy()).float(),
            "raw_representations": torch.Tensor(
                self.raw_representations[idx, :].copy()
            ).float(),
        }

        if self.has_embeddings():
            item["embeddings"] = torch.Tensor(
                self.embeddings[idx, :].copy()
            ).float()

        if self.has_local_features():
            item["local_features"] = torch.Tensor(
                self.local_features[idx, :].copy()
            ).float()

        if self.has_image_features():
            item["image_features"] = torch.Tensor(
                self.image_features[idx, :].copy()
            ).float()

        return item

# for eval only
class Full2D(Dataset):
    def __init__(self, side_length=200):
        self.side_length = side_length
        linspace = torch.linspace(-1, 1, side_length)
        self.coordinates = torch.cartesian_prod(linspace, linspace)
    
    def __len__(self):
        return self.side_length ** 2
    
    def __getitem__(self, idx):
        return {
            "idx": idx,
            "coordinates": self.coordinates[idx,:].float(),
        }


class ST3D(Dataset):
    def __init__(self, data_file, require_coordnorm=True, keep_ratio=True, **kwargs):
        super().__init__()

        adata = read_anndata(data_file)

        self.coordinates = adata.obsm["spatial"]
        
        if sp.issparse(adata.X):
            self.raw_representations = adata.X.toarray()
        else:
            self.raw_representations = adata.X
        
        # take statistics
        self.n_cell = self.raw_representations.shape[0]
        self.n_gene = self.raw_representations.shape[1]
        self.n_channels = self.n_gene
        
        if "embeddings" in adata.obsm:
            self.embeddings = adata.obsm["embeddings"]
            assert self.raw_representations.shape[0] == self.embeddings.shape[0]
            self.n_embd = self.embeddings.shape[1]
        else: 
            self.embeddings = None

        if require_coordnorm:
            self._normalize_coordinates(keep_ratio=keep_ratio)

        # self.raw_pca = PCA(n_components=3, random_state=0) # map raw representation dimension to 3 for visualization
        # self.raw_pca.fit(self.raw_representations)

        if self.has_embeddings():
            self.embd_pca = PCA(n_components=3, random_state=0) # map embedding dimension to 3 for visualization
            self.embd_pca.fit(self.embeddings)
        
    
    def has_embeddings(self):
        return False if self.embeddings is None else True

    def plot_raw_representations(self, spot_size=2, train_indices=None, val_indices=None):
        if train_indices and val_indices:
            train_fig = plot_ST(self.coordinates[train_indices,:], self.raw_pca.transform(self.raw_representations[train_indices,:]), spot_size)
            val_fig = plot_ST(self.coordinates[val_indices,:], self.raw_pca.transform(self.raw_representations[val_indices,:]), spot_size)
            return train_fig, val_fig
        else:
            fig = plot_ST(self.coordinates, self.raw_pca.transform(self.raw_representations), spot_size)
        return fig
    
    def plot_embeddings(self, spot_size=2, train_indices=None, val_indices=None):
        assert self.has_embeddings(), "The current adata file has NO embeddings!"
        if train_indices and val_indices:
            train_fig = plot_ST(self.coordinates[train_indices,:], self.embd_pca.transform(self.embeddings[train_indices,:]), spot_size)
            val_fig = plot_ST(self.coordinates[val_indices,:], self.embd_pca.transform(self.embeddings[val_indices,:]), spot_size)
            return train_fig, val_fig
        else:
            fig = plot_ST(self.coordinates, self.embd_pca.transform(self.embeddings), spot_size)
            return fig
    

    # normalize coordinates to [-1.0, +1.0]
    def _normalize_coordinates(self, keep_ratio):
        x_min, y_min, z_min = list(self.coordinates.min(axis=0))
        x_max, y_max, z_max = list(self.coordinates.max(axis=0))
        x_range, y_range, z_range = x_max - x_min, y_max - y_min, z_max - z_min

        self.coordinates[:,0] = (self.coordinates[:,0] - x_min) / x_range
        self.coordinates[:,1] = (self.coordinates[:,1] - y_min) / y_range
        self.coordinates[:,2] = (self.coordinates[:,2] - z_min) / z_range

        self.coordinates -= 0.5
        self.coordinates *= 2.0

        if keep_ratio: # may cause waste of space in the short side
            max_range = max(x_range, y_range, z_range)
            scale_x, scale_y, scale_z = x_range / max_range, y_range / max_range, z_range / max_range
            self.coordinates[:,0] *= scale_x
            self.coordinates[:,1] *= scale_y
            self.coordinates[:,2] *= scale_z
    
    def __len__(self):
        return self.n_cell
    
    def get_raw_dim(self):
        return self.n_gene
    
    def get_embd_dim(self):
        if self.has_embeddings():
            return self.n_embd
        else:
            return None
    
    def __getitem__(self, idx):
        if self.has_embeddings():
            return {
                "idx": idx,
                "coordinates": torch.Tensor(self.coordinates[idx,:].copy()).float(),
                "embeddings": torch.Tensor(self.embeddings[idx,:].copy()).float(),
                "raw_representations": torch.Tensor(self.raw_representations[idx,:].copy()).float()
            }
        else:
            return {
                "idx": idx,
                "coordinates": torch.Tensor(self.coordinates[idx,:].copy()).float(),
                "raw_representations": torch.Tensor(self.raw_representations[idx,:].copy()).float()
            }

class HRSS(Dataset):
    def __init__(self, data_file, **kwargs):
        super().__init__()
        self.data = np.load(data_file)

        if "embeddings" in self.data.keys():
            self.embeddings = self.data["embeddings"]
            self.raw = self.data["raw"]
            self.coordinates = self.data["spatial"]
            self.dataset_length = self.coordinates.shape[0]
            self.n_channels = self.raw.shape[-1]
            self.n_embd = self.embeddings.shape[-1]
            self.embd_pca = PCA(n_components=3, random_state=0)
            self.embd_pca.fit(self.embeddings)

        else:
            self.raw = self.data["raw"]
            
            # switch normalization strategy here
            #self._uint16_normalize()
            self._minmax_normalize()

            # custom slicing of self.raw
            self.raw = self.raw[:,:,::10] # every 10 channels

            self.height, self.width, self.n_channels = self.raw.shape
            self.dataset_length = self.height * self.width
            h_linspace = torch.linspace(-1, 1, self.height) # rows
            w_linspace = torch.linspace(-1, 1, self.width) # columns
            self.coordinates = torch.cartesian_prod(h_linspace, w_linspace)
            self.embeddings = None
            self.raw = rearrange(self.raw, "H W C -> (H W) C")

            

        self.raw_pca = PCA(n_components=3, random_state=0)
        self.raw_pca.fit(self.raw)

    def has_embeddings(self):
        return False if self.embeddings is None else True
    
    def _uint16_normalize(self):
        self.raw = self.raw / 65535

    def _minmax_normalize(self):
        normalized_image = np.zeros_like(self.raw, dtype=float)

        for i in range(self.raw.shape[2]):
            channel = self.raw[:, :, i]
            min_val, max_val = np.min(channel), np.max(channel)
            
            if max_val != min_val:
                # normalized_image[:, :, i] = 2 * (channel - min_val) / (max_val - min_val) - 1
                normalized_image[:, :, i] = (channel - min_val) / (max_val - min_val)
            else:
                normalized_image[:, :, i] = 0
        self.raw = normalized_image
    
    def plot_raw_representations(self, train_indices=None, val_indices=None):
        rgb = self.raw_pca.transform(self.raw)
        if train_indices and val_indices:
            train_fig = plot_ST(self.coordinates[train_indices,:], rgb[train_indices,:], spot_size=1)
            val_fig = plot_ST(self.coordinates[val_indices,:], rgb[val_indices,:], spot_size=1)
            return train_fig, val_fig
        else:
            fig = plot_ST(self.coordinates, rgb)
            return fig
    
    def plot_embeddings(self, spot_size=2, train_indices=None, val_indices=None):
        assert self.has_embeddings(), "The current adata file has NO embeddings!"
        if train_indices and val_indices:
            train_fig = plot_ST(self.coordinates[train_indices,:], self.embd_pca.transform(self.embeddings[train_indices,:]), spot_size)
            val_fig = plot_ST(self.coordinates[val_indices,:], self.embd_pca.transform(self.embeddings[val_indices,:]), spot_size)
            return train_fig, val_fig
        else:
            fig = plot_ST(self.coordinates, self.embd_pca.transform(self.embeddings), spot_size)
            return fig
    
    def __len__(self):
        return self.dataset_length
    
    def get_raw_dim(self):
        return self.n_channels
    
    def get_embd_dim(self):
        if self.has_embeddings():
            return self.n_embd
        else:
            return None

    def __getitem__(self, idx):
        if self.has_embeddings():
            return {
                "idx": idx,
                "coordinates": torch.Tensor(self.coordinates[idx,:].copy()).float(),
                "embeddings": torch.Tensor(self.embeddings[idx,:].copy()).float(),
                "raw_representations": torch.Tensor(self.raw[idx,:].copy()).float()
            }
        else:
            return {
                "idx": idx,
                "coordinates": torch.Tensor(self.coordinates[idx,:]).float(),
                "raw_representations": torch.Tensor(self.raw[idx,:]).float()
            }

from torch.utils.data import Dataset
import torch
import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import PCA

from utils import plot_ST, read_anndata

class GraphST2D(Dataset):
    # 【修改点 1】：在参数列表里增加了 require_coordnorm=True
    def __init__(self, h5ad_file, neighbors, require_coordnorm=True, keep_ratio=True, **kwargs):
        super().__init__()
        #if h5ad_file is not str, adata = h5ad_file
        if type(h5ad_file) == str:

            adata = read_anndata(h5ad_file)
        else:
            adata = h5ad_file
        self.sb = True
        
        self.coordinates = adata.obsm["spatial"]
        self.neighbors = neighbors
        if sp.issparse(adata.X):
            self.raw_representations = adata.X.toarray()
        else:
            self.raw_representations = adata.X
        
        # take statistics
        self.n_cell = self.raw_representations.shape[0]
        self.n_gene = self.raw_representations.shape[1]
        
        if "embeddings" in adata.obsm:
            self.embeddings = adata.obsm["embeddings"]
            assert self.raw_representations.shape[0] == self.embeddings.shape[0]
            self.n_embd = self.embeddings.shape[1]
        else: 
            self.embeddings = None

        # 【修改点 2】：增加 if 判断，避免强制的二次归一化
        if require_coordnorm:
            self._normalize_coordinates(keep_ratio=keep_ratio)

        self.raw_pca = PCA(n_components=3, random_state=0) # map raw representation dimension to 3 for visualization
        self.raw_pca.fit(self.raw_representations)

        if self.has_embeddings():
            self.embd_pca = PCA(n_components=3, random_state=0) # map embedding dimension to 3 for visualization
            self.embd_pca.fit(self.embeddings)
        
    
    def has_embeddings(self):
        return False if self.embeddings is None else True

    def plot_raw_representations(self, spot_size=2, train_indices=None, val_indices=None):
        if train_indices and val_indices:
            train_fig = plot_ST(self.coordinates[train_indices,:], self.raw_pca.transform(self.raw_representations[train_indices,:]), spot_size)
            val_fig = plot_ST(self.coordinates[val_indices,:], self.raw_pca.transform(self.raw_representations[val_indices,:]), spot_size)
            return train_fig, val_fig
        else:
            fig = plot_ST(self.coordinates, self.raw_pca.transform(self.raw_representations), spot_size)
        return fig
    
    def plot_embeddings(self, spot_size=2, train_indices=None, val_indices=None):
        assert self.has_embeddings(), "The current adata file has NO embeddings!"
        if train_indices and val_indices:
            train_fig = plot_ST(self.coordinates[train_indices,:], self.embd_pca.transform(self.embeddings[train_indices,:]), spot_size)
            val_fig = plot_ST(self.coordinates[val_indices,:], self.embd_pca.transform(self.embeddings[val_indices,:]), spot_size)
            return train_fig, val_fig
        else:
            fig = plot_ST(self.coordinates, self.embd_pca.transform(self.embeddings), spot_size)
            return fig
    

    # normalize coordinates to [-1.0, +1.0]
    def _normalize_coordinates(self, keep_ratio):
        x_min, y_min = list(self.coordinates.min(axis=0))
        x_max, y_max = list(self.coordinates.max(axis=0))
        self.coordinates = self.coordinates.astype(np.float64)
        x_range, y_range = x_max - x_min, y_max - y_min
        
        self.coordinates[:,0] = (self.coordinates[:,0] - x_min) / x_range
        
        self.coordinates[:,1] = (self.coordinates[:,1] - y_min) / y_range
     
        self.coordinates -= 0.5
        self.coordinates *= 2.0

        if keep_ratio: # may cause waste of space in the short side
            max_range = max(x_range, y_range)
            scale_x, scale_y = x_range / max_range, y_range / max_range
            self.coordinates[:,0] *= scale_x
            self.coordinates[:,1] *= scale_y
    
    def get_raw_dim(self):
        return self.n_gene
    
    def get_embd_dim(self):
        if self.has_embeddings():
            return self.n_embd
        else:
            return None

    def __len__(self):
        return self.n_cell
    
    def __getitem__(self, idx):
        if self.has_embeddings():
            return {
                "idx": idx,
                "coordinates": torch.Tensor(self.coordinates[idx,:].copy()).float(),
                "embeddings": torch.Tensor(self.embeddings[idx,:].copy()).float(),
                "raw_representations": torch.Tensor(self.raw_representations[idx,:].copy()).float(),
                "neighbors": self.neighbors[idx]
            }
        else:
            return {
                "idx": idx,
                "coordinates": torch.Tensor(self.coordinates[idx,:].copy()).float(),
                "raw_representations": torch.Tensor(self.raw_representations[idx,:].copy()).float(),
                "neighbors": self.neighbors[idx]
            }


if __name__ == "__main__":

    ds = ST2D("./data/preprocessed_data/E11.5.h5ad", True, True)
    raw = ds.raw_representations
    print(raw.max(), raw.min())
    sparsity = (raw==0).sum() / (raw>=0).sum()
    print(sparsity)
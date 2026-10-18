import torch
from torch.utils.data import DataLoader, Subset
import lightning as L
from lightning import pytorch as pl
from sklearn.model_selection import train_test_split
from rich import print
import anndata as ad
import os
import numpy as np
from omegaconf import ListConfig
import importlib
import gc
from datasets import ST2D, GraphST2D
from networks import AE
from utils import metrics, plot_ST, construct_subgraph
from omegaconf import DictConfig
from networks.image_projector import ImageProjector, symmetric_info_nce
     

class ExtraVariableCallback(pl.Callback):
    def __init__(self, extra_variable):
        self.extra_variable = extra_variable  # Store your extra variable

    def on_train_start(self, trainer, pl_module):
        pl_module.extra_variable = self.extra_variable
        print(f"Extra variable passed to LightningModule for training.")

    def on_validation_start(self, trainer, pl_module):
        pl_module.extra_variable = self.extra_variable
        print(f"Extra variable passed to LightningModule for validation.")

    def on_predict_start(self, trainer, pl_module):
        pl_module.extra_variable = self.extra_variable
        print(f"Extra variable passed to LightningModule for prediction.")

    # These methods are required by some stateful callbacks to avoid the "Expected a parent" error
    def state_dict(self):
        # Return an empty dict since no state needs to be saved
        return {}
    
class EmbedderFittingSystem(L.LightningModule):
    def __init__(self, configs, val_pca=None):
        super().__init__()

        self.save_hyperparameters()

        self.pipeline_configs = configs
        network_configs = self.pipeline_configs.embedder

        if network_configs.model == "AE":
            self.GNN = False
            self.fitting_model = AE(
                dim_in=network_configs.dim_in,
                dim_hidden=network_configs.dim_hidden,
                dim_latent=network_configs.dim_latent,
                in_logarithm=network_configs.get("in_logarithm", False),
                dice_lambda=network_configs.get("dice_lambda", 0.01),
            )
        else:
            raise NotImplementedError("The aligned publication pipeline uses AE.")

        self.val_pca = val_pca
        self.output_cache = []

        self.use_alignment = False
        self.image_projector = None

        if hasattr(self.pipeline_configs, "alignment"):
            align_configs = self.pipeline_configs.alignment
            self.use_alignment = align_configs.get("enable", False)

            if self.use_alignment:
                self.image_projector = ImageProjector(
                    image_dim=align_configs.get("image_dim", 1024),
                    latent_dim=network_configs.dim_latent,
                    hidden_dim=align_configs.get("hidden_dim", 512),
                )

                self.align_alpha = align_configs.get("alpha", 0.001)
                self.temperature = align_configs.get("temperature", 0.1)
                self.detach_gene_for_align = align_configs.get(
                    "detach_gene_for_align", False
                )
            else:
                self.align_alpha = 0.0
                self.temperature = 0.1
                self.detach_gene_for_align = False
        else:
            self.align_alpha = 0.0
            self.temperature = 0.1
            self.detach_gene_for_align = False
        
    
    def forward(self, x):
        recon, embd = self.fitting_model(x)
        return recon, embd
    
    def training_step(self, batch, batch_idx):
        if self.GNN:
            idx, neighbors = batch["idx"],batch["neighbors"]
            extra_info = self.extra_variable
            adj = extra_info["adj_train"]
            raw_rep = extra_info["raw_rep_train"]
            neighbours = list(neighbors.cpu().numpy().flatten())
            neighbourhoods = list(set(neighbours))
            sub_set_y_raw, sub_set_adj,sub_set_idx = construct_subgraph(raw_rep, adj, neighbourhoods,idx.cpu().numpy())
            sub_set_y_raw = sub_set_y_raw.cuda()
            sub_set_adj = sub_set_adj.cuda()
            loss, _, _ = self.fitting_model.forward_loss(sub_set_y_raw,sub_set_adj,sub_set_idx)
            del adj
            del raw_rep
            gc.collect()
        else:
            y_raw = batch["raw_representations"]

            recon_loss, y_hat, z_st, loss_dict = self.fitting_model.forward_loss(y_raw)

            loss = recon_loss

            self.log("train/recon_loss", loss_dict["recon_loss"], on_step=False, on_epoch=True)
            self.log("train/mse_loss", loss_dict["mse_loss"], on_step=False, on_epoch=True)
            self.log("train/dice_loss", loss_dict["dice_loss"], on_step=False, on_epoch=True)

            if self.use_alignment:
                if "image_features" not in batch:
                    raise KeyError(
                        "Alignment is enabled, but batch does not contain 'image_features'. "
                        "Please make sure ST2D reads adata.obsm['phikon_cls']."
                    )

                h_img = batch["image_features"]
                z_img = self.image_projector(h_img)

                z_for_contra = z_st.detach() if self.detach_gene_for_align else z_st

                contra_loss = symmetric_info_nce(
                    z_st=z_for_contra,
                    z_img=z_img,
                    temperature=self.temperature,
                )

                loss = loss + self.align_alpha * contra_loss

                self.log("train/contra_loss", contra_loss, on_step=False, on_epoch=True)
                self.log("train/total_loss", loss, on_step=False, on_epoch=True)
        self.log("train/loss", loss, on_step=False, on_epoch=True)
        torch.cuda.empty_cache()
        return loss

    def validation_step(self, batch, batch_idx):
        if self.GNN:
            y_raw, x,idx,neighbors = batch["raw_representations"], batch["coordinates"], batch["idx"],batch["neighbors"]
            extra_info = self.extra_variable
            adj = extra_info["adj_val"]
            raw_rep = extra_info["raw_rep_val"]
            neighbours = list(neighbors.cpu().numpy().flatten()) 
            neighbourhoods = list(set(neighbours))
            sub_set_y_raw, sub_set_adj,sub_set_idx = construct_subgraph(raw_rep, adj, neighbourhoods,idx.cpu().numpy())
            sub_set_y_raw = sub_set_y_raw.cuda()
            sub_set_adj = sub_set_adj.cuda()
            loss, y_hat, _ = self.fitting_model.forward_loss(sub_set_y_raw,sub_set_adj,sub_set_idx)

        else:
            y_raw, x = batch["raw_representations"], batch["coordinates"]

            recon_loss, y_hat, z_st, loss_dict = self.fitting_model.forward_loss(y_raw)

            loss = recon_loss

            self.log("val/recon_loss", loss_dict["recon_loss"], on_step=False, on_epoch=True)
            self.log("val/mse_loss", loss_dict["mse_loss"], on_step=False, on_epoch=True)
            self.log("val/dice_loss", loss_dict["dice_loss"], on_step=False, on_epoch=True)

            if self.use_alignment:
                if "image_features" not in batch:
                    raise KeyError(
                        "Alignment is enabled, but batch does not contain 'image_features'."
                    )

                h_img = batch["image_features"]
                z_img = self.image_projector(h_img)

                z_for_contra = z_st.detach() if self.detach_gene_for_align else z_st

                contra_loss = symmetric_info_nce(
                    z_st=z_for_contra,
                    z_img=z_img,
                    temperature=self.temperature,
                )

                loss = loss + self.align_alpha * contra_loss

                self.log("val/contra_loss", contra_loss, on_step=False, on_epoch=True)
                self.log("val/total_loss", loss, on_step=False, on_epoch=True)
        self.log("val/loss", loss, on_step=False, on_epoch=True)
        self.output_cache.append({
            "y_raw": y_raw.detach().cpu().numpy(),
            "y_hat": y_hat.detach().cpu().numpy(),
            "x": x.detach().cpu().numpy()
        })

    def on_validation_epoch_end(self):
        outputs = self.output_cache
        all_y_raw = np.concatenate([x["y_raw"] for x in outputs], axis=0)
        all_y_hat = np.concatenate([x["y_hat"] for x in outputs], axis=0)
        all_x = np.concatenate([x["x"] for x in outputs], axis=0)
        
        scores = metrics(all_y_raw, all_y_hat)
        self.log_dict(scores, sync_dist=False)
        fig = plot_ST(all_x, self.val_pca.transform(all_y_hat))
        self.logger.experiment.add_figure("val/y_hat", fig, self.current_epoch)
        print(scores)

        # Don't forget to clear the memory for the next epoch!
        self.output_cache.clear()


    def configure_optimizers(self):
        param_list = list(self.fitting_model.parameters())

        if self.use_alignment and self.image_projector is not None:
            param_list += list(self.image_projector.parameters())

        optimizer = torch.optim.Adam(
            param_list,
            lr=self.pipeline_configs.optimization.lr,
        )
        return optimizer

    def predict_step(self, batch, batch_idx):
        if self.GNN:
            y_raw, x,neighbors,idx  = batch["raw_representations"], batch["coordinates"],batch["neighbors"], batch["idx"]
            extra_info = self.extra_variable
            adj = extra_info["adj"]
            raw_rep = extra_info["raw_rep"]
            
            neighbours = list(neighbors.cpu().numpy().flatten()) 
            neighbourhoods = list(set(neighbours))
            sub_set_y_raw, sub_set_adj,sub_set_idx = construct_subgraph(raw_rep, adj, neighbourhoods,idx.cpu().numpy())
            sub_set_y_raw = sub_set_y_raw.cuda()
            sub_set_adj = sub_set_adj.cuda()
            y_hat, embd = self.fitting_model.generate(sub_set_y_raw, sub_set_adj, sub_set_idx)
            output = {
                "x": x.detach().cpu().numpy(),
                "y_raw": y_raw.detach().cpu().numpy(),
                "y_hat": y_hat.detach().cpu().numpy(),
                "embd": embd.detach().cpu().numpy(),
            }
            self.output_cache.append(output)
            return

        else:
            y_raw, x = batch["raw_representations"], batch["coordinates"]
            y_hat, embd = self.fitting_model(y_raw)

            output = {
                "x": x.detach().cpu().numpy(),
                "y_raw": y_raw.detach().cpu().numpy(),
                "y_hat": y_hat.detach().cpu().numpy(),
                "embd": embd.detach().cpu().numpy(),
            }

            if self.use_alignment and "image_features" in batch:
                h_img = batch["image_features"]
                z_local = self.image_projector(h_img)
                output["z_local"] = z_local.detach().cpu().numpy()

            self.output_cache.append(output)
            return
        
    def on_predict_epoch_end(self):
        from scipy.sparse import csr_matrix
        outputs = self.output_cache
        all_y_raw = np.concatenate([x["y_raw"] for x in outputs], axis=0)
        all_y_raw = csr_matrix(all_y_raw)
        #all_y_hat = np.concatenate([x["y_hat"] for x in outputs], axis=0)
        all_x = np.concatenate([x["x"] for x in outputs], axis=0)
        all_embd = np.concatenate([x["embd"] for x in outputs], axis=0)
        has_z_local = "z_local" in outputs[0]
        if has_z_local:
            all_z_local = np.concatenate([x["z_local"] for x in outputs], axis=0)
        else:
            all_z_local = None
        del outputs
        del self.output_cache
        gc.collect()
        if isinstance(self.pipeline_configs.embedded_data, ListConfig):
            save_name = self.pipeline_configs.embedded_data.pop(0)
        else:
            save_name = self.pipeline_configs.embedded_data

        # Keep the stage-1 artifact in a stable project-level location when
        # requested by the experiment config. This lets stage 2 consume the
        # output without manually copying it out of a Lightning version dir.
        configured_output = self.pipeline_configs.get("embedded_output", None)
        log_file = os.path.join(self.logger.log_dir, save_name)
        output_files = [str(configured_output)] if configured_output else []
        if log_file not in output_files:
            output_files.append(log_file)
        for output_file in output_files:
            os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        print(f"Writing stage-1 output to {output_files} ... It may take some time ...")
        if save_name.endswith(".h5ad"):
            # 1. 直接读取原始文件，继承完整的 metadata (如 split 列)
            original_data_file = self.original_data_file
            embedded_data = ad.read_h5ad(original_data_file)
            
            # 2. 覆盖特征和坐标
            embedded_data.X = all_y_raw
            embedded_data.obsm["spatial"] = all_x
            embedded_data.obsm["embeddings"] = all_embd

            if all_z_local is not None:
                embedded_data.obsm["z_local"] = all_z_local

            # 3. 保存带有完整 obs 信息的预测文件。项目级路径供 stage 2 使用，
            # Lightning 日志路径保留一份便于审计单次运行。
            for output_file in output_files:
                embedded_data.write(output_file)
        elif save_name.endswith(".npz"):
            for output_file in output_files:
                np.savez(output_file, spatial=all_x, raw=all_y_raw, embeddings=all_embd)
        


def train_embedder(configs):
    dataset_configs = configs.dataset
    pipeline_configs = configs.pipeline
    
    pl.seed_everything(pipeline_configs.optimization.seed, workers=True) # fix seed globally
    torch.set_float32_matmul_precision("highest") # make use of tensorcore
    if pipeline_configs.embedder.model != "AE":
        raise NotImplementedError("The aligned publication pipeline uses AE.")

    # The publication release intentionally executes only the aligned AE path.
    # The legacy graph branch is retained below solely for provenance and is unreachable.
    if False:
        import scanpy as sc
        from scipy.sparse import issparse
        from sklearn.neighbors import kneighbors_graph
        
        n_neighbors = configs.dataset.n_neighbors
        adata =  sc.read_h5ad(dataset_configs.data_file)
        
        adj = kneighbors_graph(adata.obsm['spatial'],n_neighbors,mode='connectivity',n_jobs=8,include_self=True)
        adj = adj.astype(np.float32)
        if dataset_configs.type == 'GraphST2D':
            dataset = GraphST2D(h5ad_file=dataset_configs.data_file, neighbors=adj.indices.reshape(-1,n_neighbors),  keep_ratio=dataset_configs.keep_ratio)
        elif dataset_configs.type == 'GraphST3D':
            dataset = GraphST3D(h5ad_file=dataset_configs.data_file, neighbors=adj.indices.reshape(-1,n_neighbors), keep_ratio=dataset_configs.keep_ratio,require_coordnorm=dataset_configs.require_coordnorm)
        if issparse(adata.X):
            adata.X = adata.X.toarray()
        raw_rep = torch.tensor(adata.X).float()
        if "split" in adata.obs:
            train_idx = np.where(adata.obs["split"] == "train")[0].tolist()
            val_idx = np.where(adata.obs["split"] == "val")[0].tolist()
            print("[green]GAE using fixed train/held-out evaluation split from adata.obs['split'][/green]")
        else:
            train_idx, val_idx = train_test_split(
                list(range(len(adata))), 
                test_size=dataset_configs.val_proportion, 
                random_state=pipeline_configs.optimization.seed
            )
            print("[yellow]GAE using a deterministic random 80/20 split (no fixed split found)[/yellow]")
        adata_train = adata[train_idx]
        adata_val = adata[val_idx]
        print(len(adata_train),len(adata_val))
        print(f"{val_idx[:10]=}") # check whether seed works
        adj_train = kneighbors_graph(adata[train_idx].obsm['spatial'],n_neighbors,mode='connectivity',n_jobs=8,include_self=True)
        adj_val = kneighbors_graph(adata[val_idx].obsm['spatial'],n_neighbors,mode='connectivity',n_jobs=8,include_self=True)
        if dataset_configs.type == 'GraphST2D':
            train_dataset = GraphST2D(h5ad_file=adata_train,neighbors=adj_train.indices.reshape(-1,n_neighbors) ,keep_ratio=dataset_configs.keep_ratio)
            val_dataset = GraphST2D(h5ad_file=adata_val,neighbors=adj_val.indices.reshape(-1,n_neighbors) , keep_ratio=dataset_configs.keep_ratio)

        batch_size = pipeline_configs.optimization.batch_size
        raw_rep_train = torch.tensor(adata[train_idx].X).float()
        raw_rep_val = torch.tensor(adata[val_idx].X).float()
        adj_train = adj_train.astype(np.float32)
        adj_val = adj_val.astype(np.float32)

        extra_info = {"raw_rep":raw_rep,"adj": adj,"raw_rep_train": raw_rep_train, "adj_train":adj_train, "raw_rep_val":raw_rep_val, "adj_val":adj_val}
        pipeline_configs.embedder.dim_in = train_dataset.n_gene
        extra_callback = ExtraVariableCallback(extra_variable=extra_info)
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=18, drop_last=False)
        val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=18, drop_last=False)

    else:
        # dataset configuration
        dataset_class = getattr(importlib.import_module("datasets"), dataset_configs.type)
        dataset = dataset_class(**dataset_configs)

        pipeline_configs.embedder.dim_in = dataset.get_raw_dim()

        print("Fitting embedder ...")

        # Use the fixed train/held-out-evaluation split from adata.obs["split"]
        # when available. The internal ``val`` label is retained for compatibility.
        # This is required to avoid data leakage.
        if hasattr(dataset, "has_split") and dataset.has_split():
            train_idx, val_idx = dataset.get_train_val_indices()
            print("[green]Using fixed train/held-out evaluation split from adata.obs['split'][/green]")
        else:
            train_idx, val_idx = train_test_split(
                list(range(len(dataset))),
                test_size=dataset_configs.val_proportion,
                random_state=pipeline_configs.optimization.seed,
                shuffle=True,
            )
            print("[yellow]No fixed split found. Falling back to a deterministic random 80/20 split.[/yellow]")

        print(len(train_idx), len(val_idx))
        print(f"{val_idx[:10]=}")

        train_dataset, val_dataset = Subset(dataset, train_idx), Subset(dataset, val_idx)
        

        batch_size = pipeline_configs.optimization.batch_size
        num_workers = int(pipeline_configs.optimization.get("num_workers", 0))
        train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, drop_last=False)
        val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers, drop_last=False)
        extra_callback = ExtraVariableCallback(extra_variable=None)
    # pipeline configuration
    fitting_system = EmbedderFittingSystem(pipeline_configs, val_pca=dataset.raw_pca)
    
    tb_logger = pl.loggers.TensorBoardLogger(pipeline_configs.optimization.logs)

    # log raw representations to tensorboard
    train_raw_fig, val_raw_fig = dataset.plot_raw_representations(train_indices=train_idx, val_indices=val_idx)
    tb_logger.experiment.add_figure("dataset/train_raw", train_raw_fig)
    tb_logger.experiment.add_figure("dataset/val_raw", val_raw_fig)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val/loss",
        mode="min",
        filename="best-{epoch:03d}",
        auto_insert_metric_name=False,
        save_top_k=1,
        save_last=True,
    )

    trainer = L.Trainer(
                #num_sanity_val_steps=0,
                max_epochs=pipeline_configs.optimization.epochs,
                check_val_every_n_epoch=pipeline_configs.optimization.val_freq,
                log_every_n_steps=1, 
                logger=tb_logger, 
                callbacks=[extra_callback, checkpoint_callback],
                devices=1
            )
    trainer.fit(fitting_system, train_dataloader, val_dataloader)
    best_ckpt = checkpoint_callback.best_model_path

    print(f"[INFO] Best checkpoint: {best_ckpt}")
    print(f"[INFO] Best val loss: {checkpoint_callback.best_model_score}")

    if best_ckpt is not None and best_ckpt != "":
        print("[INFO] Loading best checkpoint for prediction...")

        fitting_system = EmbedderFittingSystem.load_from_checkpoint(
            best_ckpt,
            configs=pipeline_configs,
            val_pca=dataset.raw_pca,
            weights_only=False,
        )

    fitting_system.original_data_file = dataset_configs.data_file
    # del trainer
    if dataset_configs.type == 'GraphST2D':
        del adj_train
        del adj
        del raw_rep
        del raw_rep_train
    del train_dataset
    del train_dataloader
    del extra_callback
    import gc
    gc.collect()
    # predict
    if pipeline_configs.predict_mode=="all":
        del val_dataset
        del val_dataloader
        gc.collect()
        test_dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers, drop_last=False)
        trainer.predict(fitting_system, dataloaders=test_dataloader)
    elif pipeline_configs.predict_mode=="val":
        test_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers, drop_last=False)
        trainer.predict(fitting_system, dataloaders=test_dataloader)
    elif pipeline_configs.predict_mode=="resample":
        test_dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers, drop_last=False)
        trainer.predict(fitting_system, dataloaders=test_dataloader)
        test_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers, drop_last=False)
        trainer.predict(fitting_system, dataloaders=test_dataloader)
    

    else:
        print("End without writing predictions.")

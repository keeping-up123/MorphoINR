import torch
import lightning as L
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import anndata as ad
from rich import print
from lightning import pytorch as pl
from sklearn.model_selection import train_test_split
import numpy as np
import os
import glob
import importlib
import torch.nn as nn
from networks.siren import SirenNet
from utils import metrics, plot_ST
from systems.embedder_fitting_system import EmbedderFittingSystem
from datasets import ST2D


class INRFittingSystem(L.LightningModule):
    def __init__(self, configs, val_pca, decoder=None):
        super().__init__()

        self.save_hyperparameters(ignore=['decoder'])
        
        self.pipeline_configs = configs
        network_configs = self.pipeline_configs.inr

        final_act = "ReLU" if self.pipeline_configs.target == "raw_representations" else "Identity"

        if network_configs.model == "SIREN":
            self.fitting_model = SirenNet(
                    dim_in=network_configs.dim_in,
                    dim_hidden=network_configs.num_hidden_features, 
                    dim_out=network_configs.dim_out, 
                    num_layers=network_configs.num_hidden_layers, 
                    final_activation=final_act
                )
        else:
            raise NotImplementedError("The aligned publication pipeline uses SIREN.")
        
        self.val_pca = val_pca
        self.decoder = decoder

        self.output_cache = []

        # Optional image-conditioned FiLM/gating module.
        # mode = "film_output":
        #     coord-only SIREN predicts z_coord
        #     z_local predicts gamma/beta
        #     z_hat = z_coord * (1 + gamma) + beta
        self.conditioning_mode = "none"
        self.film_mlp = None

        if hasattr(self.pipeline_configs.inr, "conditioning"):
            self.conditioning_mode = self.pipeline_configs.inr.conditioning.get("mode", "none")

        if self.conditioning_mode == "film_output":
            local_dim = self.pipeline_configs.inr.local_dim
            latent_dim = network_configs.dim_out
            hidden_dim = self.pipeline_configs.inr.conditioning.get("hidden_dim", 256)
            dropout = self.pipeline_configs.inr.conditioning.get("dropout", 0.0)

            self.film_mlp = nn.Sequential(
                nn.Linear(local_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 2 * latent_dim),
            )

            # Important: initialize last layer to zero.
            # Then at epoch 0:
            # gamma = 0, beta = 0, so z_hat = z_coord.
            # This makes FiLM start exactly from coord-only behavior.
            nn.init.zeros_(self.film_mlp[-1].weight)
            nn.init.zeros_(self.film_mlp[-1].bias)

            print(
                f"[green]Using output-level FiLM conditioning: "
                f"local_dim={local_dim}, latent_dim={latent_dim}, hidden_dim={hidden_dim}[/green]"
            )

    def is_decoder_stage(self):
        phase = self.pipeline_configs.inr.phase
        return phase is not None and self.current_epoch > phase


    def set_module_trainable(self, module, trainable: bool):
        if module is None:
            return
        for p in module.parameters():
            p.requires_grad_(trainable)


    def on_train_epoch_start(self):
        """
        Stage 1:
            train INR + FiLM to fit embeddings / raw representations.

        Stage 2:
            freeze INR + FiLM, train decoder only.
        """
        decoder_stage = self.is_decoder_stage()

        if decoder_stage:
            # freeze coordinate INR backbone
            self.set_module_trainable(self.fitting_model, False)

            # freeze FiLM MLP as part of the INR-side mapping
            self.set_module_trainable(self.film_mlp, False)

            # only decoder is trainable
            if self.decoder is not None and self.pipeline_configs.inr.decoder.finetune:
                self.set_module_trainable(self.decoder, True)

        else:
            # INR fitting stage: train INR backbone
            self.set_module_trainable(self.fitting_model, True)

            # train FiLM if it exists
            self.set_module_trainable(self.film_mlp, True)

            # decoder should not be trained before decoder stage
            self.set_module_trainable(self.decoder, False)

    def forward(self, x):
        output = self.fitting_model(x)
        return output

    def forward_batch(self, batch):
        """
        Forward INR from batch.

        Modes:
        - none / concat:
            use prepare_inr_input(batch)
        - film_output:
            SIREN only takes coordinates.
            z_local modulates the output latent.
        """
        if self.conditioning_mode == "film_output":
            if "local_features" not in batch:
                raise RuntimeError(
                    "conditioning.mode='film_output' requires batch['local_features']. "
                    "Please set dataset.local_key: z_local in the config."
                )

            coords = batch["coordinates"]
            z_local = batch["local_features"]

            z_coord = self.fitting_model(coords)

            gamma_beta = self.film_mlp(z_local)
            gamma, beta = torch.chunk(gamma_beta, chunks=2, dim=-1)

            pred = z_coord * (1.0 + gamma) + beta
            return pred

        else:
            x = self.prepare_inr_input(batch)
            pred = self.fitting_model(x)
            return pred

    def prepare_inr_input(self, batch):
        """
        Prepare input for INR.

        Original SUICA:
            input = coordinates, shape [B, 2]

        Our image-conditioned version:
            input = concat(coordinates, local_features), shape [B, 2 + D_local]
        """
        x = batch["coordinates"]

        if "local_features" in batch:
            x = torch.cat([x, batch["local_features"]], dim=-1)
 
        return x
    
    
    def dice_loss(self, pred, gt, eps=1.0):
        """
        [0, 1] Dice loss for zero / non-zero expression pattern.

        gt:
            log-normalized expression, non-negative.
            gt > 0 means this gene is expressed.

        pred:
            decoder output, expected to be non-negative.
            We convert it to a soft non-zero probability in [0, 1].
        """
        gt_binary = (gt > 0).float()

        # pred = 0 -> 0
        # pred larger -> close to 1
        pred_prob = 1.0 - torch.exp(-torch.clamp(pred, min=0.0))

        intersection = (pred_prob * gt_binary).sum()
        denominator = pred_prob.sum() + gt_binary.sum()

        dice_coeff = (2.0 * intersection + eps) / (denominator + eps)
        return 1.0 - dice_coeff
    
    def recons_loss(self, pred, gt):
        # MSE for non-zero elements
        nonzero_mask = gt != 0
        if nonzero_mask.any():
            mse_loss = F.mse_loss(pred[nonzero_mask], gt[nonzero_mask])
        else:
            mse_loss = F.mse_loss(pred, gt)

        # L1 loss to encourage sparsity
        l1_loss = F.l1_loss(pred, gt)

        dice = self.dice_loss(pred, gt)

        dice_lambda = getattr(
            self.pipeline_configs.inr.decoder,
            "dice_lambda",
            0.01,
        )

        return mse_loss + l1_loss + dice_lambda * dice
    
    def fitting_loss(self, pred, gt):
        return F.l1_loss(pred,gt) + F.mse_loss(pred,gt)

    def training_step(self, batch, batch_idx):
        if self.pipeline_configs.target == "embeddings":
            gt = batch["embeddings"]
        elif self.pipeline_configs.target == "raw_representations":
            gt = batch["raw_representations"]
        else:
            raise NotImplementedError

        decoder_stage = self.is_decoder_stage()

        # Stage 2: INR + FiLM 已经冻结，只需要给 decoder 一个固定输入
        if decoder_stage:
            with torch.no_grad():
                pred = self.forward_batch(batch)
        else:
            pred = self.forward_batch(batch)

        # Stage 1: train INR / FiLM to fit target embeddings
        if not decoder_stage:
            fitting_loss = self.fitting_loss(pred, gt)
        else:
            fitting_loss = torch.tensor(0.0, device=self.device)

        self.log(
            "train/fitting_loss",
            fitting_loss,
            sync_dist=True,
            rank_zero_only=True,
            on_step=False,
            on_epoch=True,
        )

        loss = fitting_loss

        # Stage 2: decoder-only fine-tuning
        if (
            self.decoder is not None
            and self.pipeline_configs.inr.decoder.recon_loss
            and decoder_stage
        ):
            recons = self.decoder(pred)
            recons_loss = self.recons_loss(recons, batch["raw_representations"])

            self.log(
                "train/recons_loss",
                recons_loss,
                sync_dist=True,
                rank_zero_only=True,
                on_step=False,
                on_epoch=True,
            )

            loss = loss + recons_loss

        self.log(
            "train/loss",
            loss,
            sync_dist=True,
            rank_zero_only=True,
            on_step=False,
            on_epoch=True,
        )

        return loss
    
    def validation_step(self, batch, batch_idx):
        results = {}

        if self.pipeline_configs.target == "embeddings":
            gt = batch["embeddings"]
            pred = self.forward_batch(batch)
            x = batch["coordinates"]

            fitting_loss = self.fitting_loss(pred, gt)
            loss = fitting_loss

            recons_loss = None

            if self.decoder:
                raw = batch["raw_representations"]
                recons = self.decoder(pred)
                recons_loss = self.recons_loss(recons, raw)

                self.log("val/recons_loss", recons_loss, on_step=False, on_epoch=True)

                results["raw"] = raw.detach().cpu().numpy()
                results["recons"] = recons.detach().cpu().numpy()

        else:
            gt = batch["raw_representations"]
            pred = self.forward_batch(batch)
            x = batch["coordinates"]

            fitting_loss = F.mse_loss(pred, gt)
            recons_loss = None
            loss = fitting_loss

        phase = self.pipeline_configs.inr.phase

        if phase is not None:
            if self.current_epoch <= phase:
                # INR latent fitting stage
                val_inr_phase_loss = fitting_loss
                val_decoder_phase_loss = torch.tensor(1e9, device=self.device)
                loss = fitting_loss
            else:
                # Decoder fine-tuning stage
                val_inr_phase_loss = torch.tensor(1e9, device=self.device)

                if recons_loss is not None:
                    val_decoder_phase_loss = recons_loss
                    loss = recons_loss
                else:
                    val_decoder_phase_loss = torch.tensor(1e9, device=self.device)
                    loss = fitting_loss
        else:
            val_inr_phase_loss = fitting_loss
            val_decoder_phase_loss = torch.tensor(1e9, device=self.device)
            loss = fitting_loss

        self.log("val/fitting_loss", fitting_loss, on_step=False, on_epoch=True)
        self.log("val/inr_phase_loss", val_inr_phase_loss, on_step=False, on_epoch=True)
        self.log("val/decoder_phase_loss", val_decoder_phase_loss, on_step=False, on_epoch=True)
        self.log("val/loss", loss, on_step=False, on_epoch=True)

        results["x"] = x.detach().cpu().numpy()
        results["pred"] = pred.detach().cpu().numpy()
        results["gt"] = gt.detach().cpu().numpy()

        self.output_cache.append(results)

    def on_validation_epoch_end(self):
        outputs = self.output_cache
        all_gt = np.concatenate([x["gt"] for x in outputs], axis=0)
        all_pred = np.concatenate([x["pred"] for x in outputs], axis=0)
        all_x = np.concatenate([x["x"] for x in outputs], axis=0)
        
        scores_fitting = metrics(all_gt, all_pred, prefix="val_fitting",fast=False)
        self.log_dict(scores_fitting, sync_dist=False)
        print(scores_fitting)

        if self.decoder:
            all_raw = np.concatenate([x["raw"] for x in outputs], axis=0)
            all_recons = np.concatenate([x["recons"] for x in outputs], axis=0)
            scores_recon = metrics(all_raw, all_recons, prefix="val_recon", fast=False)
            self.log_dict(scores_recon, sync_dist=False)
            # Save self.decoder
            decoder_path = os.path.join(self.logger.log_dir, "decoder.pth")
            torch.save(self.decoder.state_dict(), decoder_path)
            print(f"Decoder saved to {decoder_path}")
            
            print(scores_recon)
        # fig = plot_ST(all_x, self.val_pca.transform(all_pred))
        # self.logger.experiment.add_figure("val/pred", fig, self.current_epoch)
        
        self.output_cache.clear()

    def configure_optimizers(self):
        param_list = list(self.fitting_model.parameters())

        if self.film_mlp is not None:
            param_list += list(self.film_mlp.parameters())

        if (
            self.decoder is not None
            and self.pipeline_configs.inr.decoder.finetune
        ):
            param_list += list(self.decoder.parameters())

        optimizer = torch.optim.AdamW(
            param_list,
            lr=self.pipeline_configs.optimization.lr,
        )
        return optimizer

    def predict_step(self, batch, batch_idx):
        x, raw = batch["coordinates"], batch["raw_representations"]
        pred = self.forward_batch(batch)
        step_output = {}
        if self.pipeline_configs.target == "embeddings":
            embd = batch["embeddings"] 
            step_output["embd"] = embd.detach().cpu().numpy()
            if self.decoder:
                recons = self.decoder(pred)
                step_output["recons"] = recons.detach().cpu().numpy()
        
        step_output["x"] = x.detach().cpu().numpy()
        step_output["raw"] = raw.detach().cpu().numpy()
        step_output["pred"] = pred.detach().cpu().numpy()
        self.output_cache.append(step_output)

    def on_predict_epoch_end(self):
        outputs = self.output_cache
        all_raw = np.concatenate([x["raw"] for x in outputs], axis=0)
        all_pred = np.concatenate([x["pred"] for x in outputs], axis=0)
        all_x = np.concatenate([x["x"] for x in outputs], axis=0)

        adata = ad.AnnData(X=all_raw)
        adata.obsm["spatial"] = all_x

        if self.pipeline_configs.target == "raw_representations":
            adata.obsm["fitted_raw"] = all_pred
        elif self.pipeline_configs.target == "embeddings":
            if self.decoder:
                all_recons = np.concatenate([x["recons"] for x in outputs], axis=0)
                adata.obsm["reconstructed_raw"] = all_recons
            all_embd = np.concatenate([x["embd"] for x in outputs], axis=0)
            adata.obsm["embeddings"] = all_embd
            adata.obsm["fitted_embd"] = all_pred

        write_file = os.path.join(self.logger.log_dir, self.pipeline_configs.reconstructed_data)
        print("Saving prediction to:", write_file)
        print("Prediction shape:", adata.shape)
        adata.write_h5ad(write_file)
        

def train_inr(configs):
    dataset_configs = configs.dataset
    pipeline_configs = configs.pipeline

    pl.seed_everything(pipeline_configs.optimization.seed, workers=True) # fix seed globally
    torch.set_float32_matmul_precision("highest") # make use of tensorcore, highest = default pytorch

    # dataset configuration
    dataset_class = getattr(importlib.import_module("datasets"), dataset_configs.type)
    dataset = dataset_class(**dataset_configs)
    assert pipeline_configs.target in ["embeddings", "raw_representations"]
    
    conditioning_mode = "none"
    if hasattr(pipeline_configs.inr, "conditioning"):
        conditioning_mode = pipeline_configs.inr.conditioning.get("mode", "none")

    if conditioning_mode == "film_output":
        # SIREN only takes coordinates.
        pipeline_configs.inr.dim_in = 2

        if not (hasattr(dataset, "has_local_features") and dataset.has_local_features()):
            raise RuntimeError(
                "conditioning.mode='film_output' requires local features. "
                "Please set dataset.local_key: z_local."
            )

        pipeline_configs.inr.local_dim = dataset.get_local_dim()

    else:
        # Original behavior: concat local features if available.
        if hasattr(dataset, "has_local_features") and dataset.has_local_features():
            pipeline_configs.inr.dim_in = 2 + dataset.get_local_dim()
        else:
            pipeline_configs.inr.dim_in = 2

    print(f"[INFO] INR dim_in = {pipeline_configs.inr.dim_in}")
    if hasattr(pipeline_configs.inr, "local_dim"):
        print(f"[INFO] local_dim = {pipeline_configs.inr.local_dim}")
    print(f"[INFO] conditioning_mode = {conditioning_mode}")
        
    if pipeline_configs.target == "embeddings":
        pipeline_configs.inr.dim_out = dataset.get_embd_dim()
        # val_pca = dataset.embd_pca
    elif pipeline_configs.target == "raw_representations":
        pipeline_configs.inr.dim_out = dataset.get_raw_dim()
        # val_pca = dataset.raw_pca


    print(f"Fitting ST [italic red]{pipeline_configs.target}[/italic red] with INR ...")

    # Use the fixed train/held-out-evaluation split from adata.obs["split"]
    # when available. The internal ``val`` label is retained for compatibility.
    # This avoids data leakage and keeps all experiments comparable.
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
    num_batches_in_epoch = len(train_dataloader)

    # pipeline configuration
    if pipeline_configs.inr.decoder:
        ckpt = pipeline_configs.inr.decoder.ckpt

        # TensorBoardLogger increments version_0 to version_1, version_2, ...
        # when a case is rerun. Keep the explicit config path for archival
        # runs, but fall back to the newest AE checkpoint if that exact path
        # is no longer present.
        if not os.path.isfile(ckpt):
            lightning_root = os.path.dirname(os.path.dirname(os.path.dirname(ckpt)))
            candidates = glob.glob(
                os.path.join(lightning_root, "version_*", "checkpoints", "best-*.ckpt")
            )
            if candidates:
                ckpt = max(candidates, key=os.path.getmtime)
                print(f"[yellow]Configured decoder checkpoint not found; using {ckpt}[/yellow]")
            else:
                raise FileNotFoundError(
                    f"Decoder checkpoint not found: {pipeline_configs.inr.decoder.ckpt}. "
                    f"Run the aligned AE stage first."
                )

        print(f"[INFO] Loading decoder from: {ckpt}")

        decoder_module = EmbedderFittingSystem.load_from_checkpoint(
            ckpt,
            weights_only=False,
        )
        decoder = decoder_module.fitting_model.decoder
        print("[yellow]with pretrained decoder[/yellow]")
    else:
        decoder = None
        pipeline_configs.inr.decoder = {"recon_loss": False, "finetune": False}
        print("[yellow]with no decoder[/yellow]")

    fitting_system = INRFittingSystem(pipeline_configs, val_pca=None, decoder=decoder)
    tb_logger = pl.loggers.TensorBoardLogger(pipeline_configs.optimization.logs)
    
    checkpoint_inr_callback = pl.callbacks.ModelCheckpoint(
        monitor="val/inr_phase_loss",
        mode="min",
        filename="best-inr-{epoch:03d}",
        auto_insert_metric_name=False,
        save_top_k=1,
        save_last=False,
    )

    checkpoint_decoder_callback = pl.callbacks.ModelCheckpoint(
        monitor="val/decoder_phase_loss",
        mode="min",
        filename="best-decoder-{epoch:03d}",
        auto_insert_metric_name=False,
        save_top_k=1,
        save_last=True,
    )
    trainer = L.Trainer(
                #num_sanity_val_steps=0,
                accumulate_grad_batches=num_batches_in_epoch,
                max_epochs=pipeline_configs.optimization.epochs,
                check_val_every_n_epoch=pipeline_configs.optimization.val_freq,
                log_every_n_steps=1, 
                logger=tb_logger, 
                callbacks=[checkpoint_inr_callback, checkpoint_decoder_callback],
                devices=1
            )
    trainer.fit(fitting_system, train_dataloader, val_dataloader)
    print(f"[green]Best INR checkpoint:[/green] {checkpoint_inr_callback.best_model_path}")
    print(f"[green]Best INR val loss:[/green] {checkpoint_inr_callback.best_model_score}")

    print(f"[green]Best Decoder checkpoint:[/green] {checkpoint_decoder_callback.best_model_path}")
    print(f"[green]Best Decoder val loss:[/green] {checkpoint_decoder_callback.best_model_score}")
    
    best_predict_ckpt = checkpoint_decoder_callback.best_model_path

    if best_predict_ckpt is not None and best_predict_ckpt != "":
        print(
            f"[green]Loading best decoder-stage INR checkpoint for prediction:[/green] "
            f"{best_predict_ckpt}"
        )

        fitting_system = INRFittingSystem.load_from_checkpoint(
            best_predict_ckpt,
            configs=pipeline_configs,
            val_pca=None,
            decoder=decoder,
            weights_only=False,
        )

    del train_dataset
    del train_dataloader
    import gc
    gc.collect()
    # predict
    if pipeline_configs.predict_mode=="all":
        del val_dataset
        del val_dataloader
        gc.collect()
        test_dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers, drop_last=False)
        trainer.predict(fitting_system, test_dataloader)        
    elif pipeline_configs.predict_mode=="val":
        del dataset
        gc.collect()
        test_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers, drop_last=False)
        trainer.predict(fitting_system, test_dataloader) 
    else:
        print("End without writing predictions.")

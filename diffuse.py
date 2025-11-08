import torch, os, random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
from diffusers.models.unets.unet_2d import UNet2DModel
from transformers import get_cosine_schedule_with_warmup
from torchvision.utils import save_image, make_grid
import torch.nn as nn
import lightning as l
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from contextlib import contextmanager
from final import VAE
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

torch.cuda.empty_cache()

class TrainingOnlyProgressBar(TQDMProgressBar):
    def __init__(self, refresh_rate=10):
        super().__init__(refresh_rate=refresh_rate)

    def on_validation_epoch_start(self, trainer, pl_module):
        pass

    def on_validation_epoch_end(self, trainer, pl_module):
        pass

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        pass

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        pass

class LitEma(nn.Module):
    def __init__(self, model, decay=0.9999, use_num_upates=True):
        super().__init__()
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')

        self.m_name2s_name = {}
        self.register_buffer('decay', torch.tensor(decay, dtype=torch.float32))
        self.register_buffer('num_updates', torch.tensor(0, dtype=torch.int) if use_num_upates
        else torch.tensor(-1, dtype=torch.int))

        for name, p in model.named_parameters():
            if p.requires_grad:
                s_name = name.replace('.', '')
                self.m_name2s_name.update({name: s_name})
                self.register_buffer(s_name, p.clone().detach().data)

        self.collected_params = []

    def reset_num_updates(self):
        del self.num_updates
        self.register_buffer('num_updates', torch.tensor(0, dtype=torch.int))

    def forward(self, model):
        decay = self.decay

        if self.num_updates >= 0:
            self.num_updates += 1
            decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))

        one_minus_decay = 1.0 - decay

        with torch.no_grad():
            m_param = dict(model.named_parameters())
            shadow_params = dict(self.named_buffers())

            for key in m_param:
                if m_param[key].requires_grad:
                    sname = self.m_name2s_name[key]
                    shadow_params[sname] = shadow_params[sname].type_as(m_param[key])
                    shadow_params[sname].sub_(one_minus_decay * (shadow_params[sname] - m_param[key]))
                else:
                    assert not key in self.m_name2s_name

    def copy_to(self, model):
        m_param = dict(model.named_parameters())
        shadow_params = dict(self.named_buffers())
        for key in m_param:
            if key in self.m_name2s_name:
                m_param[key].data.copy_(shadow_params[self.m_name2s_name[key]].data)

    def store(self, parameters):
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters):
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)

class VariableRatioDataset(Dataset):
    def __init__(self, path, base_height=64, width_multiple=128):
        self.path = path
        self.base_height = base_height
        self.width_multiple = width_multiple
        self.image_files = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(('.png', '.PNG'))]
        filtered_files = []
        for filepath in self.image_files:
            with Image.open(filepath) as img:
                width, height = img.size
                if width // height == 1:
                    continue
                filtered_files.append(filepath)
        self.image_files = filtered_files
        self.buckets = self._create_buckets()
        self.normalize = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    def _get_target_dim(self, width, height):
        new_height = self.base_height
        new_width = int(width * (new_height / height))
        cropped_width = (new_width // self.width_multiple) * self.width_multiple
        if cropped_width == 0: cropped_width = self.width_multiple
        return (cropped_width, new_height)

    def _create_buckets(self):
        buckets = {}
        for i, filepath in enumerate(self.image_files):
            with Image.open(filepath) as img:
                target_dim = self._get_target_dim(*img.size)
                if target_dim not in buckets: buckets[target_dim] = []
                buckets[target_dim].append(i)
        return buckets

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filepath = self.image_files[idx]
        image = Image.open(filepath).convert("RGB")
        target_width, target_height = self._get_target_dim(*image.size)
        original_width, original_height = image.size
        new_width = int(original_width * (target_height / original_height))
        resized_image = image.resize((new_width, target_height), Image.Resampling.LANCZOS)
        crop_total = new_width - target_width
        crop_left = crop_total // 2
        crop_right = crop_left + target_width
        cropped_image = resized_image.crop((crop_left, 0, crop_right, target_height))
        return self.normalize(cropped_image)

class BucketBatchSampler:
    def __init__(self, buckets, batch_size=32):
        self.buckets = buckets
        self.batch_size = batch_size
        self.num_batches = sum(len(indices) // batch_size for indices in buckets.values())

    def __iter__(self):
        all_batches = []
        for bucket_indices in self.buckets.values():
            shuffled_indices = list(bucket_indices)
            random.shuffle(shuffled_indices)
            
            for i in range(len(shuffled_indices) // self.batch_size):
                batch = shuffled_indices[i * self.batch_size : (i + 1) * self.batch_size]
                all_batches.append(batch)

        random.shuffle(all_batches)
        return iter(all_batches)

    def __len__(self):
        return self.num_batches

def compute_latent_scaling_factor(vae_model, dataloader, n_max_images=None):
    vae_model.eval()
    device = next(vae_model.parameters()).device
    collected = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if n_max_images and i * batch.shape[0] >= n_max_images:
                break
            imgs = batch.to(device)
            mu, log_var = vae_model.encode(imgs)
            # sigma = torch.exp(0.5 * log_var)
            # eps = torch.randn_like(sigma)
            # z = mu + sigma * eps
            z = mu
            collected.append(z.detach().cpu().flatten())
    all_latents = torch.cat(collected, dim=0)
    return float(all_latents.std(unbiased=False))  # scalar std

class LitDiffusion(l.LightningModule):
    def __init__(self, vae_model, latent_scaling_factor=1.0, use_ema=True, lr=8e-5, use_snr_weighting=True):
        super().__init__()
        self.save_hyperparameters(ignore=['vae_model'])
        self.image_log_dir = "./logs/diffusion/images"
        self.lr = lr
        self.latent_scaling_factor = latent_scaling_factor
        self.use_snr_weighting = use_snr_weighting
        self.vae = vae_model
        self.vae.requires_grad_(False)
        self.vae.eval()
        self.use_ema = use_ema
        self.unet = UNet2DModel(
            sample_size=None,
            in_channels=4,
            out_channels=4,
            layers_per_block=2,
            block_out_channels=(128, 256, 512),
            down_block_types=(
                "AttnDownBlock2D", 
                "AttnDownBlock2D", 
                "DownBlock2D", 
            ),
            up_block_types=(
                "UpBlock2D", 
                "AttnUpBlock2D", 
                "AttnUpBlock2D", 
            ),
        )
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule="scaled_linear",
            clip_sample=False,
        )
        self.noise_scheduler.set_timesteps(num_inference_steps=70)
        if self.use_ema:
            self.model_ema = LitEma(self.unet, decay=0.97)

    @contextmanager
    def ema_scope(self):
        if self.use_ema:
            self.model_ema.store(self.unet.parameters())
            self.model_ema.copy_to(self.unet)
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.unet.parameters())

    def training_step(self, batch, batch_idx):
        images = batch
        with torch.no_grad():
            mu, log_var = self.vae.encode(images)
            # latents = self.vae.reparameterize(mu, log_var)
            latents = mu / self.latent_scaling_factor
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=self.device).long()
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        noise_pred = self.unet(noisy_latents, timesteps).sample
        if self.use_snr_weighting:
            snr = self.compute_snr(timesteps)
            weights = torch.clamp(snr, min=1.0, max=5.0)
            loss = F.mse_loss(noise_pred, noise, reduction='none')
            loss = (loss * weights.view(-1, 1, 1, 1)).mean()
        else:
            loss = F.mse_loss(noise_pred, noise)
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def compute_snr(self, timesteps):
        alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(self.device)
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod[timesteps])
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod[timesteps])
        snr = (sqrt_alphas_cumprod / sqrt_one_minus_alphas_cumprod) ** 2
        return snr

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.unet.parameters(), lr=self.lr, betas=(0.6, 0.9))
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': get_cosine_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=200,
                    num_training_steps=self.trainer.estimated_stepping_batches
                ),
                'interval': 'step',
            }
        }

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        images = batch
        with torch.no_grad():
            mu, log_var = self.vae.encode(images)
            latents = mu / self.latent_scaling_factor
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=self.device).long()
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
            noise_pred = self.unet(noisy_latents, timesteps).sample
            val_loss = F.mse_loss(noise_pred, noise)
            
        self.log('val_loss', val_loss, on_epoch=True, prog_bar=True, sync_dist=True)

        if batch_idx == 0 and self.current_epoch % 5 == 0:
            with self.ema_scope():
                latents_shape = (images.shape[0], 4, images.shape[2] // 4, images.shape[3] // 4)
                gen_noise = torch.randn(latents_shape, device=self.device)
                
                self.noise_scheduler.set_timesteps(num_inference_steps=50)
                for t in self.noise_scheduler.timesteps:
                    pred_noise = self.unet(gen_noise, t).sample
                    gen_noise = self.noise_scheduler.step(pred_noise, t, gen_noise).prev_sample
            with self.vae.ema_scope():
                decoded_images = self.vae.decode(gen_noise * self.latent_scaling_factor)
            grid = make_grid((decoded_images + 1) / 2, nrow=4)
            filename = f"generated_epoch_{self.current_epoch:03d}.png"
            filepath = os.path.join(self.image_log_dir, filename)
            save_image(grid, filepath)
            
        return val_loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.use_ema:
            self.model_ema(self.unet)

if __name__ == "__main__":
    dataset = VariableRatioDataset("./relabelled", base_height=64, width_multiple=128)
    sampler = BucketBatchSampler(dataset.buckets, batch_size=64)
    dataloader = DataLoader(dataset, batch_sampler=sampler, num_workers=os.cpu_count(), pin_memory=True)

    val_dataset = VariableRatioDataset("./data_val", base_height=64, width_multiple=128)
    val_sampler = BucketBatchSampler(val_dataset.buckets, batch_size=8)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, num_workers=os.cpu_count(), pin_memory=True)

    vae_path = "/scratch/e1375459/vae-epoch=98-val_loss=0.0084.ckpt"
    vae_model = VAE.load_from_checkpoint(vae_path).to("cuda")
    
    # scaling_factor = compute_latent_scaling_factor(vae_model, dataloader)
    # print(f"\nlatent scaling factor: {scaling_factor}")

    diffusion_model = LitDiffusion(
        vae_model=vae_model,
        latent_scaling_factor=1.0,
        lr=1e-4,
        use_snr_weighting=True
    )

    checkpoint_path = "/scratch/e1375459/vae-epoch=14-val_loss=0.0265.ckpt"
    
    if os.path.exists(checkpoint_path):
        print(f"Manually loading weights from: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path)
        diffusion_model.load_state_dict(ckpt['state_dict'], strict=True)
    else:
        raise Exception("Load from what?")

    trainer = l.Trainer(
        max_epochs=100,
        accelerator="gpu",
        devices=1,
        callbacks=[
            ModelCheckpoint(
                monitor='val_loss', 
                mode='min', 
                save_top_k=3, 
                filename='diffusion-{epoch:02d}-{val_loss:.4f}', 
                every_n_epochs=10
            ),
            TrainingOnlyProgressBar(refresh_rate=10) 
        ],
        default_root_dir="./diffusion_checkpoints",
        precision="bf16-mixed",
        gradient_clip_val=1.0,
    )

    trainer.fit(diffusion_model, dataloader, val_loader)

import torch, os, random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
from transformers import get_cosine_schedule_with_warmup
from torchvision.utils import save_image, make_grid
import torch.nn as nn
import lightning as l
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
import lpips
from contextlib import contextmanager

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

class KLAnnealing(l.Callback):
    def __init__(self, start_step, anneal_steps, final_kl_weight):
        super().__init__()
        self.start_step = start_step
        self.anneal_steps = anneal_steps
        self.final_kl_weight = final_kl_weight

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        current_step = trainer.global_step
        if current_step >= self.start_step:
            progress = min(1.0, (current_step - self.start_step) / self.anneal_steps)
            new_kl_weight = self.final_kl_weight * progress
            pl_module.kl_weight = new_kl_weight
            pl_module.log('kl_anneal_weight', new_kl_weight, on_step=True, on_epoch=False)

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
        """
        Save the current parameters for restoring later.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            temporarily stored.
        """
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters):
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored parameters.
        """
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
    
class Encoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=4):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, 128, kernel_size=3, padding=1)
        self.blocks = nn.Sequential(
            ResBlock(128, 128),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # 64->32
            ResBlock(256, 256),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1), # 32->16
            ResBlock(512, 512)
        )
        self.norm_out = nn.GroupNorm(32, 512)
        self.silu_out = nn.SiLU()
        self.conv_out = nn.Conv2d(512, 2 * latent_dim, kernel_size=3, padding=1)

    def forward(self, x):
        h = self.conv_in(x)
        h = self.blocks(h)
        h = self.silu_out(self.norm_out(h))
        mu_logvar = self.conv_out(h)
        mu, log_var = torch.chunk(mu_logvar, 2, dim=1)
        return mu, log_var

class Decoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=4):
        super().__init__()
        self.conv_in = nn.Conv2d(latent_dim, 512, kernel_size=3, padding=1)
        self.blocks = nn.Sequential(
            ResBlock(512, 512),
            nn.Upsample(scale_factor=2), # 16->32
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            ResBlock(256, 256),
            nn.Upsample(scale_factor=2), # 32->64
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            ResBlock(128, 128)
        )
        self.norm_out = nn.GroupNorm(32, 128)
        self.silu_out = nn.SiLU()
        self.conv_out = nn.Conv2d(128, in_channels, kernel_size=3, padding=1)

    def forward(self, z):
        h = self.conv_in(z)
        h = self.blocks(h)
        h = self.silu_out(self.norm_out(h))
        recon = self.conv_out(h)
        recon= torch.tanh(recon)
        return recon
    
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.silu1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.silu2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        h = self.silu1(self.norm1(x))
        h = self.conv1(h)
        h = self.silu2(self.norm2(h))
        h = self.conv2(h)
        return h + self.shortcut(x)

class VAE(l.LightningModule):
    def __init__(self, 
                 in_channels=3, 
                 latent_dim=4, 
                 kl_weight=0.0, 
                 lpips_weight=3e-2, 
                 ema_decay=0.99,
                 use_ema=False,
                 log_dir="./logs/vae",
                 lr=8e-5):
        
        super().__init__()
        self.save_hyperparameters()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight
        # self.lpips_weight = lpips_weight
        self.use_ema = use_ema
        self.log_dir = log_dir
        self.lr = lr
        # self.lpips_loss = lpips.LPIPS(net='vgg').requires_grad_(False)
        # self.lpips_loss.eval()
        self.image_log_dir = os.path.join(self.log_dir, "images")
        os.makedirs(self.image_log_dir, exist_ok=True)
        self.encoder = Encoder(in_channels=self.in_channels, latent_dim=self.latent_dim)
        self.decoder = Decoder(in_channels=self.in_channels, latent_dim=self.latent_dim)
        if self.use_ema:
            self.model_ema = LitEma(self, decay=ema_decay)

    def encode(self, x):
        return self.encoder(x)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    @contextmanager
    def ema_scope(self):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())

    def forward(self, x):
        mu, log_var = self.encode(x)
        log_var = torch.clamp(log_var, min=-25.0, max=25.0)
        z = self.reparameterize(mu, log_var)
        recon = self.decode(z)
        return recon, mu, log_var

    def training_step(self, batch, batch_idx):
        images = batch 
        reconstructions, mu, log_var = self.forward(images)
        recon_loss = F.l1_loss(reconstructions, images)
        kl_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=[1,2,3]))
        # lpips_loss = torch.mean(self.lpips_loss(reconstructions, images))
        loss = recon_loss + self.kl_weight * kl_loss # + self.lpips_weight * lpips_loss
        self.log('train_loss', loss)
        self.log('recon_loss', recon_loss)
        self.log('kl_loss', kl_loss)
        # self.log('lpips_loss', lpips_loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas=(0.6, 0.9))
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': get_cosine_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=100,
                    num_training_steps=self.trainer.estimated_stepping_batches
                ),
                'interval': 'step',
            }
        }
    
    def validation_step(self, batch, batch_idx):
        images = batch 
        with self.ema_scope():
            reconstructions, mu, log_var = self.forward(images)
        recon_loss = F.l1_loss(reconstructions, images) 
        kl_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=[1,2,3]))
        # lpips_loss = torch.mean(self.lpips_loss(reconstructions, images))
        loss = recon_loss + self.kl_weight * kl_loss # + self.lpips_weight * lpips_loss
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_recon_loss', recon_loss, on_epoch=True, sync_dist=True)
        self.log('val_kl_loss', kl_loss, on_epoch=True, sync_dist=True)
        
        if batch_idx == 0 and self.current_epoch % 5 == 0:
            grid = self.make_image_grid(images, reconstructions)
            filename = f"recons_epoch_{self.current_epoch:03d}.png"
            filepath = os.path.join(self.image_log_dir, filename)
            save_image(grid, filepath)
        
        return loss

    def make_image_grid(self, images, recons, max_images=8):
        images = (images[:max_images] + 1) / 2
        recons = (recons[:max_images] + 1) / 2
        images = torch.clamp(images, 0, 1)
        recons = torch.clamp(recons, 0, 1)
        grid = torch.cat((images, recons))
        grid = make_grid(grid, nrow=images.shape[0])
        return grid

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.use_ema:
            self.model_ema(self)

if __name__ == "__main__":
    dataset = VariableRatioDataset("./data", base_height=64, width_multiple=128)
    sampler = BucketBatchSampler(dataset.buckets, batch_size=64)
    dataloader = DataLoader(dataset, batch_sampler=sampler, num_workers=os.cpu_count(), pin_memory=True)

    val_dataset = VariableRatioDataset("./data_val", base_height=64, width_multiple=128)
    val_sampler = BucketBatchSampler(val_dataset.buckets, batch_size=4)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, num_workers=os.cpu_count(), pin_memory=True)

    vae_model = VAE(in_channels=3, latent_dim=4, kl_weight=4e-6, use_ema=True, lr=1e-4)
    vae_model.compile()

    # checkpoint_path = None
    
    # if os.path.exists(checkpoint_path):
    #     print(f"Manually loading weights from: {checkpoint_path}")
    #     ckpt = torch.load(checkpoint_path)
    #     vae_model.load_state_dict(ckpt['state_dict'], strict=True)
    # else:
    #     raise Exception("Load from what?")

    trainer = l.Trainer(
        max_epochs=100,
        accelerator="gpu",
        devices=1,
        callbacks=
        [
            # KLAnnealing(start_step=0, anneal_steps=3675, final_kl_weight=5e-4),
            ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=3, filename='vae-{epoch:02d}-{val_loss:.4f}', every_n_epochs=3),
            TrainingOnlyProgressBar(refresh_rate=10)
        ],
        default_root_dir="./vae_checkpoints",
        precision="32-true",
        enable_checkpointing=True,
        gradient_clip_val=1.0
    )

    trainer.fit(vae_model, dataloader, val_loader, ckpt_path=None)
    torch.save(vae_model.state_dict(), f"./vae_checkpoints/last_model_kl5e4.pth")

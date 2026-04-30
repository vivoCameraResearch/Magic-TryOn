from magictryon_turbo.data import ODERegressionDataset, ODERegressionLMDBDataset
from magictryon_turbo.ode_regression import ODERegression
from magictryon_turbo.models import get_block_class
from collections import defaultdict
from magictryon_turbo.util import (
    launch_distributed_job,
    set_seed, init_logging_folder,
    fsdp_wrap, cycle,
    fsdp_state_dict,
    barrier
)
import imageio, numpy as np
import torch.distributed as dist
from omegaconf import OmegaConf
import argparse
import torch
import wandb
import time
import os


class Trainer:
    def __init__(self, config):
        self.config = config

        # Step 1: Initialize the distributed training environment (rank, seed, dtype, logging etc.)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        # self.is_main_process = global_rank == 0
        self.is_main_process=True

        # use a random seed for the training
        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()

        set_seed(config.seed + global_rank)

        if self.is_main_process:
            # self.output_path, self.wandb_folder = init_logging_folder(config)
            self.output_path = config.output_path

        # Step 2: Initialize the model and optimizer

        if config.distillation_loss == "ode":
            self.distillation_model = ODERegression(config, device=self.device)
        else:
            raise ValueError("Invalid distillation loss type")

        self.distillation_model.generator = fsdp_wrap(
            self.distillation_model.generator,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy,
            transformer_module=(get_block_class(config.generator_fsdp_transformer_module),
                                ) if config.generator_fsdp_wrap_strategy == "transformer" else None
        )
        self.distillation_model.text_encoder = fsdp_wrap(
            self.distillation_model.text_encoder,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
            transformer_module=(get_block_class(config.text_encoder_fsdp_transformer_module),
                                ) if config.text_encoder_fsdp_wrap_strategy == "transformer" else None
        )

        self.generator_optimizer = torch.optim.AdamW(
            [param for param in self.distillation_model.generator.parameters()
             if param.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2)
        )

        # Step 3: Initialize the dataloader
        # dataset = ODERegressionDataset(config.data_path)
        dataset = ODERegressionLMDBDataset(
            config.data_path, max_pair=getattr(config, "max_pair", int(1e8)))
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=True, drop_last=True)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=config.batch_size, sampler=sampler, num_workers=8)
        self.dataloader = cycle(dataloader)

        self.step = 0
        self.max_grad_norm = 10.0
        self.previous_time = None

    def save(self):
        print("Start gathering distributed model states...")
        generator_state_dict = fsdp_state_dict(
            self.distillation_model.generator)
        state_dict = {
            "generator": generator_state_dict
        }

        if self.is_main_process:
            os.makedirs(os.path.join(self.output_path,
                        f"checkpoint_model_{self.step:06d}"), exist_ok=True)
            torch.save(state_dict, os.path.join(self.output_path,
                       f"checkpoint_model_{self.step:06d}", "model.pt"))
            print("Model saved to", os.path.join(self.output_path,
                  f"checkpoint_model_{self.step:06d}", "model.pt"))

    def save_video(self, arr, path, fps=16):
        arr = np.asarray(arr)
        # 取 batch 中第一个样本，并将 (T, C, H, W) 转为 (T, H, W, C)
        if arr.ndim == 5:
            arr = arr[0].transpose(0, 2, 3, 1)  # (T, C, H, W) -> (T, H, W, C)
        # 检查通道数是否合规
        if arr.shape[-1] not in [1, 2, 3, 4]:
            print(f"[Warning] Unexpected channel count: {arr.shape[-1]}, converting to RGB.")
            if arr.shape[-1] > 3:
                arr = arr[..., :3]
            else:
                arr = np.repeat(arr[..., :1], 3, axis=-1)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        imageio.mimwrite(path, arr, fps=fps, codec="libx264", quality=8)
        

    def train_one_step(self):
        self.distillation_model.eval()  # prevent any randomness (e.g. dropout)

        # Step 1: Get the next batch of text prompts
        batch = next(self.dataloader)
        text_prompts = batch["prompt"]
        ode_latent = batch["ode_latent"].to(device=self.device, dtype=self.dtype)
        ode_latent = ode_latent.permute(0, 1, 3, 2, 4, 5) # meet the setting
        
        masked_video_latents = batch["masked_video_latents"].to(device=self.device, dtype=self.dtype)
        pose_latents = batch["pose_latents"].to(device=self.device, dtype=self.dtype)
        mask_input = batch["mask_input"].to(device=self.device, dtype=self.dtype)
        cloth_latents = batch["cloth_latents"].to(device=self.device, dtype=self.dtype)
        cloth_line_latents = batch["cloth_line_latents"].to(device=self.device, dtype=self.dtype)

        image_clip = batch["image_clip"].to(device=self.device, dtype=self.dtype)
        
        y = [masked_video_latents, pose_latents, mask_input, cloth_latents, cloth_line_latents]
        
        # Step 2: Extract the conditional infos
        with torch.no_grad():
            conditional_dict = self.distillation_model.text_encoder(
                text_prompts=text_prompts)

        # Step 3: Train the generator
        generator_loss, log_dict = self.distillation_model.generator_loss(
            ode_latent=ode_latent,
            conditional_dict=conditional_dict,
            #### other condition ####
            y = y,
            image_clip = image_clip
        )

        unnormalized_loss = log_dict["unnormalized_loss"]
        timestep = log_dict["timestep"]

        if self.world_size > 1:
            gathered_unnormalized_loss = torch.zeros(
                [self.world_size, *unnormalized_loss.shape],
                dtype=unnormalized_loss.dtype, device=self.device)
            gathered_timestep = torch.zeros(
                [self.world_size, *timestep.shape],
                dtype=timestep.dtype, device=self.device)

            dist.all_gather_into_tensor(
                gathered_unnormalized_loss, unnormalized_loss)
            dist.all_gather_into_tensor(gathered_timestep, timestep)
        else:
            gathered_unnormalized_loss = unnormalized_loss
            gathered_timestep = timestep

        loss_breakdown = defaultdict(list)
        stats = {}

        for index, t in enumerate(timestep):
            loss_breakdown[str(int(t.item()) // 250 * 250)].append(
                unnormalized_loss[index].item())

        for key_t in loss_breakdown.keys():
            stats["loss_at_time_" + key_t] = sum(loss_breakdown[key_t]) / \
                len(loss_breakdown[key_t])

        self.generator_optimizer.zero_grad()
        generator_loss.backward()
        generator_grad_norm = self.distillation_model.generator.clip_grad_norm_(
            self.max_grad_norm)
        self.generator_optimizer.step()
        
        
        # Visualization
        if self.is_main_process:
            # ---- 准备视频 (B,T,C,H,W) → (T,H,W,C) & [0,255] uint8 ----
            def to_np_uint8(x):       # x: Tensor on GPU
                x = (x * 0.5 + 0.5).clamp(0, 1)       # [-1,1] → [0,1]
                return (255.0 * x.cpu().numpy()).astype(np.uint8)
            
            input_vid  = to_np_uint8(self.distillation_model.vae.decode_to_pixel(log_dict["input"]))         # (T,H,W,C)
            output_vid = to_np_uint8(self.distillation_model.vae.decode_to_pixel(log_dict["output"]))
            gt_vid     = to_np_uint8(self.distillation_model.vae.decode_to_pixel(ode_latent[:, -1]))
            
            # ---- 手动保存到磁盘 ----
            save_dir = os.path.join("result_ode/vivid")   # 想放哪就改哪
            os.makedirs(save_dir, exist_ok=True)
            
            self.save_video(input_vid,  os.path.join(save_dir, f"{self.step:07d}_input.mp4"))
            self.save_video(output_vid, os.path.join(save_dir, f"{self.step:07d}_output.mp4"))
            self.save_video(gt_vid,     os.path.join(save_dir, f"{self.step:07d}_gt.mp4"))

        # Step 4: Logging
        # if self.is_main_process:
        #     wandb_loss_dict = {
        #         "generator_loss": generator_loss.item(),
        #         "generator_grad_norm": generator_grad_norm.item(),
        #         **stats
        #     }
        #     wandb.log(wandb_loss_dict, step=self.step)
        
        if self.is_main_process:
            # ---------- 终端打印 ----------
            print(f"[step {self.step:>7d}] "
                f"generator_loss: {generator_loss.item()} | "
                f"generator_grad_norm: {generator_grad_norm.item()}")

    def train(self):
        while True:
            self.train_one_step()
            if (not self.config.no_save) and self.step % self.config.log_iters == 0:
                self.save()
                torch.cuda.empty_cache()

            barrier()
            if self.is_main_process:
                current_time = time.time()
                if self.previous_time is None:
                    self.previous_time = current_time
                else:
                    # wandb.log({"per iteration time": current_time -
                    #           self.previous_time}, step=self.step)
                    self.previous_time = current_time

            self.step += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--no_save", action="store_true")

    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)
    config.no_save = args.no_save

    trainer = Trainer(config)
    trainer.train()

    # wandb.finish()


if __name__ == "__main__":
    main()

from magictryon_turbo.data import ODERegressionLMDBDataset
from magictryon_turbo.models import get_block_class
from magictryon_turbo.data import TextDataset
from magictryon_turbo.util import (
    launch_distributed_job,
    prepare_for_saving,
    set_seed, init_logging_folder,
    fsdp_wrap, cycle,
    fsdp_state_dict,
    barrier
)
import torch.distributed as dist
from omegaconf import OmegaConf
from magictryon_turbo.dmd import DMD
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

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = global_rank == 0

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
        if config.distillation_loss == "dmd":
            self.distillation_model = DMD(config, device=self.device)
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

        self.distillation_model.real_score = fsdp_wrap(
            self.distillation_model.real_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.real_score_fsdp_wrap_strategy,
            transformer_module=(get_block_class(config.real_score_fsdp_transformer_module),
                                ) if config.real_score_fsdp_wrap_strategy == "transformer" else None
        )

        self.distillation_model.fake_score = fsdp_wrap(
            self.distillation_model.fake_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.fake_score_fsdp_wrap_strategy,
            transformer_module=(get_block_class(config.fake_score_fsdp_transformer_module),
                                ) if config.fake_score_fsdp_wrap_strategy == "transformer" else None
        )

        self.distillation_model.text_encoder = fsdp_wrap(
            self.distillation_model.text_encoder,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
            transformer_module=(get_block_class(config.text_encoder_fsdp_transformer_module),
                                ) if config.text_encoder_fsdp_wrap_strategy == "transformer" else None
        )

        if not config.no_visualize:
            self.distillation_model.vae = self.distillation_model.vae.to(
                device=self.device, dtype=torch.bfloat16 if config.mixed_precision else torch.float32)

        self.generator_optimizer = torch.optim.AdamW(
            [param for param in self.distillation_model.generator.parameters()
             if param.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2)
        )

        self.critic_optimizer = torch.optim.AdamW(
            [param for param in self.distillation_model.fake_score.parameters()
             if param.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2)
        )

        # Step 3: Initialize the dataloader

        self.backward_simulation = getattr(config, "backward_simulation", True)

        if self.backward_simulation:
            dataset = TextDataset(config.data_path)
        else:
            print("load tryon dataset")
            dataset = ODERegressionLMDBDataset(
                config.data_path, max_pair=int(1e8))
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=True, drop_last=True)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=config.batch_size, sampler=sampler)
        self.dataloader = cycle(dataloader)

        self.step = 0
        self.max_grad_norm = 10.0
        self.previous_time = None

    def save(self):
        print("Start gathering distributed model states...")
        generator_state_dict = fsdp_state_dict(
            self.distillation_model.generator)
        critic_state_dict = fsdp_state_dict(
            self.distillation_model.fake_score)
        state_dict = {
            "generator": generator_state_dict,
            "critic": critic_state_dict
        }

        if self.is_main_process:
            os.makedirs(os.path.join(self.output_path,
                        f"checkpoint_model_{self.step:06d}"), exist_ok=True)
            torch.save(state_dict, os.path.join(self.output_path,
                       f"checkpoint_model_{self.step:06d}", "model.pt"))
            print("Model saved to", os.path.join(self.output_path,
                  f"checkpoint_model_{self.step:06d}", "model.pt"))

    def train_one_step(self):
        self.distillation_model.eval()  # prevent any randomness (e.g. dropout)

        TRAIN_GENERATOR = self.step % self.config.dfake_gen_update_ratio == 0
        VISUALIZE = self.step % self.config.log_iters == 0 and not self.config.no_visualize

        if self.step % 20 == 0:
            torch.cuda.empty_cache()

        # Step 1: Get the next batch of text prompts
        if not self.backward_simulation:
            batch = next(self.dataloader)
            text_prompts = batch["prompt"]
            ode_latent = batch["ode_latent"].to(device=self.device, dtype=self.dtype)
            ode_latent = ode_latent.permute(0, 1, 3, 2, 4, 5) # meet the setting
            clean_latent = ode_latent[:, -1]
            masked_video_latents = batch["masked_video_latents"].to(device=self.device, dtype=self.dtype)
            pose_latents = batch["pose_latents"].to(device=self.device, dtype=self.dtype)
            mask_input = batch["mask_input"].to(device=self.device, dtype=self.dtype)
            cloth_latents = batch["cloth_latents"].to(device=self.device, dtype=self.dtype)
            cloth_line_latents = batch["cloth_line_latents"].to(device=self.device, dtype=self.dtype)

            image_clip = batch["image_clip"].to(device=self.device, dtype=self.dtype)
            
            y = [masked_video_latents, pose_latents, mask_input, cloth_latents, cloth_line_latents]
        else:
            text_prompts = next(self.dataloader)
            clean_latent = None

        batch_size = len(text_prompts)
        image_or_video_shape = list(self.config.image_or_video_shape)
        image_or_video_shape[0] = batch_size

        # Step 2: Extract the conditional infos
        with torch.no_grad():
            conditional_dict = self.distillation_model.text_encoder(
                text_prompts=text_prompts)

            if not getattr(self, "unconditional_dict", None):
                unconditional_dict = self.distillation_model.text_encoder(
                    text_prompts=[self.config.negative_prompt] * batch_size)
                unconditional_dict = {k: v.detach()
                                      for k, v in unconditional_dict.items()}
                self.unconditional_dict = unconditional_dict  # cache the unconditional_dict
            else:
                unconditional_dict = self.unconditional_dict

        # Step 3: Train the generator
        if TRAIN_GENERATOR:
            generator_loss, generator_log_dict = self.distillation_model.generator_loss(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                clean_latent=clean_latent,
                #### other condition ####
                y = y,
                image_clip = image_clip
            )

            self.generator_optimizer.zero_grad()
            generator_loss.backward()
            generator_grad_norm = self.distillation_model.generator.clip_grad_norm_(
                self.max_grad_norm)
            self.generator_optimizer.step()
        else:
            generator_log_dict = {}

        # Step 4: Train the critic
        critic_loss, critic_log_dict = self.distillation_model.critic_loss(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            clean_latent=clean_latent,
            #### other condition ####
            y = y,
            image_clip = image_clip
        )

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_grad_norm = self.distillation_model.fake_score.clip_grad_norm_(
            self.max_grad_norm)
        self.critic_optimizer.step()

        # Step 5: Logging
        if self.is_main_process:
            # wandb_loss_dict = {
            #     "critic_loss": critic_loss.item(),
            #     "critic_grad_norm": critic_grad_norm.item()
            # }
            print(f"[step {self.step:>7d}] "
                f"critic_loss: {critic_loss.item()} | "
                f"critic_grad_norm: {critic_grad_norm.item()}")


            if TRAIN_GENERATOR:
                # wandb_loss_dict.update(
                #     {
                #         "generator_loss": generator_loss.item(),
                #         "generator_grad_norm": generator_grad_norm.item(),
                #         "dmdtrain_gradient_norm": generator_log_dict["dmdtrain_gradient_norm"].item()
                #     }
                # )
                print(f"[step {self.step:>7d}] "
                f"generator_loss: {generator_loss.item()} | "
                f"generator_grad_norm: {generator_grad_norm.item()} | "
                f"dmdtrain_gradient_norm: {generator_log_dict['dmdtrain_gradient_norm'].item()}")

            # if VISUALIZE:
            #     self.add_visualization(generator_log_dict, critic_log_dict, wandb_loss_dict)

            # wandb.log(wandb_loss_dict, step=self.step)

    def add_visualization(self, generator_log_dict, critic_log_dict, wandb_loss_dict):
        critictrain_latent, critictrain_noisy_latent, critictrain_pred_image = map(
            lambda x: self.distillation_model.vae.decode_to_pixel(
                x).squeeze(1),
            [critic_log_dict['critictrain_latent'], critic_log_dict['critictrain_noisy_latent'],
                critic_log_dict['critictrain_pred_image']]
        )

        wandb_loss_dict.update({
            "critictrain_latent": prepare_for_saving(critictrain_latent),
            "critictrain_noisy_latent": prepare_for_saving(critictrain_noisy_latent),
            "critictrain_pred_image": prepare_for_saving(critictrain_pred_image)
        })

        if "dmdtrain_clean_latent" in generator_log_dict:
            (dmdtrain_clean_latent, dmdtrain_noisy_latent, dmdtrain_pred_real_image, dmdtrain_pred_fake_image) = map(
                lambda x: self.distillation_model.vae.decode_to_pixel(
                    x).squeeze(1),
                [generator_log_dict['dmdtrain_clean_latent'], generator_log_dict['dmdtrain_noisy_latent'],
                    generator_log_dict['dmdtrain_pred_real_image'], generator_log_dict['dmdtrain_pred_fake_image']]
            )

            wandb_loss_dict.update(
                {
                    "dmdtrain_clean_latent": prepare_for_saving(dmdtrain_clean_latent),
                    "dmdtrain_noisy_latent": prepare_for_saving(dmdtrain_noisy_latent),
                    "dmdtrain_pred_real_image": prepare_for_saving(dmdtrain_pred_real_image),
                    "dmdtrain_pred_fake_image": prepare_for_saving(dmdtrain_pred_fake_image)
                }
            )

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
    parser.add_argument("--no_visualize", action="store_true")

    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)
    config.no_save = args.no_save
    config.no_visualize = args.no_visualize

    trainer = Trainer(config)
    trainer.train()

    # wandb.finish()


if __name__ == "__main__":
    main()

from magictryon_turbo.models import (
    get_diffusion_wrapper,
    get_text_encoder_wrapper,
    get_vae_wrapper
)
from magictryon_turbo.models.wan.videox_fun.models.wan_image_encoder import CLIPModel
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torchvision.transforms.functional as TF
import torch.nn.functional as F 
from omegaconf import OmegaConf
import torch
import os
from PIL import Image
from tqdm import tqdm
import time

# 可以自定义一个日志文件路径
TIME_LOG_PATH = "path//magictryon_turbo-master/result/inference_time_log.txt"


class BidirectionalInferencePipeline(torch.nn.Module):
    def __init__(self, args, device):
        super().__init__()
        
        #### load image clip
        clip_config_path  = "path//VideoX-Fun-Camera/config/wan2.1/wan_civitai.yaml"
        clip_config = OmegaConf.load(clip_config_path)
        clip_name = "path//Wan2.1-Fun-V1.1-1.3B-Control"
        
        # Step 1: Initialize all models
        self.generator_model_name = getattr(
            args, "generator_name", args.model_name)
        self.generator = get_diffusion_wrapper(
            model_name=self.generator_model_name)()
        self.text_encoder = get_text_encoder_wrapper(
            model_name=args.model_name)()
        self.vae = get_vae_wrapper(model_name=args.model_name)()
        self.clip_image_encoder = CLIPModel.from_pretrained(
                                    os.path.join(clip_name, clip_config['image_encoder_kwargs'].get('image_encoder_subpath', 'image_encoder')),
                                ).to(device="cuda", dtype=torch.bfloat16).eval()

        # Step 2: Initialize all bidirectional wan hyperparmeters
        self.denoising_step_list = torch.tensor(
            args.denoising_step_list, dtype=torch.long, device=device)

        self.scheduler = self.generator.get_scheduler()
        if args.warp_denoising_step:  # Warp the denoising step according to the scheduler time shift
            timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32))).cuda()
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

    def resize_mask(self, mask, latent, process_first_frame_only=False):
        latent_size = latent.size()
        batch_size, channels, num_frames, height, width = mask.shape

        if process_first_frame_only:
            target_size = list(latent_size[2:])
            target_size[0] = 1
            first_frame_resized = F.interpolate(
                mask[:, :, 0:1, :, :],
                size=target_size,
                mode='trilinear',
                align_corners=False
            )
            
            target_size = list(latent_size[2:])
            target_size[0] = target_size[0] - 1
            if target_size[0] != 0:
                remaining_frames_resized = F.interpolate(
                    mask[:, :, 1:, :, :],
                    size=target_size,
                    mode='trilinear',
                    align_corners=False
                )
                resized_mask = torch.cat([first_frame_resized, remaining_frames_resized], dim=2)
            else:
                resized_mask = first_frame_resized
        else:
            target_size = list(latent_size[2:])
            resized_mask = F.interpolate(
                mask,
                size=target_size,
            )
        return resized_mask
    
    def inference(self, noise: torch.Tensor, 
                    text_prompts: Optional[Union[str, List[str]]] = None,
                    masked_video: Union[torch.FloatTensor] = None,
                    mask_video: Union[torch.FloatTensor] = None,
                    pose_video: Union[torch.FloatTensor] = None,
                    cloth_image: Union[torch.FloatTensor] = None,
                    cloth_line_image: Union[torch.FloatTensor] = None,
                    clip_image: Image = None,
                  ) -> torch.Tensor:
        """
        Perform inference on the given noise and text prompts.
        Inputs:
            noise (torch.Tensor): The input noise tensor of shape
                (batch_size, num_frames, num_channels, height, width).
            text_prompts (List[str]): The list of text prompts.
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_frames, num_channels, height, width). It is normalized to be in the range [0, 1].
        """
        conditional_dict = self.text_encoder(
            text_prompts=text_prompts
        )
        masked_video = masked_video.to(device="cuda", dtype=torch.bfloat16)
        pose_video = pose_video.to(device="cuda", dtype=torch.bfloat16)
        cloth_image = cloth_image.to(device="cuda", dtype=torch.bfloat16)
        cloth_line_image = cloth_line_image.to(device="cuda", dtype=torch.bfloat16)
        
        
        masked_video_latents = self.vae.encode_to_latent(masked_video).to(device="cuda", dtype=torch.bfloat16)
        
        bs, _, video_length, height, width = masked_video.size()

        mask_video = mask_video[:, :1]
        mask_condition = torch.concat(
                    [
                        torch.repeat_interleave(mask_video[:, :, 0:1], repeats=4, dim=2), 
                        mask_video[:, :, 1:]
                    ], dim=2
                )

        mask_condition = mask_condition.view(bs, mask_condition.shape[2] // 4, 4, height, width)
        mask_condition = mask_condition.transpose(1, 2)
        mask_latents = self.resize_mask(mask_condition, masked_video_latents, False).to(device="cuda", dtype=torch.bfloat16) 
        
        pose_latents = self.vae.encode_to_latent(pose_video).to(device="cuda", dtype=torch.bfloat16)
        cloth_latents = self.vae.encode_to_latent(cloth_image).to(device="cuda", dtype=torch.bfloat16)
        cloth_line_latents = self.vae.encode_to_latent(cloth_line_image).to(device="cuda", dtype=torch.bfloat16)
        
        y = [masked_video_latents, pose_latents, mask_latents, cloth_latents, cloth_line_latents]
        print(masked_video_latents.size(),
              pose_latents.size(),
              mask_latents.size(),
              cloth_latents.size(),
              cloth_line_latents.size()
              )
        # print(masked_video_latents.dtype,
        #       pose_latents.dtype,
        #       mask_latents.dtype,
        #       cloth_latents.dtype,
        #       cloth_line_latents.dtype
        #       )
        
        clip_image = TF.to_tensor(clip_image).sub_(0.5).div_(0.5).to(device="cuda", dtype=torch.bfloat16) 
        image_clip = self.clip_image_encoder([clip_image[:, None, :, :]])

        # initial point
        noisy_image_or_video = noise
        print(noisy_image_or_video.size())
        # print(noisy_image_or_video.dtype)

        # for index, current_timestep in enumerate(self.denoising_step_list):
        #     pred_image_or_video = self.generator(
        #         noisy_image_or_video=noisy_image_or_video,
        #         conditional_dict=conditional_dict,
        #         timestep=torch.ones(
        #             noise.shape[:2], dtype=torch.long, device=noise.device) * current_timestep,
        #         #### other condition ####
        #         y = y,
        #         image_clip = image_clip
        #     )  # [B, F, C, H, W]

        #     if index < len(self.denoising_step_list) - 1:
        #         next_timestep = self.denoising_step_list[index + 1] * torch.ones(
        #             noise.shape[:2], dtype=torch.long, device=noise.device)

        #         noisy_image_or_video = self.scheduler.add_noise(
        #             pred_image_or_video.flatten(0, 1),
        #             torch.randn_like(pred_image_or_video.flatten(0, 1)),
        #             next_timestep.flatten(0, 1)
        #         ).unflatten(0, noise.shape[:2])
        
        # ① ↓↓↓ 开始计时
        # t_start = time.perf_counter()

        # ② ↓↓↓ tqdm 包装原来的循环
        for index, current_timestep in tqdm(
                enumerate(self.denoising_step_list),
                total=len(self.denoising_step_list),
                desc="Denoising",
                unit="step"):

            pred_image_or_video = self.generator(
                noisy_image_or_video=noisy_image_or_video,
                conditional_dict=conditional_dict,
                timestep=torch.ones(
                    noise.shape[:2], dtype=torch.long, device=noise.device
                ) * current_timestep,
                # ---- other condition ----
                y=y,
                image_clip=image_clip,
            )  # [B, F, C, H, W]

            if index < len(self.denoising_step_list) - 1:
                next_timestep = self.denoising_step_list[index + 1] * torch.ones(
                    noise.shape[:2], dtype=torch.long, device=noise.device
                )

                noisy_image_or_video = self.scheduler.add_noise(
                    pred_image_or_video.flatten(0, 1),
                    torch.randn_like(pred_image_or_video.flatten(0, 1)),
                    next_timestep.flatten(0, 1),
                ).unflatten(0, noise.shape[:2])

        # ③ ↓↓↓ 结束计时并打印
        # t_end = time.perf_counter()
        # elapsed = t_end - t_start
        # print(f"\n✅ Inference finished in {elapsed:.2f}s ")


        # 追加写入到 txt，每次运行一行
        # os.makedirs(os.path.dirname(TIME_LOG_PATH) or ".", exist_ok=True)
        # with open(TIME_LOG_PATH, "a", encoding="utf-8") as f:
        #     f.write(f"{elapsed:.6f}\n")
        
        video = self.vae.decode_to_pixel(pred_image_or_video)
        video = (video * 0.5 + 0.5).clamp(0, 1)
        return video

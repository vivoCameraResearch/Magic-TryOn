from magictryon_turbo.models.model_interface import (
    InferencePipelineInterface,
    DiffusionModelInterface,
    TextEncoderInterface
)
from magictryon_turbo.scheduler import SchedulerInterface
from typing import List
import torch


class BidirectionalInferenceWrapper(InferencePipelineInterface):
    def __init__(self, denoising_step_list: List[int],
                 scheduler: SchedulerInterface,
                 generator: DiffusionModelInterface, **kwargs):
        super().__init__()
        self.scheduler = scheduler
        self.generator = generator
        self.denoising_step_list = denoising_step_list

    def inference_with_trajectory(self, noise: torch.Tensor, conditional_dict: dict) -> torch.Tensor:
        output_list = [noise]

        # initial point
        noisy_image_or_video = noise

        # use the last n-1 timesteps to simulate the generator's input
        for index, current_timestep in enumerate(self.denoising_step_list[:-1]):
            pred_image_or_video = self.generator(
                noisy_image_or_video=noisy_image_or_video,
                conditional_dict=conditional_dict,
                timestep=torch.ones(
                    noise.shape[:2], dtype=torch.long, device=noise.device) * current_timestep
            )  # [B, F, C, H, W]

            # TODO: Change backward simulation for causal video
            next_timestep = self.denoising_step_list[index + 1] * torch.ones(
                noise.shape[:2], dtype=torch.long, device=noise.device)
            noisy_image_or_video = self.scheduler.add_noise(
                pred_image_or_video.flatten(0, 1),
                torch.randn_like(pred_image_or_video.flatten(0, 1)),
                next_timestep.flatten(0, 1)
            ).unflatten(0, noise.shape[:2])
            output_list.append(noisy_image_or_video)

        # [B, T, F, C, H, W]
        output = torch.stack(output_list, dim=1)
        return output

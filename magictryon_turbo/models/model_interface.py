from magictryon_turbo.scheduler import SchedulerInterface
from abc import abstractmethod, ABC
from typing import List, Optional
import torch
import types


class DiffusionModelInterface(ABC, torch.nn.Module):
    scheduler: SchedulerInterface

    @abstractmethod
    def forward(
        self, noisy_image_or_video: torch.Tensor, conditional_dict: dict,
        timestep: torch.Tensor, kv_cache: Optional[List[dict]] = None,
        crossattn_cache: Optional[List[dict]] = None,
        current_start: Optional[int] = None,
        current_end: Optional[int] = None
    ) -> torch.Tensor:
        """
        A method to run diffusion model.
        Input:
            - noisy_image_or_video: a tensor with shape [B, F, C, H, W] where the number of frame is 1 for images.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - timestep: a tensor with shape [B, F]  where the number of frame is 1 for images.
            all data should be on the same device as the model.
            - kv_cache: a list of dictionaries containing the key and value tensors for each attention layer.
            - current_start: the start index of the current frame in the sequence.
            - current_end: the end index of the current frame in the sequence.
        Output: a tensor with shape [B, F, C, H, W] where the number of frame is 1 for images.
        We always expect a X0 prediction form for the output.
        """
        pass

    def get_scheduler(self) -> SchedulerInterface:
        """
        Update the current scheduler with the interface's static method
        """
        scheduler = self.scheduler
        scheduler.convert_x0_to_noise = types.MethodType(
            SchedulerInterface.convert_x0_to_noise, scheduler)
        scheduler.convert_noise_to_x0 = types.MethodType(
            SchedulerInterface.convert_noise_to_x0, scheduler)
        scheduler.convert_velocity_to_x0 = types.MethodType(
            SchedulerInterface.convert_velocity_to_x0, scheduler)
        self.scheduler = scheduler
        return scheduler

    def post_init(self):
        """
        A few custom initialization steps that should be called after the object is created.
        Currently, the only one we have is to bind a few methods to scheduler.
        We can gradually add more methods here if needed.
        """
        self.get_scheduler()

    def set_module_grad(self, module_grad: dict) -> None:
        """
        Adjusts the state of each module in the object.

        Parameters:
        - module_grad (dict): A dictionary where each key is the name of a module (as an attribute of the object),
          and each value is a bool indicating whether the module's parameters require gradients.

        Functionality:
        For each module name in the dictionary:
        - Updates whether its parameters require gradients based on 'is_trainable'.
        """
        for k, is_trainable in module_grad.items():
            getattr(self, k).requires_grad_(is_trainable)

    @abstractmethod
    def enable_gradient_checkpointing(self) -> None:
        """
        Activates gradient checkpointing for the current model (may be referred to as *activation checkpointing* or
        *checkpoint activations* in other frameworks).
        """
        pass


class VAEInterface(ABC, torch.nn.Module):
    @abstractmethod
    def decode_to_pixel(self, latent: torch.Tensor) -> torch.Tensor:
        """
        A method to decode a latent representation to an image or video.
        Input: a tensor with shape [B, F // T, C, H // S, W // S] where T and S are temporal and spatial compression factors.
        Output: a tensor with shape [B, F, C, H, W] where the number of frame is 1 for images.
        """
        pass


class TextEncoderInterface(ABC, torch.nn.Module):
    @abstractmethod
    def forward(self, text_prompts: List[str]) -> dict:
        """
        A method to tokenize text prompts with a tokenizer and encode them into a latent representation.
        Input: a list of strings.
        Output: a dictionary containing the encoded representation of the text prompts.
        """
        pass


class InferencePipelineInterface(ABC):
    @abstractmethod
    def inference_with_trajectory(self, noise: torch.Tensor, conditional_dict: dict) -> torch.Tensor:
        """
        Run inference with the given diffusion / distilled generators.
        Input:
            - noise: a tensor sampled from N(0, 1) with shape [B, F, C, H, W] where the number of frame is 1 for images.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
        Output:
            - output: a tensor with shape [B, T, F, C, H, W].
            T is the total number of timesteps. output[0] is a pure noise and output[i] and i>0
            represents the x0 prediction at each timestep.
        """

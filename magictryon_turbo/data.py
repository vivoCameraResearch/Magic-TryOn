from magictryon_turbo.ode_data.create_lmdb_iterative import get_array_shape_from_lmdb, retrieve_row_from_lmdb
from torch.utils.data import Dataset
import numpy as np
import torch
import lmdb
from typing import Any, Dict


class TextDataset(Dataset):
    def __init__(self, data_path):
        self.texts = []
        with open(data_path, "r") as f:
            for line in f:
                self.texts.append(line.strip())

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]


class ODERegressionDataset(Dataset):
    def __init__(self, data_path, max_pair=int(1e8)):
        self.data_dict = torch.load(data_path, weights_only=False)
        self.max_pair = max_pair

    def __len__(self):
        return min(len(self.data_dict['prompts']), self.max_pair)

    def __getitem__(self, idx):
        """
        Outputs:
            - prompts: List of Strings
            - latents: Tensor of shape (num_denoising_steps, num_frames, num_channels, height, width). It is ordered from pure noise to clean image.
        """
        return {
            "prompts": self.data_dict['prompts'][idx],
            "ode_latent": self.data_dict['latents'][idx].squeeze(0),
        }


# class ODERegressionLMDBDataset(Dataset):
#     def __init__(self, data_path: str, max_pair: int = int(1e8)):
#         self.env = lmdb.open(data_path, readonly=True,
#                              lock=False, readahead=False, meminit=False)

#         self.latents_shape = get_array_shape_from_lmdb(self.env, 'latents')
#         self.max_pair = max_pair

#     def __len__(self):
#         return min(self.latents_shape[0], self.max_pair)

#     def __getitem__(self, idx):
#         """
#         Outputs:
#             - prompts: List of Strings
#             - latents: Tensor of shape (num_denoising_steps, num_frames, num_channels, height, width). It is ordered from pure noise to clean image.
#         """
#         latents = retrieve_row_from_lmdb(
#             self.env,
#             "latents", np.float16, idx, shape=self.latents_shape[1:]
#         )

#         if len(latents.shape) == 4:
#             latents = latents[None, ...]

#         prompts = retrieve_row_from_lmdb(
#             self.env,
#             "prompts", str, idx
#         )
#         return {
#             "prompts": prompts,
#             "ode_latent": torch.tensor(latents, dtype=torch.float32)
#         }


class ODERegressionLMDBDataset(Dataset):
    """
    读取 8 个字段：
        -- prompt:                 str
        -- stored_data:            (1, 4, 16, 21, 104, 78)
        -- masked_video_latents:   (2, 16, 21, 104, 78)
        -- pose_latents:           (2, 16, 21, 104, 78)   
        -- mask_input:             (2, 4, 21, 104, 78)   
        -- cloth_latents:          (2, 16, 1, 104, 78)    
        -- cloth_line_latents:     (2, 16, 1, 104, 78)   
        -- image_clip:             (2, 257, 1280)
    """

    def __init__(self, data_path: str, max_pair: int = int(1e8)) -> None:
        # ---------- 打开 LMDB ----------
        self.env = lmdb.open(
            data_path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        # ---------- 逐字段保存形状 ----------
        self.stored_data_shape = get_array_shape_from_lmdb(self.env, "latents")
        self.masked_latent_shape = get_array_shape_from_lmdb(self.env, "masked_video_latents")
        self.pose_latent_shape   = get_array_shape_from_lmdb(self.env, "pose_latents")
        self.mask_input_shape    = get_array_shape_from_lmdb(self.env, "mask_input")
        self.cloth_latent_shape  = get_array_shape_from_lmdb(self.env, "cloth_latents")
        self.cloth_line_latent_shape = get_array_shape_from_lmdb(self.env, "cloth_line_latents")
        self.clip_shape          = get_array_shape_from_lmdb(self.env, "image_clip")

        # 样本总数由主数组第一维决定
        self.max_pair = max_pair
        self._length  = min(self.stored_data_shape[0], self.max_pair)


    # ---------- Dataset 标准接口 ----------
    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        try:
            # -------- prompt --------
            prompt = retrieve_row_from_lmdb(self.env, "prompts", str, idx)

            # -------- 主 / 相关 latent --------
            latents = retrieve_row_from_lmdb(
                self.env, "latents", np.float16, idx, shape=self.stored_data_shape[1:]
            )
            masked_video_latents = retrieve_row_from_lmdb(
                self.env, "masked_video_latents", np.float16, idx, shape=self.masked_latent_shape[1:]
            )

            # -------- 其他辅助输入 --------
            pose_latents = retrieve_row_from_lmdb(
                self.env, "pose_latents", np.float16, idx, shape=self.pose_latent_shape[1:]
            )
            mask_input = retrieve_row_from_lmdb(
                self.env, "mask_input", np.float16, idx, shape=self.mask_input_shape[1:]
            )
            cloth_latents = retrieve_row_from_lmdb(
                self.env, "cloth_latents", np.float16, idx, shape=self.cloth_latent_shape[1:]
            )
            cloth_line_latents = retrieve_row_from_lmdb(
                self.env, "cloth_line_latents", np.float16, idx, shape=self.cloth_line_latent_shape[1:]
            )
            image_clip = retrieve_row_from_lmdb(
                self.env, "image_clip", np.float16, idx, shape=self.clip_shape[1:]
            )

            # -------- 打包返回 --------
            sample = {
                "prompt": prompt,
                "ode_latent": torch.tensor(latents, dtype=torch.float32),
                "masked_video_latents": torch.tensor(masked_video_latents, dtype=torch.float32),
                "pose_latents": torch.tensor(pose_latents, dtype=torch.float32),
                "mask_input": torch.tensor(mask_input, dtype=torch.float32),
                "cloth_latents": torch.tensor(cloth_latents, dtype=torch.float32),
                "cloth_line_latents": torch.tensor(cloth_line_latents, dtype=torch.float32),
                "image_clip": torch.tensor(image_clip, dtype=torch.float32),
            }

            # 选填打印检查
            # print(sample["ode_latent"].size(),
            #     sample["masked_video_latents"].size(),
            #     sample["mask_input"].size(),
            #     sample["cloth_latents"].size(),
            #     sample["cloth_line_latents"].size(),
            #     )
            return sample

        except Exception as e:
            print(f"[Warning] Failed to load sample at idx={idx}: {e}")
            return self.__getitem__((idx + 1) % self._length)  # 尝试加载下一个

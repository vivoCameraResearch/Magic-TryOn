from magictryon_turbo.models.wan.bidirectional_inference import BidirectionalInferencePipeline
from huggingface_hub import hf_hub_download
from diffusers.utils import export_to_video
from magictryon_turbo.models.wan.videox_fun.models.wan_image_encoder import CLIPModel
from magictryon_turbo.models.wan.videox_fun.utils.utils import get_video_to_video_latent_tryon_full
from magictryon_turbo.data import TextDataset
from omegaconf import OmegaConf
from tqdm import tqdm
import argparse
import torch
import json
import os
import cv2

#### support def
def get_video_properties(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("无法打开视频文件")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    cap.release()
    
    return width, height, frame_count, fps

def get_description_from_json(cloth_image, json_path):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for item in data:
            if item['name'] == cloth_image:
                return item['describe']
        return None  # 若未找到
    except Exception as e:
        print(f"读取 JSON 文件失败: {e}")
        return None

def get_jsonl_line_with_keys(filepath, line_number):
    with open(filepath, 'r', encoding='utf-8') as f:
        for current_line, line in enumerate(f):
            if current_line == line_number:
                data = json.loads(line.strip())
                return data  # 返回字典
    raise IndexError(f"Line {line_number} not found in file.")

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str)
parser.add_argument("--checkpoint_folder", type=str)
parser.add_argument("--output_folder", type=str)
parser.add_argument("--prompt_file_path", type=str)

args = parser.parse_args()

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_grad_enabled(False)

#### load dmd pipeline
config = OmegaConf.load(args.config_path)
pipe = BidirectionalInferencePipeline(config, device="cuda")
state_dict = torch.load(os.path.join(args.checkpoint_folder, "model.pt"), map_location="cpu")[
    'generator']
pipe.generator.load_state_dict(state_dict)
pipe = pipe.to(device="cuda", dtype=torch.bfloat16)


# for testdata_id in range(0,181):
for testdata_id in range(0,181):
    print(f"Processing testdata_id {testdata_id}")

    # Load dataset
    testdata = get_jsonl_line_with_keys(
            "ViViD-Test/test_data_180_unpaired.jsonl", 
            testdata_id
    )

    video_id = testdata['video']
    image_id = testdata['image']
    front_seqs = testdata['front_seqs']
    subdir = testdata['subdir']
    paired_image_id = testdata['paired_image']

    org_video_path = os.path.join("../TryOn_dataset/ViViD-Test", subdir, "videos", video_id)
    masked_video_path = os.path.join("../TryOn_dataset/ViViD-Test", subdir, "agnostic", video_id)
    mask_video_path = os.path.join("../TryOn_dataset/ViViD-Test", subdir, "agnostic_mask", video_id)
    pose_video_path = os.path.join("../TryOn_dataset/ViViD-Test", subdir, "densepose", video_id)

    cloth_image_path = os.path.join("../TryOn_dataset/ViViD-Test", subdir, "images", paired_image_id)
    cloth_line_image_path = os.path.join("../TryOn_dataset/ViViD-Test", subdir, "images_anilines", paired_image_id)

    cloth_text = get_description_from_json(
        paired_image_id, 
        os.path.join("../TryOn_dataset/ViViD", subdir, "caption_qwen.json")
    )

    no_all_frame = True
    width, height, frame_count, v_fps = get_video_properties(org_video_path)

    sample_size = [height, width]
    video_length = frame_count
    fps = v_fps
    prompt = "Model is wearing " + cloth_text

    temporal_compression_ratio = 4
    spatial_compression_ratio = 8
    
    video_length = int((video_length - 1) // temporal_compression_ratio * temporal_compression_ratio) + 1 if video_length != 1 else 1
    

    input_video, masked_video, mask_video, pose_video, _, clip_image, cloth_image, cloth_line_image = get_video_to_video_latent_tryon_full(
                                            org_video_path, masked_video_path, mask_video_path, pose_video_path, 
                                            video_length=video_length, sample_size=sample_size, fps=fps, ref_image=cloth_image_path, line_image_path = cloth_line_image_path)

    
    latent_height = sample_size[0] // spatial_compression_ratio
    latent_width = sample_size[1] // spatial_compression_ratio
    
    if no_all_frame:
        start_frame = 20 + front_seqs[0]
        end_frame = 21 + front_seqs[1]

        input_video = input_video[:, :, start_frame:end_frame, :, :]
        masked_video = masked_video[:, :, start_frame:end_frame, :, :]
        mask_video = mask_video[:, :, start_frame:end_frame, :, :]
        pose_video = pose_video[:, :, start_frame:end_frame, :, :]

        video_length = end_frame - start_frame
        
    latent_frames = (video_length - 1) // temporal_compression_ratio + 1
    os.makedirs(args.output_folder, exist_ok=True)

    #### inference start 
    video = pipe.inference(
            noise=torch.randn(
                1, latent_frames, 16, latent_height, latent_width, generator=torch.Generator(device="cuda").manual_seed(42),
                dtype=torch.bfloat16, device="cuda"
            ),
            text_prompts=prompt,
            ####condition input####
            masked_video=masked_video,
            mask_video=mask_video,
            pose_video=pose_video,
            cloth_image=cloth_image,
            cloth_line_image=cloth_line_image,
            clip_image=clip_image
        )[0].permute(0, 2, 3, 1).cpu().numpy()

    prefix = video_id.split('.')[0]
    export_to_video(
        video, os.path.join(args.output_folder,  prefix + ".mp4"), fps=fps)


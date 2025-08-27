import os
from typing import Optional
from PIL import Image
import torch

# Image generation uses a diffusers pipeline passed in

def generate_image(pipe_sdxl, prompt: str, filename: str = "stage_image.png") -> str:
    """Generate an image using the provided SDXL pipeline.

    Args:
        pipe_sdxl: A diffusers AutoPipelineForText2Image instance moved to device.
        prompt: Full text prompt.
        filename: Output filename within static/ dir.
    Returns:
        Path to the saved image (within static/ directory).
    """
    short_prompt = " ".join(prompt.split(" ")[:50])
    full_prompt = (
        f"Акварельная иллюстрация в стиле импрессионизма, русская народная сказка — {short_prompt}. "
        "Impressionist watercolor, dreamy light, Russian fairy tale, 4k"
    )
    image = pipe_sdxl(prompt=full_prompt, num_inference_steps=4, guidance_scale=0.0).images[0]

    os.makedirs("static", exist_ok=True)
    img_path = f"static/{filename}"
    image.save(img_path)

    print(f"🖼️ Иллюстрация сгенерирована: {img_path}")

    try:
        image.show()
    except Exception as e:
        print(f"⚠️ Не удалось открыть окно просмотра изображения: {e}")

    return img_path


def generate_video(pipe_svd, image_path: str, prompt: str, output_path: str = "video.mp4") -> Optional[str]:
    """Generate a short video using the provided SVD pipeline.

    Args:
        pipe_svd: StableVideoDiffusionPipeline instance moved to device.
        image_path: Path to input image.
        prompt: Text (used only for logging/metadata now).
        output_path: Path to resulting mp4 file.
    Returns:
        Path to video, or None if pipeline is not provided.
    """
    if pipe_svd is None:
        return None

    from diffusers.utils import export_to_video

    short_prompt = " ".join(prompt.split()[:70])
    image = Image.open(image_path).convert("RGB")
    image = image.resize((1024, 576))

    print(f"🎥 Генерация видео: {short_prompt[:60]}...")

    frames = pipe_svd(
        image,
        decode_chunk_size=8,
        generator=torch.manual_seed(42),
        motion_bucket_id=120,
        noise_aug_strength=0.02,
        num_frames=25,
        frame_rate=6
    ).frames[0]

    export_to_video(frames, output_path, fps=6)
    print(f"✅ Видео сохранено: {output_path}")
    return output_path

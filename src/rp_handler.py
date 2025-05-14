'''Worker handler to generate SDXL images and upload them to GCS or return base64 fallback 2.'''

import os
import base64
import shutil
import concurrent.futures
import logging
import json
import uuid
import torch
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, AutoencoderKL
from diffusers.utils import load_image
from diffusers import (
    PNDMScheduler,
    LMSDiscreteScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    DPMSolverMultistepScheduler,
)
import runpod
from runpod.serverless.utils.rp_validator import validate
from rp_schemas import INPUT_SCHEMA
from google.cloud import storage
from google.oauth2 import service_account

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sdxl_worker")

torch.cuda.empty_cache()

# ------------------------------- Model Handler ------------------------------ #
class ModelHandler:
    def __init__(self):
        self.base = None
        self.refiner = None
        self.load_models()

    def load_base(self):
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
        base_pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", vae=vae,
            torch_dtype=torch.float16, variant="fp16", use_safetensors=True, add_watermarker=False
        )
        base_pipe = base_pipe.to("cuda", silence_dtype_warnings=True)
        base_pipe.enable_xformers_memory_efficient_attention()
        return base_pipe

    def load_refiner(self):
        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
        refiner_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0", vae=vae,
            torch_dtype=torch.float16, variant="fp16", use_safetensors=True, add_watermarker=False
        )
        refiner_pipe = refiner_pipe.to("cuda", silence_dtype_warnings=True)
        refiner_pipe.enable_xformers_memory_efficient_attention()
        return refiner_pipe

    def load_models(self):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_base = executor.submit(self.load_base)
            future_refiner = executor.submit(self.load_refiner)
            self.base = future_base.result()
            self.refiner = future_refiner.result()

MODELS = ModelHandler()

# ---------------------------------- Upload ---------------------------------- #
def upload_to_gcs(local_path, bucket_name, destination_blob):
    try:
        credentials_json = os.getenv("GCS_CREDENTIALS_JSON")
        credentials_dict = json.loads(credentials_json)
        credentials = service_account.Credentials.from_service_account_info(credentials_dict)

        client = storage.Client(credentials=credentials)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(destination_blob)
        blob.upload_from_filename(local_path)
        url = blob.generate_signed_url(version="v4", expiration=3600, method="GET")
        logger.info(f"✅ Uploaded to GCS: {url}")
        return url
    except Exception as e:
        logger.warning(f"⚠️ GCS upload failed: {e}. Falling back to base64.")
        with open(local_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode("utf-8")
            return f"data:image/png;base64,{image_data}"

# ---------------------------------- Helper ---------------------------------- #
def _save_and_upload_images(images, job_id):
    output_dir = f"/workspace/output/{job_id}"
    os.makedirs(output_dir, exist_ok=True)
    image_urls = []

    bucket_name = os.getenv("GCS_BUCKET_NAME")

    for index, image in enumerate(images):
        filename = f"{index}.png"
        image_path = os.path.join(output_dir, filename)
        image.save(image_path)

        if bucket_name:
            blob_path = f"output/{job_id}/{filename}"
            image_url = upload_to_gcs(image_path, bucket_name, blob_path)
        else:
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode("utf-8")
                image_url = f"data:image/png;base64,{image_data}"

        image_urls.append(image_url)

    logger.info(f"🧹 Cleaning up: {output_dir}")
    try:
        shutil.rmtree(output_dir)
    except Exception as e:
        logger.warning(f"⚠️ Cleanup failed: {e}")

    return image_urls

# ---------------------------- Scheduler Builder ---------------------------- #
def make_scheduler(name, config):
    return {
        "PNDM": PNDMScheduler.from_config(config),
        "KLMS": LMSDiscreteScheduler.from_config(config),
        "DDIM": DDIMScheduler.from_config(config),
        "K_EULER": EulerDiscreteScheduler.from_config(config),
        "DPMSolverMultistep": DPMSolverMultistepScheduler.from_config(config),
    }[name]

# ----------------------------- Main Entry Point ---------------------------- #
@torch.inference_mode()
def generate_image(job):
    job_input = job["input"]
    validated_input = validate(job_input, INPUT_SCHEMA)
    if 'errors' in validated_input:
        return {"error": validated_input['errors']}
    job_input = validated_input['validated_input']

    starting_image = job_input['image_url']
    job_id = job['id']

    if job_input['seed'] is None:
        job_input['seed'] = int.from_bytes(os.urandom(2), "big")

    generator = torch.Generator("cuda").manual_seed(job_input['seed'])
    MODELS.base.scheduler = make_scheduler(job_input['scheduler'], MODELS.base.scheduler.config)

    if starting_image:
        init_image = load_image(starting_image).convert("RGB")
        output = MODELS.refiner(
            prompt=job_input['prompt'],
            num_inference_steps=job_input['refiner_inference_steps'],
            strength=job_input['strength'],
            image=init_image,
            generator=generator
        ).images
    else:
        image = MODELS.base(
            prompt=job_input['prompt'],
            negative_prompt=job_input['negative_prompt'],
            height=job_input['height'],
            width=job_input['width'],
            num_inference_steps=job_input['num_inference_steps'],
            guidance_scale=job_input['guidance_scale'],
            denoising_end=job_input['high_noise_frac'],
            output_type="latent",
            num_images_per_prompt=job_input['num_images'],
            generator=generator
        ).images

        try:
            output = MODELS.refiner(
                prompt=job_input['prompt'],
                num_inference_steps=job_input['refiner_inference_steps'],
                strength=job_input['strength'],
                image=image,
                num_images_per_prompt=job_input['num_images'],
                generator=generator
            ).images
        except RuntimeError as err:
            return {
                "error": f"RuntimeError: {err}",
                "refresh_worker": True
            }

    image_urls = _save_and_upload_images(output, job_id)

    return {
        "images": image_urls,
        "image_url": image_urls[0],
        "seed": job_input['seed'],
        "refresh_worker": starting_image is not None
    }

runpod.serverless.start({"handler": generate_image})

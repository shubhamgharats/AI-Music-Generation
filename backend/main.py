import modal
import os
import uuid
import base64
import requests
import torch
from typing import List
import boto3
import inspect

from pydantic import BaseModel
from prompts import PROMPT_GENERATOR_PROMPT, LYRICS_GENERATOR_PROMPT


app = modal.App("music-generator")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "ffmpeg")
    .pip_install_from_requirements("requirements.txt")
    .pip_install("torchcodec")
    .run_commands([
        "git clone https://github.com/ace-step/ACE-Step.git /tmp/ACE-Step",
        "cd /tmp/ACE-Step && pip install ."
    ])
    .env({"HF_HOME": "/.cache/huggingface"})
    .add_local_python_source("prompts")
)

model_volume = modal.Volume.from_name(
    "ace-step-models", create_if_missing=True
)
hf_volume = modal.Volume.from_name(
    "qwen-hf-cache", create_if_missing=True
)

music_gen_secrets = modal.Secret.from_name("music-gen-secrett")


class AudioGenerationBase(BaseModel):
    audio_duration: float = 180.0
    seed: int = -1
    guidance_scale: float = 15.0
    infer_step: int = 60
    instrumental: bool = False


class GenerateFromDescriptionRequests(AudioGenerationBase):
    full_described_song: str


class GenerateWithCustomLyricsRequests(AudioGenerationBase):
    prompt: str
    lyrics: str


class GenerateWithDescribedLyricsRequest(AudioGenerationBase):
    prompt: str
    described_lyrics: str


class GenerateMusicResponseS3(BaseModel):
    s3_key: str
    cover_image_s3_key: str
    categories: List[str]
    lyrics: str


class GenerateMusicResponse(BaseModel):
    audio_data: str


@app.cls(
    image=image,
    gpu="L40S",
    volumes={"/models": model_volume, "/.cache/huggingface": hf_volume},
    secrets=[music_gen_secrets],
    scaledown_window=15
)
class MusicGenServer:

    @modal.enter()
    def load_model(self):
        from acestep.pipeline_ace_step import ACEStepPipeline
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from diffusers import AutoPipelineForText2Image

        self.music_model = ACEStepPipeline(
            checkpoint_dir="/models",
            dtype="bfloat16",
            torch_compile=False,
            cpu_offload=False,
            overlapped_decode=False
        )
        print(inspect.signature(self.music_model.__call__))
        model_id = "Qwen/Qwen2-7B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        self.llm_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto",
            cache_dir="/.cache/huggingface"
        )

        self.image_pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo",
            torch_dtype=torch.float16,
            variant="fp16"
        )
        self.image_pipe.to("cuda")

    def prompt_qwen(self, question: str):
        messages = [{"role": "user", "content": question}]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = self.tokenizer(
            [text], return_tensors="pt"
        ).to(self.llm_model.device)

        generated_ids = self.llm_model.generate(
            **model_inputs,
            max_new_tokens=512
        )

        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(
                model_inputs.input_ids, generated_ids
            )
        ]

        response = self.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]

        return response

    def generate_prompt(self, description: str):
        full_prompt = PROMPT_GENERATOR_PROMPT.format(
            user_prompt=description
        )
        return self.prompt_qwen(full_prompt)

    def generate_lyrics(self, description: str):
        full_prompt = LYRICS_GENERATOR_PROMPT.format(
            description=description
        )
        return self.prompt_qwen(full_prompt)
    
    def generate_categories(self, description: str) -> List[str]:
        prompt = (
            "Based on the following music description, list 3-5 relevant "
            "genres or categories as a comma-separated list. "
            "For example: Pop, Electronic, Sad, 80s.\n"
            f"Description: '{description}'"
        )

        response_text = self.prompt_qwen(prompt)
        categories = [
            cat.strip()
            for cat in response_text.split(",")
            if cat.strip()
        ]

        return categories

    def generate_and_upload_to_s3(
        self,
        prompt: str,
        lyrics: str,
        instrumental: bool,
        audio_duration: float,
        infer_step: int,
        guidance_scale: float,
        description_for_categorization: str,) -> GenerateMusicResponseS3:

        final_lyrics = "[instrumental]" if instrumental else lyrics

        s3_client = boto3.client("s3")
        bucket_name = os.environ["S3_BUCKET_NAME"]

        output_dir = "/tmp/outputs"
        os.makedirs(output_dir, exist_ok=True)

        # -------- AUDIO GENERATION --------
        output_path = os.path.join(output_dir, f"{uuid.uuid4()}.wav")



        self.music_model(
            prompt=prompt,
            lyrics=final_lyrics,
            audio_duration=audio_duration,
            infer_step=infer_step,
            guidance_scale=guidance_scale,
            save_path=output_path,
        )

        audio_s3_key = f"{uuid.uuid4()}.wav"
        s3_client.upload_file(output_path, bucket_name, audio_s3_key)
        os.remove(output_path)

        # -------- IMAGE GENERATION --------
        image = self.image_pipe(
            prompt=f"{prompt}, album cover art",
            num_inference_steps=2,
            guidance_scale=0.0,
        ).images[0]

        image_output_path = os.path.join(output_dir, f"{uuid.uuid4()}.png")
        image.save(image_output_path)

        image_s3_key = f"{uuid.uuid4()}.png"
        s3_client.upload_file(image_output_path, bucket_name, image_s3_key)
        os.remove(image_output_path)

        # -------- CATEGORY GENERATION --------
        categories = self.generate_categories(description_for_categorization)

        return GenerateMusicResponseS3(
            s3_key=audio_s3_key,
            cover_image_s3_key=image_s3_key,
            categories=categories,
            lyrics=final_lyrics,
        )
    
    @modal.fastapi_endpoint(method="POST", requires_proxy_auth=True)
    def generate(self) -> GenerateMusicResponse:
        output_dir = "/tmp/outputs"
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(
            output_dir,
            f"{uuid.uuid4()}.wav"
        )

        self.music_model(
            prompt="Cuban music, salsa, son, Afro-Cuban, traditional Cuban",
            lyrics="your_lyrics_here",
            audio_duration=180,
            infer_step=60,
            guidance_scale=15,
            save_path=output_path
        )

        with open(output_path, "rb") as f:
            audio_bytes = f.read()

        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

        os.remove(output_path)

        return GenerateMusicResponse(audio_data=audio_b64)

    
    @modal.fastapi_endpoint(method="POST", requires_proxy_auth=True)
    def generate_from_description(
        self,
        request: GenerateFromDescriptionRequests,
    ) -> GenerateMusicResponseS3:

        prompt = self.generate_prompt(request.full_described_song)

        lyrics = ""
        if not request.instrumental:
            lyrics = self.generate_lyrics(request.full_described_song)

        return self.generate_and_upload_to_s3(
        prompt=request.prompt,
        lyrics=request.lyrics,
        instrumental=request.instrumental,
        audio_duration=request.audio_duration,
        infer_step=request.infer_step,
        guidance_scale=request.guidance_scale,
        description_for_categorization=request.prompt,
)




    @modal.fastapi_endpoint(method="POST",requires_proxy_auth=True)
    def generate_with_lyrics(
        self,
        request: GenerateWithCustomLyricsRequests,
    ) -> GenerateMusicResponseS3:

        return self.generate_and_upload_to_s3(
            prompt=request.prompt,
            lyrics=request.lyrics,
            instrumental=request.instrumental,
            audio_duration=request.audio_duration,
            infer_step=request.infer_step,
            guidance_scale=request.guidance_scale,
            description_for_categorization=request.prompt,
        )

    @modal.fastapi_endpoint(method="POST", requires_proxy_auth=True)
    def generate_with_described_lyrics(
        self,
        request: GenerateWithDescribedLyricsRequest,
    ) -> GenerateMusicResponseS3:

        lyrics = self.generate_lyrics(request.described_lyrics)

        return self.generate_and_upload_to_s3(
            prompt=request.prompt,
            lyrics=lyrics,
            instrumental=request.instrumental,
            audio_duration=request.audio_duration,
            infer_step=request.infer_step,
            guidance_scale=request.guidance_scale,
            description_for_categorization=request.prompt,
        )




@app.local_entrypoint()
def main():
    server = MusicGenServer()

    endpoint_url = server.generate_with_described_lyrics.get_web_url()

    request_data = GenerateWithDescribedLyricsRequest(
        prompt="Rave, funk, 140BPM, Hardcore",
        described_lyrics="Hindi-Rap-song on Mumbai streets",
        guidance_scale=15,
    )

    headers = {
        "Modal-Key": "wk-BDTBC8cp6k7LdbCaau5txi",
        "Modal-Secret": "ws-pO5xKqzqJbfV0GLCYVSGQi"
    }

    payload = request_data.model_dump()

    response = requests.post(
        endpoint_url,
        json=payload,
        headers=headers
    )

    print("STATUS:", response.status_code)
    print("RESPONSE:", response.text)

    response.raise_for_status()

    result = GenerateMusicResponseS3(**response.json())

    print("S3 Key:", result.s3_key)
    print("Cover:", result.cover_image_s3_key)
    print("Categories:", result.categories)
    print("\nGenerated Lyrics:\n")
    print(result.lyrics)
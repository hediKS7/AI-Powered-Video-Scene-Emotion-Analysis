import torch
from PIL import Image
from tqdm import tqdm
import pandas as pd
import os
os.environ["USE_TF"] = "0"

from transformers import BlipProcessor, BlipForConditionalGeneration

print("Transformers BLIP imported successfully!")


device = "cuda" if torch.cuda.is_available() else "cpu"


class ImageCaptioner:
    def __init__(self, model_name="Salesforce/blip-image-captioning-base"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(self.device)
    
    def generate_caption(self, img_path):
        try:
            image = Image.open(img_path).convert("RGB")
            inputs = self.processor(image, return_tensors="pt").to(self.device)
            out = self.model.generate(**inputs)
            caption = self.processor.decode(out[0], skip_special_tokens=True)
            return caption
        except Exception as e:
            print(f"Error on {img_path}: {e}")
            return "caption error"
    
    def caption_scenes(self, scenes_df):
        scenes_df = scenes_df.copy()
        tqdm.pandas()
        scenes_df['caption'] = scenes_df['path'].progress_apply(self.generate_caption)
        return scenes_df
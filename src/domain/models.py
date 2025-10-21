import torch
from transformers import AutoImageProcessor, SiglipForImageClassification

from settings import ModelConfig

config = ModelConfig()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_processor():
    # Load Model and Processor
    try:
        print(f"Loading processor from: {config.MODEL_IDENTIFIER}")
        processor = AutoImageProcessor.from_pretrained(config.MODEL_IDENTIFIER, use_fast=True)
        print("Processor loaded successfully.")

    except Exception as e:
        print(f"Error loading processor: {e}")
        exit()

    return processor


def load_model():
    try:
        print(f"Loading model from: {config.MODEL_IDENTIFIER}")
        model = SiglipForImageClassification.from_pretrained(config.MODEL_IDENTIFIER)
        model.to(device)
        model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        exit()

    return model

class ModelWrapper(torch.nn.Module):
    # model wrapper for gradcam
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        outputs = self.model(x)
        return outputs.logits

    def get_image_size(self):
        return self.model.config.image_size
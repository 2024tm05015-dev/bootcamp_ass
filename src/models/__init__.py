def __init__(self):
    self.api_key = os.getenv("HUGGINGFACE_API_KEY", "")
    self.model_name = os.getenv(
        "VISION_MODEL",
        "Salesforce/blip-image-captioning-base"
    )
    self.api_url = f"https://api-inference.huggingface.co/models/{self.model_name}"

    print("DEBUG VLM HF key loaded:", bool(self.api_key))
import os
from dotenv import load_dotenv
import requests

def test_models(model_name):
	print(f"Testing model {model_name}...")

	# Load environment variables from .env file
	load_dotenv()

	API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
	if not API_TOKEN:
		raise ValueError("Hugging Face API token not found. Please set it in the environment or .env file.")
	API_URL = f"https://api-inference.huggingface.co/models/{model_name}"
	headers = {
		"Authorization": f"Bearer {API_TOKEN}",
		"Content-Type": "application/json",
		"x-wait-for-model": "true"	
		}

	def query(payload):
		response = requests.post(API_URL, headers=headers, json=payload)
		return response.content

	image_bytes = query({
		"inputs": "Outlines of cartoon boat for coloring",
		"negative_prompts": ["blurry", "not cartoon", "not for kids"],
	})


	# check for errors, e.g. {"error":"Model stable-diffusion-v1-5/stable-diffusion-v1-5 is currently loading","estimated_time":945.1328125}
	if b"error" in image_bytes:
		print("Error:", image_bytes)
		return

	# remove slashes from model name
	model_name_s = model_name.replace("/", "_")

	with open(f"output_{model_name_s}.jpg", "wb") as f:
		f.write(image_bytes)

	print(f"Saved output to output_{model_name_s}.jpg")

if __name__ == "__main__":
	test_models("openfree/claude-monet")
	test_models("stable-diffusion-v1-5/stable-diffusion-v1-5")
	test_models("black-forest-labs/FLUX.1-schnell")		


# In order for this code to run, the user must use their own Openai API key.
# You can create them in your account:
# https://platform.openai.com/settings/organization/api-keys
# Create a .env file to define the keys, and use them in this code (see sample image).
# The variable name in the code is: OPENAI_KEY (see sample .env file).
# It opens a website with port 5000 (http://127.0.0.1:5000/).
# You can upload a jpg (or png) file. It will spit out hashtags to select and copy.
import os
from flask import Flask, render_template, request, jsonify
import base64
import requests
from dotenv import load_dotenv
app = Flask(__name__)
load_dotenv()

#encoding image and returning
# This function takes the file path (image_path) of an image as input,
# reads the image file in binary mode, and encodes its contents into a Base64 string.
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# This is a Flask route decorator specifying the types of HTTP methods (GET and POST).
# The path of the route is the root directory.
@app.route('/', methods=['GET', 'POST'])

# This Flask view function index handles both GET and POST requests for rendering and processing an image upload page.
# When a user submits an image, the image is saved locally and converted to a Base64 string.
# The encoded image is sent to OpenAI's API along with a prompt to generate hashtags for the image.
# The API response is parsed to extract the generated hashtags.
def index():
    base64_image = None
    if request.method == 'POST':
        print('\n posting\n\n')
        image = request.files['image']
        image.save("uploaded_image.jpg")

        base64_image = encode_image("uploaded_image.jpg")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('OPENAI_KEY')}"
        }

        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a hashtag generation model. When you get an image as input, your response should always contain exactly 30 hashtags separated by commas."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Provide the hashtags for this image:"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 300
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        hashtags = response.json().get("choices")[0].get("message").get("content").split(',')
        return render_template('index.html', hashtags=hashtags, base64_image=base64_image)

    return render_template('index.html', hashtags=None)


# Starting the Flask development server with debugging enabled.
if __name__ == '__main__':
    app.run(debug=True)

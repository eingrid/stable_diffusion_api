
import sys
# sys.path.append('../dreambooth-dreamfusion/dreamfusion')
# sys.path.append('../dreambooth-dreamfusion/')
# from config import key, secret
print("Relative Imports")
import json

from flask import Flask, jsonify, request
from train import train as train_dreambooth
from PIL import Image
import os
import io
from io import BytesIO
import string
import random
from diffusers import StableDiffusionPipeline
import torch
from flask import send_file
import base64

app = Flask(__name__)

@app.route('/dreambooth_train', methods=['POST'])
def dreambooth():
    if request.method == 'POST':

        training = os.path.exists('training_started')

        if training:
            return jsonify({
            'status': 'Success',
            'path_to_imgs': '',
            'path_to_model':''
            })
        open('training_started', 'w+')
        training = True
        
        json_data = request.get_json()
        concept_name = json_data.get('concept_name')
        sd_version = json_data.get('sd_version')
        model_name = json_data.get('model_name')
        images_path = json_data.get('images_path')

        files = request.files.getlist('files')
        for file in files:
            file.save(os.path.join(images_path, file.filename))
            
        prompt = json_data.get('prompt')
        model_path = ''
        if images_path != '':
            prompt = f'a {concept_name}'
            if sd_version != '1.5':
                which_model = 'v2-1-512'
            else:
                which_model = 'v1-5-512'
            model_path = train_dreambooth(images_path,prompt,object_name = concept_name, which_model=which_model)
            if isinstance(model_path,list):
                model_path = model_path[0]

        file_name = 'model_map.json'
        if os.path.exists(file_name):
            # If the file exists, load the existing data from the file
            with open(file_name, 'r') as f:
                model_map = json.load(f)
        else:
            # If the file does not exist, initialize an empty dictionary
            model_map = {}

        # Add the new data to the model map dictionary
        model_map.update({f'{model_name}':model_path})

        # Save the updated model map dictionary to the file
        with open(file_name, 'w') as f:
            json.dump(model_map, f)

        print('finished trainin dreambooth')
        print(model_path)    
        print('########')
        print('model_path', model_path)
        os.remove('training_started')

    
    return jsonify({
        'status': 'Success',
        'path_to_model':model_path,
        'path_to_imgs':''
    })


from base64 import encodebytes
def get_response_image(pil_img):
    byte_arr = io.BytesIO()
    pil_img.save(byte_arr, format='PNG') # convert the PIL image to byte array
    encoded_img = encodebytes(byte_arr.getvalue()).decode('ascii') # encode as base64
    return encoded_img

@app.route('/generate_images', methods = ['POST'])
def generate_images():
    print('generating ')
    json_data = request.get_json()
    
    model_path = json_data['model_path']
    style = json_data['style']
    concept_name = model_path.split('_')[-1]
    if style == 'Cyberpunk':
        prompt = f'a photo of {concept_name} in a cyberpunk style, realistic'
    elif style == 'Cartoon' :
        prompt = f'a photo of {concept_name} in a anime style, vivid'
    elif style=='Picasso':
        prompt = f'a painting of {concept_name} in Picasso style, colorful'
    torch.cuda.empty_cache()
    print(prompt)
    pipe = StableDiffusionPipeline.from_pretrained(
    f'/content/dreambooth-training/output_model_{concept_name}',
    torch_dtype=torch.float16,
    ).to("cuda")
    n_samples = 8
    print('loaded mode')
    prompt = [prompt] * n_samples
    images = pipe(prompt, num_inference_steps=100,).images
    encoded_imgs = []
    for image in images:
        encoded_imgs.append(get_response_image(image))
    
    return jsonify({'encoded_images': encoded_imgs})


@app.route('/temp', methods = ['POST'])
def temp():
    return jsonify({
        'status': 'Success',
        'path_to_model':'some_path',
        'path_to_imgs':''
    })

if __name__ == '__main__':
    from waitress import serve
    if os.path.exists('training_started'):
        os.remove('training_started')
    serve(app, host="0.0.0.0", port=5000)

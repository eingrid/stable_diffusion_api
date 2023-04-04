import gradio as gr
from PIL import Image 
import string
import random
import os
import requests
from io import BytesIO
import io
import string, random
import json
import base64

name_model_to_path_mapping = {}

def make_request(prompt, images_path, concept_name, sd_version,model_name):
    url = 'http://localhost:5000/dreambooth_train'
    myobj = {'prompt': prompt,'images_path':images_path,'concept_name':concept_name, 'sd_version': sd_version, '3d_framework': 'dreamfusion','model_name':model_name}
    print(myobj)
    return requests.post(url, json = myobj)

def save_images(input,concept_name):
    concept_dir = os.path.join('.',concept_name)
    os.mkdir(concept_name)

    for j,file in enumerate(input):
        image = Image.open(file.name)
        image.save(os.path.join(concept_dir,concept_name+f'{j}.jpg'))

    return concept_dir
        
    


def save_and_request(input,prompt, sd_version,name_model):
    print('save')
    global name_model_to_path_mapping
    concept_name = ''.join(random.choices(string.ascii_uppercase, k=5))
    print(prompt,concept_name)
    concept_dir  = save_images(input,concept_name)

    request_result = make_request(prompt,concept_dir,concept_name, sd_version,model_name=name_model)
    res = request_result.json()
    name_model_to_path_mapping[name_model] = res['path_to_model']
    print('model map in save_and_request',model_map)
    model_map[f'{name_model}'] = res['path_to_model']
    return res['path_to_model'], res['path_to_imgs']
    

def generate_images(style,model_name):

    url = 'http://localhost:5000/generate_images'
    # for os.listdir(;)
    model_path = model_map.get(model_name)
    obj = {'style':style,'model_path':model_path}
    response_data = requests.post(url, json = obj).json()
    encoded_images = response_data['encoded_images']
    pil_images = []
    for encoded_image in encoded_images:
        decoded_img = base64.b64decode(encoded_image)
        pil_images.append(Image.open(io.BytesIO(decoded_img)))

    return pil_images

def update_models_f(x):
    print('model map in update_models_f',model_map)
    
    return gr.Dropdown.update(choices=[k for k in model_map])


options = ['Picasso', 'Cyberpunk', 'Neon', 'Cartoon']
def main():
    with gr.Blocks() as demo:
        with gr.Column():
            with gr.Row():
                image_loader = gr.File(file_count="multiple")
            
            with gr.Column():
                prompt = gr.Textbox(label='Prompt')
                name_model = gr.Textbox(label='Model Name')
                sd_version = gr.Dropdown(['1.5', '2.0'], label='StableDiffusion Version', value='1.5')
            submit = gr.Button('Submit')
            
            update_models = gr.Button('Update Models')

            submit.click(save_and_request,
            inputs=[image_loader, prompt, sd_version, name_model],
            )
            
            text_options = gr.Dropdown(options, label="Top 5 options")
            model = gr.Dropdown([k for k in model_map], label="Choose a model")
            update_models.click(update_models_f,outputs = model)
            
            generate = gr.Button("Generate images")
            gallery = gr.Gallery(label="Generated images", show_label=False, elem_id="gallery").style(grid=[2], height="auto")
            generate.click(generate_images,inputs=[text_options,model],outputs=[gallery])
            
    demo.queue()
    demo.launch(server_name="0.0.0.0",share=True)


if __name__ == '__main__':
    file_name = 'model_map.json'
    if os.path.exists(file_name):
        # If the file exists, load the existing data from the file
        with open(file_name, 'r') as f:
            model_map = json.load(f)
    else:
        model_map = {} 
    main()
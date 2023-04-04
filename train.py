from subprocess import getoutput
import os

gpu_info = getoutput('nvidia-smi')
if("A10G" in gpu_info):
    which_gpu = "A10G"
    os.system(f"pip install -q https://github.com/camenduru/stable-diffusion-webui-colab/releases/download/0.0.15/xformers-0.0.15.dev0+4c06c79.d20221205-cp38-cp38-linux_x86_64.whl")
elif("T4" in gpu_info):
    which_gpu = "T4"
    os.system(f"pip install -q https://github.com/camenduru/stable-diffusion-webui-colab/releases/download/0.0.15/xformers-0.0.15.dev0+1515f77.d20221130-cp38-cp38-linux_x86_64.whl")
else:
    which_gpu = "CPU"
    
import gradio as gr
from pathlib import Path
import argparse
import shutil
from train_dreambooth import run_training
from convertosd import convert
from PIL import Image
from slugify import slugify
import requests
import torch
import zipfile
import tarfile
import urllib.parse
import gc
from diffusers import StableDiffusionPipeline
from huggingface_hub import snapshot_download, update_repo_visibility, HfApi

# import boto3

is_spaces = True if "SPACE_ID" in os.environ else False
if(is_spaces):
    is_shared_ui = True if "multimodalart/dreambooth-training" in os.environ['SPACE_ID'] else False
else:
    is_shared_ui = False

is_gpu_associated = torch.cuda.is_available()

import torch

def generate(prompt, steps,n_images,model_path='stabilityai/stable-diffusion-2-1-base'):
    torch.cuda.empty_cache()
    from diffusers import StableDiffusionPipeline
    pipe_is_set = False
    if(not pipe_is_set):
        global pipe
        pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
        pipe = pipe.to("cuda")
        pipe_is_set = True

    prompt = [prompt] * n_images
    image = pipe(prompt, num_inference_steps=steps).images 
    return(image)

def pad_image(image):
    w, h = image.size
    if w == h:
        return image
    elif w > h:
        new_image = Image.new(image.mode, (w, w), (0, 0, 0))
        new_image.paste(image, (0, (w - h) // 2))
        return new_image
    else:
        new_image = Image.new(image.mode, (h, h), (0, 0, 0))
        new_image.paste(image, ((h - w) // 2, 0))
        return new_image


if(is_gpu_associated):
    model_v1 = snapshot_download(repo_id="multimodalart/sd-fine-tunable")
    model_v2 = snapshot_download(repo_id="stabilityai/stable-diffusion-2-1", ignore_patterns=["*.ckpt", "*.safetensors"])
    model_v2_512 = snapshot_download(repo_id="stabilityai/stable-diffusion-2-1-base", ignore_patterns=["*.ckpt", "*.safetensors"])
    safety_checker = snapshot_download(repo_id="multimodalart/sd-sc")
    model_to_load = model_v2_512

def train(path_to_images=None,prompt=None,object_name=None,bucket_name=None,prefix=None, which_model = 'v2-1-512'):
    
    if is_shared_ui:
        raise gr.Error("This Space only works in duplicated instances")
    if not is_gpu_associated:
        raise gr.Error("Please associate a T4 or A10G GPU for this Space")
    hf_token = ''
    model_name = 'Model_name'
    print('hf_token',hf_token)
    print('model_name',model_name)

    print('is_spaces',is_spaces)
    if(is_spaces):
        remove_attribution_after = inputs[-6]
    else:
        remove_attribution_after = False
    
    # if(remove_attribution_after):
    #     validate_model_upload(hf_token, model_name)
    
    torch.cuda.empty_cache()
    if 'pipe' in globals():
        global pipe, pipe_is_set
        del pipe
        pipe_is_set = False
        gc.collect()
        
    if os.path.exists("output_model"): shutil.rmtree('output_model')
    if os.path.exists("instance_images"): shutil.rmtree('instance_images')
    if os.path.exists("diffusers_model.tar"): os.remove("diffusers_model.tar")
    if os.path.exists("model.ckpt"): os.remove("model.ckpt")
    if os.path.exists("hastrained.success"): os.remove("hastrained.success")
    file_counter = 0
    if which_model != "v2-1-512":
        model_to_load=model_v1
    else:
      model_to_load = model_v2_512
    print(which_model,'which_model')
    resolution = 512 if which_model != "v2-1-768" else 768
    maximum_concepts = 1

    
    # # set up a client to interact with the S3 service
    # s3 = boto3.client('s3')

    # # set the name of your S3 bucket and the prefix for the images you want to read
    # bucket_name = 'my-bucket'
    # prefix = 'path/to/images/'

    # # use the client to list all of the objects in the specified bucket and prefix
    # objects = s3.list_objects(Bucket=bucket_name, Prefix=prefix)

    
    for j,image in enumerate(os.listdir(path_to_images)):
        os.makedirs('instance_images',exist_ok=True)
        file = Image.open(os.path.join(path_to_images, image))
        image = pad_image(file)
        image = image.resize((resolution, resolution))
        image = image.convert('RGB')
        image.save(f'instance_images/{prompt}_({j+1}).jpg', format="JPEG", quality = 100)
        
        file_counter += 1
            
    # for i, input in enumerate(inputs):
    #     if(i < maximum_concepts-1):
    #         if(input):
    #             os.makedirs('instance_images',exist_ok=True)
    #             files = inputs[i+(maximum_concepts*2)]
    #             prompt ='photo of person zxcv'
    #             if(prompt == "" or prompt == None):
    #                 raise gr.Error("You forgot to define your concept prompt")
    #             for j, file_temp in enumerate(files):
    #                 file = Image.open(file_temp.name)
    #                 image = pad_image(file)
    #                 image = image.resize((resolution, resolution))
    #                 extension = file_temp.name.split(".")[1]
    #                 image = image.convert('RGB')
    #                 image.save(f'instance_images/{prompt}_({j+1}).jpg', format="JPEG", quality = 100)
    #                 file_counter += 1

    os.makedirs(f'output_model_{object_name}',exist_ok=True)
    output_dir = f'./output_model_{object_name}'
    uses_custom = False
    print(uses_custom,'uses_custom')
    type_of_thing = 'object'
    print(type_of_thing,'type_of_thing')
    experimental_face_improvement = False
    print(experimental_face_improvement,'experimental_face_improvement')
    
    if(uses_custom):
        Training_Steps = int(inputs[-3])
        print(Training_Steps,'Training_Steps')
        Train_text_encoder_for = int(inputs[-2])
        print(Train_text_encoder_for,'Train_text_encoder_for')
    else:
        if(type_of_thing == "object"):
            Train_text_encoder_for=30
            
        elif(type_of_thing == "style"):
            Train_text_encoder_for=15
            
        elif(type_of_thing == "person"):
            Train_text_encoder_for=70
        
        Training_Steps = file_counter*150
        if(type_of_thing == "person" and Training_Steps > 2600):
            Training_Steps = 2600 #Avoid overfitting on people's faces
    stptxt = int((Training_Steps*Train_text_encoder_for)/100)
    gradient_checkpointing = True if (experimental_face_improvement or which_model != "v1-5") else False 
    cache_latents = True if which_model != "v1-5" else False
    if (type_of_thing == "object" or type_of_thing == "style" or (type_of_thing == "person" and not experimental_face_improvement)):
        args_general = argparse.Namespace(
            image_captions_filename = True,
            train_text_encoder = True if stptxt > 0 else False,
            stop_text_encoder_training = stptxt,
            save_n_steps = 0,
            pretrained_model_name_or_path = model_to_load,
            instance_data_dir="instance_images",
            class_data_dir=None,
            output_dir=output_dir,
            instance_prompt=prompt,
            seed=42,
            resolution=resolution,
            mixed_precision="fp16",
            train_batch_size=1,
            gradient_accumulation_steps=1,
            use_8bit_adam=True,
            learning_rate=2e-6,
            lr_scheduler="polynomial",
            lr_warmup_steps = 0,
            max_train_steps=Training_Steps,     
            gradient_checkpointing=gradient_checkpointing,
            cache_latents=cache_latents,
        )
        print("Starting single training...")
        lock_file = open("intraining.lock", "w")
        lock_file.close()
        run_training(args_general)
    else:
        
        args_general = argparse.Namespace(
            image_captions_filename = True,
            train_text_encoder = True if stptxt > 0 else False,
            stop_text_encoder_training = stptxt,
            save_n_steps = 0,
            pretrained_model_name_or_path = model_to_load,
            instance_data_dir="instance_images",
            class_data_dir="Mix",
            output_dir=output_dir,
            with_prior_preservation=True,
            prior_loss_weight=1.0,
            instance_prompt=prompt,
            seed=42,
            resolution=resolution,
            mixed_precision="fp16",
            train_batch_size=1,
            gradient_accumulation_steps=1,
            use_8bit_adam=True,
            learning_rate=2e-6,
            lr_scheduler="polynomial",
            lr_warmup_steps = 0,
            max_train_steps=Training_Steps,
            num_class_images=200,     
            gradient_checkpointing=gradient_checkpointing,
            cache_latents=cache_latents,
        )
        print("Starting multi-training...")
        lock_file = open("intraining.lock", "w")
        lock_file.close()
        run_training(args_general)
    gc.collect()
    torch.cuda.empty_cache()
    if(which_model == "v1-5"):
        print("Adding Safety Checker to the model...")
        shutil.copytree(f"{safety_checker}/feature_extractor", "output_model/feature_extractor", dirs_exist_ok=True)
        shutil.copytree(f"{safety_checker}/safety_checker", "output_model/safety_checker", dirs_exist_ok=True)
        shutil.copy(f"model_index.json", "output_model/model_index.json")

    if(not remove_attribution_after):
        print("Archiving model file...")
        with tarfile.open("diffusers_model.tar", "w") as tar:
            tar.add(f'output_model_{object_name}', arcname=os.path.basename(f"output_model_{object_name}"))
        if os.path.exists("intraining.lock"): os.remove("intraining.lock")
        trained_file = open("hastrained.success", "w")
        trained_file.close()
        print("Training completed!")
        return [
            output_dir,
            gr.update(visible=True, value=["diffusers_model.tar"]), #result
            gr.update(visible=True), #try_your_model
            gr.update(visible=True), #push_to_hub
            gr.update(visible=True), #convert_button
            gr.update(visible=False), #training_ongoing
            gr.update(visible=True) #completed_training
        ]
    else:
        where_to_upload = inputs[-8]
        push(model_name, where_to_upload, hf_token, which_model, True)
        hardware_url = f"https://huggingface.co/spaces/{os.environ['SPACE_ID']}/hardware"
        headers = { "authorization" : f"Bearer {hf_token}"}
        body = {'flavor': 'cpu-basic'}
        requests.post(hardware_url, json = body, headers=headers)
    return f'model_{object_name}' 

if __name__ == '__main__':
    train('/content/images/',prompt='photo of person zxcv',object_name=None)

## DreamBooth Training

This repository contains code for training a multimodal image generation model using Hugging Face's Transformers library. The trained model can be used to generate images in different styles.

### Installation

To use this repository, clone it to your Google Colab account and run the following commands:

`chmod a+x ./install.sh`

`./install.sh`



This will clone the repository from `https://huggingface.co/spaces/multimodalart/dreambooth-training` with all the required dependencies and code for the UI and API.

### Running the API and UI

To launch the API, run:

`python dreambooth-training/api.py`


To launch the UI, run:

`python dreambooth-training/ui.py`


### Usage

To use the DreamBooth, first upload your images to the upload section. Then, give a name for your model. After the training is completed, press the "update model" button to enable choosing it in the dropdown menu.

Select the model with the name you specified earlier and the style in which you want to generate images, and press the "generate images" button. The generated images will be displayed in the UI.

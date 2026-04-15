from config import *  
import os   
import importlib 
from PIL import Image 
from torchvision import transforms  
import torch.backends.mkldnn as mkldnn  
mkldnn.enabled = True  
import time
SuperResolution = getattr(importlib.import_module('super-resolution'), 'SuperResolution')
DenseNet = getattr(importlib.import_module('densenet'), 'DenseNet')
GradCAM = getattr(importlib.import_module('densenet'), 'GradCAM')

deepfake_detection_model = DenseNet() 
super_resolution_model = SuperResolution() 
grad_cam_model = GradCAM()  
for index in range(1, len(os.listdir(data_dir)) + 1):
    input_image_path = f"{data_dir}/{index}.{img_format}"
    input_image = Image.open(input_image_path).convert('RGB')n
    input_image_tensor = transforms.ToTensor()(input_image).unsqueeze(0).to(torch.device(device))
    super_res_tensor = super_resolution_model.super_resolve_tensor(input_image_tensor)
    super_resolution_model.save_super_res_image(super_res_tensor, index)
    prediction = deepfake_detection_model.predict(index)
deepfake_detection_model.add_results_to_json()

# ------------------------------ Part 2: Grad-CAM --------------------
for index in range(1, len(os.listdir(data_dir)) + 1):
    grad_cam_model.save_gradCAM(index)


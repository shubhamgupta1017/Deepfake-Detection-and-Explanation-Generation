import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
import os
import torch.nn as nn
from torchvision import models, transforms
import json
from config import *

class DenseNet:
    def __init__(self):
        self.results = []
        self.model = models.densenet121(weights=None)
        self.model.classifier = nn.Sequential(
            nn.Linear(self.model.classifier.in_features, 1), 
            nn.Sigmoid()                     
        )

        self.model.load_state_dict(torch.load(denseNet_weights_path, map_location=device))
        self.model = self.model.to(device)  
        self.model.eval()  
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  
            transforms.ToTensor(),  
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
        ])
        
    def load_image(self, image_path):
        img = Image.open(image_path).convert('RGB') 
        img_tensor = self.transform(img).unsqueeze(0).to(device)  
        return img_tensor

    def predict(self, index):
        image_path = f"{super_res_dir}/{index}.{img_format}" 
        image_tensor = self.load_image(image_path) 
        output = self.model(image_tensor) 
        prediction = "real" if output < 0.5 else "fake" 
        self.add_to_results(index, prediction) 
        return prediction
    
    def add_to_results(self, index, prediction):
        self.results.append({"index": index, "prediction": prediction})
    
    def add_results_to_json(self):
        with open(task1_output_file_path, "w") as f:
            json.dump(self.results, f, indent=4)
class GradCAM(DenseNet): 
    def __init__(self):
        super().__init__() 
        os.makedirs(grad_cam_dir, exist_ok=True) 
    def register_hooks(self):
        self.activations = []  
        self.gradients = []  
        last_conv_layer = self.model.features.denseblock4.denselayer12.conv2  
        last_conv_layer.register_forward_hook(self.hook_activation) 
        last_conv_layer.register_full_backward_hook(self.hook_gradient) 
        return self.activations, self.gradients

    def hook_activation(self, module, input, output):
        self.activations.append(output) 
    def hook_gradient(self, module, grad_in, grad_out):
        self.gradients.append(grad_out[0]) 
    def gradcam_heatmap(self, image_tensor):
        self.activations, self.gradients = self.register_hooks()  
        output = self.model(image_tensor)
        self.model.zero_grad() 
        
        output = output.squeeze() 
        output.backward() 
    
        self.gradients = self.gradients[0]  
        self.activations = self.activations[0] 
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        heatmap = torch.mean(self.activations * weights, dim=1).squeeze().cpu().detach().numpy()
        heatmap = np.maximum(heatmap, 0) 
        heatmap = cv2.resize(heatmap, (image_tensor.shape[2], image_tensor.shape[3]))
        max_val = np.max(heatmap)
        if max_val != 0:
            heatmap /= max_val 
        else:
            heatmap = np.zeros_like(heatmap)  
        return heatmap
    def save_gradCAM(self, index):
        image_path = f"{super_res_dir}/{index}.{img_format}" 
        image_tensor = self.load_image(image_path) 
        heatmap = self.gradcam_heatmap(image_tensor) 
        
        img = Image.open(image_path).convert('RGB') 
        img = np.array(img)  
        
        heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))  
        heatmap = np.clip(heatmap_resized, 0, 1) 
        heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)  
        heatmap_rgb = heatmap_rgb.astype(np.float32) / 255.0 
        superimposed_img = heatmap_rgb * 0.4 + img / 255.0 
        max_loc = np.unravel_index(np.argmax(heatmap_resized), heatmap_resized.shape)
        y, x = max_loc
        
        half_box_size = 40
        x1, x2 = max(0, x - half_box_size), min(img.shape[1], x + half_box_size)
        y1, y2 = max(0, y - half_box_size), min(img.shape[0], y + half_box_size)
        
        cv2.rectangle(superimposed_img, (x1, y1), (x2, y2), color=(1, 0, 0), thickness=2) 
        superimposed_img = np.clip(superimposed_img * 255, 0, 255).astype('uint8')
        pil_image = Image.fromarray(superimposed_img)

        pil_image.save(f'{grad_cam_dir}/{index}.{img_format}')

if __name__ == "__main__":
    gradcam = GradCAM()  
    index=1
    gradcam.save_gradCAM(index)  

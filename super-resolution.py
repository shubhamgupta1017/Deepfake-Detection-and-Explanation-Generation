import os
import torch
from torchvision import transforms
from PIL import Image
from config import *  
from models.RealESRGAN import RealESRGAN 

class SuperResolution:
    def __init__(self):
        os.makedirs(f"{super_res_dir}", exist_ok=True)
        self.device = torch.device(device) 
        super_res_model = RealESRGAN(self.device, scale=4)
        self.rrdb_model = super_res_model.model.to(self.device)  
        self.rrdb_model.load_state_dict(torch.load(rrdb_weights_path, map_location=self.device))
        self.rrdb_model.eval()

    def super_resolve_tensor(self, low_res_image_tensor):
        low_res_image_tensor = low_res_image_tensor.to(self.device)
        with torch.no_grad(): 
            super_res_image_tensor = self.rrdb_model(low_res_image_tensor)
            super_res_image_tensor = torch.clamp(super_res_image_tensor.squeeze(0).cpu(), 0, 1)
            return super_res_image_tensor.cpu()
    
    def save_super_res_image(self, super_res_tensor, index):
        super_res_image = transforms.ToPILImage()(super_res_tensor)
        super_res_image.save(f"{super_res_dir}/{index}.png")

        
if __name__ == "__main__":
    index=1
    image_path = f"data/{index}.{img_format}"
    low_res_image = Image.open(image_path)
    low_res_image_tensor = transforms.ToTensor()(low_res_image).unsqueeze(0).to(device)
    super_resolute = SuperResolution()
    super_res_tensor = super_resolute.super_resolve_tensor(low_res_image_tensor)
    super_resolute.save_super_res_image(super_res_tensor, index)  # Saving the  image
    print("Super resolution completed and saved.")

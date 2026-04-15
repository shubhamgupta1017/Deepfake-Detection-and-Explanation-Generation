import torch

deepfake_detection = "deepfake-detection"  
super_resolute = "super-resolute" 

device = "cuda" if torch.cuda.is_available() else "cpu" 
print(f"Using device: {device}") 
rrdb_weights_path = "models/weights/task_1/rrdb.pth"
denseNet_weights_path = "models/weights/task_1/denseNet.pth"  

task1_output_file_path = "output/84_task1.json"  
data_dir = "data" 
temp_dir = "output/temp" 
output_dir = "output"  
super_res_dir = "output/temp/super_res" 
grad_cam_dir = "output/temp/grad_cam" 
img_format = "png"  

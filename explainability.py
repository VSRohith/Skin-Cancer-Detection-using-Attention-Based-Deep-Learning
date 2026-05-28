import torch
import torch.nn.functional as F
import numpy as np
import cv2

# This function creates the "Glow" (Heatmap)
def generate_heatmap(model, img_tensor, target_layer):
    model.eval()
    features = []
    
    def hook_feature(module, input, output):
        features.append(output)
    
    handle = target_layer.register_forward_hook(hook_feature)
    output = model(img_tensor)
    handle.remove()
    
    target_class = output.argmax(dim=1).item()
    output[:, target_class].backward()
    
    weights = torch.mean(features[0], dim=(2, 3), keepdim=True)
    cam = torch.sum(weights * features[0], dim=1).squeeze().detach().cpu().numpy()
    
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
    return cam
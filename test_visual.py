import torch
import matplotlib.pyplot as plt
from PIL import Image
from model_attention import SkinAttentionModel
from data_loader import val_transform
from explainability import generate_heatmap
import os

# 1. Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Load trained model
model = SkinAttentionModel(num_classes=7).to(device)
model.load_state_dict(torch.load('skin_attention_model.pth', map_location=device))
model.eval()

# 3. Image folder path
img_dir = r"D:\CLG DOCS\Mini-project\Method1-code\Skin_Cancer_Project\data"

# Check if folder exists
if not os.path.exists(img_dir):
    raise FileNotFoundError(f"Image folder not found: {img_dir}")

# Get first image
img_list = os.listdir(img_dir)

if len(img_list) == 0:
    raise Exception("No images found in the folder!")

img_path = os.path.join(img_dir, img_list[0])

print(f"Using image: {img_path}")

# 4. Process image
img_pil = Image.open(img_path).convert('RGB')
img_tensor = val_transform(img_pil).unsqueeze(0).to(device)

# 5. Generate heatmap
target_layer = model.features[-1]
heatmap = generate_heatmap(model, img_tensor, target_layer)

# 6. Display results
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(img_pil)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("AI Attention Heatmap")
plt.imshow(img_pil)
plt.imshow(heatmap, cmap='jet', alpha=0.5)
plt.axis('off')

# 7. Save output
plt.savefig('my_first_heatmap.png')
print("✅ Heatmap saved as 'my_first_heatmap.png'")

plt.show()
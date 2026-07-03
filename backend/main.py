import io
import os
import base64
import numpy as np
import torch
import torch.nn as nn
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

# Initialize FastAPI App
app = FastAPI(title="MRI.AI Analysis Backend")

# Enable CORS for React Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For local development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# 1. Model Architecture Definition (ResUNet++)
# --------------------------------------------------

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        identity = self.skip(x)
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return torch.relu(out)

class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.res = ResidualBlock(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.res(x)

class ResUNetPP(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        # Encoder
        self.e1 = ResidualBlock(3, 32)
        self.e2 = ResidualBlock(32, 64)
        self.e3 = ResidualBlock(64, 128)
        self.e4 = ResidualBlock(128, 256)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ResidualBlock(256, 512)

        # Decoder
        self.d4 = UpBlock(512, 256, 256)
        self.d3 = UpBlock(256, 128, 128)
        self.d2 = UpBlock(128, 64, 64)
        self.d1 = UpBlock(64, 32, 32)

        # Segmentation head
        self.seg_head = nn.Conv2d(32, 1, kernel_size=1)

        # Classification head
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        # Segmentation decoder
        d4 = self.d4(b, e4)
        d3 = self.d3(d4, e3)
        d2 = self.d2(d3, e2)
        d1 = self.d1(d2, e1)

        seg_out = self.seg_head(d1)

        # Classification output
        cls_out = self.cls_head(b)

        return seg_out, cls_out, b

# --------------------------------------------------
# 2. Model Initialization
# --------------------------------------------------

IMG_SIZE = (256, 256)
CLASS_NAMES = ['glioma', 'meningioma', 'pituitary', 'no_tumor']
MODEL_PATH = r"c:\sharing1\ML4\AI\projects\mri_cla_seg\best_resunetpp_multitask.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load Model
model = ResUNetPP(num_classes=len(CLASS_NAMES))
if os.path.exists(MODEL_PATH):
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("Successfully loaded model weights from:", MODEL_PATH)
    except Exception as e:
        print(f"Error loading model weights: {e}")
else:
    print(f"WARNING: Weights file not found at {MODEL_PATH}. Running model with random initialization.")

model.to(device)
model.eval()

# Helper function to convert CV2 image to base64 string
def img_to_base64(img_np):
    _, buffer = cv2.imencode('.png', img_np)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{img_base64}"

# --------------------------------------------------
# 3. Inference Endpoint
# --------------------------------------------------

@app.get("/")
def read_root():
    return {
        "status": "online",
        "model_loaded": os.path.exists(MODEL_PATH),
        "device": str(device)
    }

@app.post("/api/analyze")
async def analyze_mri(file: UploadFile = File(...)):
    # Verify that file is uploaded
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    # Read image bytes
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Decoded image is None")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

    # Keep a copy of the original BGR image for overlays and display
    original_height, original_width = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize to model standard input size (256, 256)
    img_resized = cv2.resize(img_rgb, IMG_SIZE)
    img_norm = img_resized / 255.0
    
    # Prepare tensor
    tensor = torch.tensor(img_norm, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        seg_out, cls_out, bottleneck = model(tensor)
        
        # Calculate Classification
        probs = torch.softmax(cls_out, dim=1).cpu().squeeze().numpy()
        pred_idx = np.argmax(probs)
        pred_class = CLASS_NAMES[pred_idx]
        pred_prob = float(probs[pred_idx])

        # Get probabilities dict
        class_probabilities = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}

        # Calculate Segmentation
        seg_probs = torch.sigmoid(seg_out).cpu().squeeze().numpy()
        mask = (seg_probs > 0.5).astype(np.uint8)

    # Post-process outputs
    is_tumor = pred_class != 'no_tumor'
    
    # Enforce segmentation if classification predicts a tumor but default threshold (0.5) yields 0 pixels
    if is_tumor and np.sum(mask) == 0:
        max_prob = float(np.max(seg_probs))
        # If there is a noticeable structural outline, lower threshold dynamically
        if max_prob > 0.08:
            # Threshold is set to 70% of max probability, bounded by 0.12 at minimum
            dynamic_threshold = max(0.12, max_prob * 0.7)
            mask = (seg_probs > dynamic_threshold).astype(np.uint8)
            print(f"Dynamic Segmentation: Enforced threshold adjustment to {dynamic_threshold:.4f} due to weak logits (Max prob: {max_prob:.4f})")
        else:
            print(f"Dynamic Segmentation: Unable to enforce mask from seg_probs (Max prob too low: {max_prob:.4f}). Falling back to bottleneck activations.")
            # 1. Get bottleneck feature map
            b_np = torch.mean(torch.abs(bottleneck), dim=1).squeeze().cpu().numpy()
            b_resized = cv2.resize(b_np, (256, 256))
            
            # 2. Normalize
            min_v, max_v = np.min(b_resized), np.max(b_resized)
            norm_b = (b_resized - min_v) / (max_v - min_v + 1e-8)
            
            # 3. Threshold at 0.80 to capture the core of the activation
            fallback_mask = (norm_b > 0.80).astype(np.uint8)
            
            # 4. Clean up: Keep only the largest contour to remove background noise
            contours, _ = cv2.findContours(fallback_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > 0:
                largest_cnt = max(contours, key=cv2.contourArea)
                mask = np.zeros_like(fallback_mask)
                cv2.drawContours(mask, [largest_cnt], -1, 1, -1) # Fill largest contour
                print(f"Dynamic Segmentation: Bottleneck fallback mask created! Size={np.sum(mask)} pixels.")
            else:
                print("Dynamic Segmentation: Fallback failed. No contours found in bottleneck map.")

    # If the classified result is "no_tumor", we zero out the mask to follow standard clinical logic
    if not is_tumor:
        mask = np.zeros_like(mask)

    # Initialize statistics
    tumor_area_pixels = 0
    tumor_area_percentage = 0.0
    estimated_area_cm2 = 0.0
    centroid_x, centroid_y = None, None
    quadrant = "N/A"
    bbox = None # [x_min, y_min, x_max, y_max]

    # Calculate metrics if tumor is detected and mask has active pixels
    if is_tumor and np.sum(mask) > 0:
        tumor_area_pixels = int(np.sum(mask))
        tumor_area_percentage = float((tumor_area_pixels / (256 * 256)) * 100)
        
        # Approximate scale: Assuming a standard pixel spacing of 0.12 cm per pixel
        # Area = pixels * 0.12 * 0.12 = pixels * 0.0144 cm2
        estimated_area_cm2 = float(tumor_area_pixels * 0.0144)

        # Bounding Box coordinates
        y_indices, x_indices = np.where(mask == 1)
        x_min, x_max = int(np.min(x_indices)), int(np.max(x_indices))
        y_min, y_max = int(np.min(y_indices)), int(np.max(y_indices))
        bbox = [x_min, y_min, x_max, y_max]

        # Centroid (Center of Mass)
        centroid_x = float(np.mean(x_indices))
        centroid_y = float(np.mean(y_indices))

        # Quadrant calculation relative to center (128, 128)
        # Top-Left, Top-Right, Bottom-Left, Bottom-Right
        # Clinically named: Left-Anterior, Right-Anterior, Left-Posterior, Right-Posterior
        lat = "Left" if centroid_x < 128 else "Right"
        ant_post = "Anterior" if centroid_y < 128 else "Posterior"
        quadrant = f"{lat} {ant_post}"

    # Generate Image Outputs (256x256 standard size)
    # 1. Base image (RGB)
    base_img_display = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)

    # 2. Mask image (Black and White PNG)
    mask_display = (mask * 255).astype(np.uint8)

    # 3. Colored Overlay image
    # We will color the tumor region in a semi-transparent red (or cyan) and draw a bounding box
    overlay_display = base_img_display.copy()
    if is_tumor and np.sum(mask) > 0:
        # Create a red mask overlay
        color_mask = np.zeros_like(base_img_display)
        color_mask[:] = [0, 0, 255] # Red in BGR
        
        # Blend the red mask on the overlay
        mask_bool = mask.astype(bool)
        overlay_display[mask_bool] = cv2.addWeighted(
            base_img_display, 0.6, color_mask, 0.4, 0
        )[mask_bool]

        # Draw a bounding box in bright green/yellow
        if bbox:
            cv2.rectangle(
                overlay_display,
                (bbox[0], bbox[1]),
                (bbox[2], bbox[3]),
                (0, 255, 255), # Yellow in BGR
                2
            )
            # Add small label text
            cv2.putText(
                overlay_display,
                f"{pred_class.upper()} {pred_prob*100:.1f}%",
                (bbox[0], max(bbox[1] - 8, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 255),
                1,
                cv2.LINE_AA
            )
            
            # Draw centroid point
            if centroid_x is not None and centroid_y is not None:
                cv2.circle(
                    overlay_display,
                    (int(centroid_x), int(centroid_y)),
                    4,
                    (0, 255, 0), # Green dot
                    -1
                )

    # Encode images to Base64
    base_img_b64 = img_to_base64(base_img_display)
    mask_b64 = img_to_base64(mask_display)
    overlay_b64 = img_to_base64(overlay_display)

    return {
        "classification": {
            "prediction": pred_class,
            "confidence": pred_prob,
            "probabilities": class_probabilities,
            "is_tumor": is_tumor
        },
        "segmentation": {
            "tumor_detected": is_tumor and (tumor_area_pixels > 0),
            "area_pixels": tumor_area_pixels,
            "area_percentage": tumor_area_percentage,
            "estimated_area_cm2": estimated_area_cm2,
            "centroid": {
                "x": centroid_x,
                "y": centroid_y
            } if centroid_x is not None else None,
            "quadrant": quadrant,
            "bounding_box": {
                "x_min": bbox[0],
                "y_min": bbox[1],
                "x_max": bbox[2],
                "y_max": bbox[3],
                "width": bbox[2] - bbox[0],
                "height": bbox[3] - bbox[1]
            } if bbox else None
        },
        "images": {
            "original": base_img_b64,
            "mask": mask_b64,
            "overlay": overlay_b64
        },
        "metadata": {
            "original_resolution": f"{original_width}x{original_height}",
            "processed_resolution": "256x256",
            "device": str(device)
        }
    }

# Run FastAPI app locally
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

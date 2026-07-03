# MRI.AI - Multitask Brain MRI Classification & Segmentation
**MRI.AI** is a professional, high-fidelity diagnostic web platform powered by a deep learning multitask convolutional neural network. The platform enables clinical practitioners to upload brain MRI scans, perform real-time tumor classification (into Glioma, Meningioma, Pituitary, or Normal/No Tumor), segment/outline visual tumor borders, calculate quantitative anatomical metrics, and compile print-ready diagnostic clinical reports.
---
## Key Features
- **Multitask Neural Inference**: Employs a dual-task encoder-decoder (`ResUNet++`) to execute simultaneous classification and segmentation in a single forward pass.
- **Dynamic Attention Fallback**: If the classification head identifies a tumor but the segmentation logits fall below the threshold (due to weak contrast or domain differences), the system extracts the **Bottleneck feature activation map** (which represents the classifier's spatial attention), normalizes it, and overlays the contour. This enforces clinical consistency between the classifier and segmenter.
- **Quantitative Analytics Core**: Calculates segmented tumor pixel counts, percentage of brain section affected, physical size estimation (in cm²), center-of-mass centroid coordinates, and anatomical hemisphere quadrant.
- **Interactive Scan Viewer**: Dynamic comparison canvas supporting original scan, binary mask, and colored overlay tab views with smooth semi-transparent opacity blending controls.
- **Printable Clinical Diagnostic Reports**: Stylesheets optimized for single-page A4 printing, outputting institutional headers, patient record cards, scan overlays, structured metric grids, attending physician notes, signature sign-offs, and clinical disclaimers.
- **LAN Startup Launcher**: Configured local startup script that binds the servers to `0.0.0.0`, dynamically resolving your local IP to make the platform shareable across a local Wi-Fi/LAN network.
---
## Directory Structure
```text
mri_cla_seg/
├── backend/
│   ├── main.py                  # FastAPI server, PyTorch model, and fallback logic
│   └── requirements.txt         # Backend Python dependencies
├── frontend/
│   ├── src/
│   │   ├── App.jsx              # Main React Workspace, state hooks, and report print card
│   │   ├── App.css              # Glassmorphic UI dashboard and compact print overrides
│   │   └── main.jsx             # React entrypoint
│   ├── index.html               # Page wrapper (MRI.AI)
│   ├── package.json             # Frontend Node.js dependencies
│   └── vite.config.js           # Vite dev server configuration
├── best_resunetpp_multitask.pth # Saved model weights (33MB, PyTorch)
└── start_project.bat            # Double-clickable LAN launcher script
```
---
## Technology Stack
- **Deep Learning Core**: PyTorch (GPU CUDA acceleration fallback to CPU)
- **Backend API**: FastAPI, Uvicorn, OpenCV, NumPy
- **Frontend Dashboard**: Vite, React, Vanilla CSS
---
## Installation & Setup
### Prerequisites
- Python 3.8+
- Node.js (with npm)
- *Optional*: NVIDIA CUDA compatible GPU and drivers (for accelerated inference)
### 1. Backend Setup
Navigate to the `backend/` directory, create a virtual environment, and install dependencies:
```bash
cd backend
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
pip install -r requirements.txt
```
### 2. Frontend Setup
Navigate to the `frontend/` directory and install the Node modules:
```bash
cd ../frontend
npm install
```
---
## Running the Platform
### Option A: Local LAN Launcher (Windows - Recommended)
Double-click the **`start_project.bat`** file in the project root. This script will:
1. Retrieve your local router-assigned IPv4 address.
2. Spin up the FastAPI backend on port `8000`.
3. Spin up the Vite React frontend on port `5173`.
4. Open the site in your default browser.
5. Output local URLs to allow other devices on the same Wi-Fi/LAN connection (e.g. tablets, phones, or other office computers) to access the platform.
### Option B: Manual Command Line Start
**Start Backend Server:**
```bash
cd backend
venv\Scripts\activate
python -m uvicorn main:app --host 0.0.0.0 --port 8000
```
**Start Frontend Dev Server:**
```bash
cd frontend
npm run dev -- --host
```
Open **`http://localhost:5173/`** in your web browser.
---
## Model Pipeline Details
- **Input Constraints**: Images are converted to RGB, resized to $256 \times 256$ pixels, and normalized by dividing by $255.0$ before being fed to the model.
- **Classification Output**: Confidences mapped to `['glioma', 'meningioma', 'pituitary', 'no_tumor']`.
- **Segmentation Output**: Binary mask overlaying coordinates indicating anomalous boundaries.
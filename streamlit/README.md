# 🛰️ Satellite Image Super-Resolution App

A Streamlit web application for enhancing low-resolution satellite images to high-resolution using a SwinIR model trained on the WorldStrat dataset.

## 🌟 Features

- **4x Super-Resolution**: Transform 160x160 images to 640x640 high-resolution outputs
- **Before/After Comparison**: Side-by-side visualization of original and enhanced images
- **Easy Upload**: Drag-and-drop interface for image uploads
- **Download Results**: Save enhanced images directly from the app
- **GPU Acceleration**: Optional CUDA support for faster processing
- **Professional UI**: Clean, modern interface with helpful guides

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.8 or higher
- Your trained `.pth` model file from the WorldStrat training

### 2. Installation

```bash
# Clone SwinIR repository (required)
git clone https://github.com/JingyunLiang/SwinIR.git

# Install dependencies
pip install -r requirements.txt
```

### 3. Prepare Your Model

Place your trained `.pth` model file in the project directory. The default expected filename is `best_model.pth`, but you can specify a different path in the app.

### 4. Run the App

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

## 📖 How to Use

1. **Load the Model**
   - In the sidebar, verify the model path (default: `best_model.pth`)
   - Select your preferred device (CPU or GPU)
   - Click "🔄 Load Model" button

2. **Upload Image**
   - Click "Browse files" or drag-and-drop a satellite image
   - Supported formats: PNG, JPG, JPEG, TIFF

3. **Enhance Image**
   - Once the image is uploaded and model is loaded
   - Click "🚀 Enhance Image" button
   - Wait for processing (a few seconds)

4. **View Results**
   - See the enhanced high-resolution image on the right
   - Compare before/after in the comparison section
   - Download the enhanced image using the download button

## 🔧 Model Architecture

The app uses a SwinIR model with the following configuration:

```python
SwinIR(
    upscale=4,              # 4x upscaling factor
    in_chans=3,             # RGB channels
    img_size=160,           # Input size
    window_size=8,
    img_range=1.0,
    depths=[6, 6, 6, 6, 6, 6],
    embed_dim=180,
    num_heads=[6, 6, 6, 6, 6, 6],
    mlp_ratio=2,
    upsampler='pixelshuffle',
    resi_connection='1conv'
)
```

## 📊 Expected Input/Output

- **Input**: Low-resolution satellite images (any size, will be resized to 160x160)
- **Output**: High-resolution images (640x640)
- **Best Results**: Images similar to WorldStrat dataset (satellite imagery)

## 🎨 UI Features

- **Responsive Design**: Works on different screen sizes
- **Dark/Light Mode**: Follows Streamlit's theme settings
- **Progress Indicators**: Shows loading states during processing
- **Error Handling**: Friendly error messages for common issues
- **Metric Cards**: Display image dimensions and processing info

## 🛠️ Troubleshooting

### Model Not Loading
- Verify the `.pth` file path is correct
- Ensure the model was trained with the same architecture parameters
- Check that you have enough RAM/VRAM

### CUDA Out of Memory
- Switch to CPU mode in the sidebar
- Close other GPU-intensive applications
- Try processing smaller images

### Image Quality Issues
- Ensure input images are satellite imagery
- The model works best on images similar to WorldStrat dataset
- Avoid overly compressed or noisy inputs

## 📝 Technical Details

**Dependencies**:
- Streamlit: Web interface
- PyTorch: Deep learning framework
- SwinIR: Transformer-based super-resolution model
- OpenCV: Image processing
- NumPy: Numerical operations

**Processing Pipeline**:
1. Image upload and validation
2. Preprocessing (resize to 160x160, normalize)
3. Model inference (4x upscaling)
4. Postprocessing (denormalize, convert to image)
5. Display and download

## 🎯 Tips for Best Results

- Use satellite imagery similar to the WorldStrat dataset
- Ensure input images are not too compressed
- For batch processing, run the model separately with custom scripts
- GPU mode significantly speeds up processing

## 👥 Team Alpha - Klymo Hackathon

This application was developed as part of the Klymo Hackathon by Team Alpha, showcasing satellite image super-resolution capabilities using the WorldStrat dataset.

## 📄 License

This project uses the SwinIR model architecture. Please refer to the original [SwinIR repository](https://github.com/JingyunLiang/SwinIR) for licensing information.

## 🤝 Contributing

Feel free to submit issues or enhancement requests. This is a hackathon project, so contributions and improvements are welcome!

---

**Made with ❤️ by Team Alpha**

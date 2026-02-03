import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import io
import sys
import os
from datetime import datetime
if st.button("🧹 Clear Cache & Reload (Dev)"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.success("Cache cleared. Reload the page (Ctrl+R).")
# Add SwinIR to path
if not os.path.exists('SwinIR'):
    st.error("SwinIR repository not found. Please run setup.sh first.")
    st.stop()

sys.path.append('SwinIR')
from models.network_swinir import SwinIR

# Page configuration
st.set_page_config(
    page_title="Satellite Image Super-Resolution | Team Alpha",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.75rem 1.5rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 10px;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .info-box {
        background: linear-gradient(135deg, #e0f7fa 0%, #b2ebf2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #00acc1;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    .success-box {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 5px solid #4caf50;
        margin: 1rem 0;
    }
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding-bottom: 1rem;
        font-size: 2.5rem;
        font-weight: 800;
    }
    .comparison-label {
        font-size: 1.2rem;
        font-weight: 600;
        color: #333;
        text-align: center;
        padding: 0.5rem;
        background-color: #f0f2f6;
        border-radius: 8px;
        margin-bottom: 0.5rem;
    }
    .stat-number {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'processing_time' not in st.session_state:
    st.session_state.processing_time = None
if 'enhanced_image' not in st.session_state:
    st.session_state.enhanced_image = None

@st.cache_resource
def load_model(model_path, device):
    model = SwinIR(
        upscale=4,
        in_chans=3,
        img_size=48,
        window_size=8,
        img_range=1.0,
        depths=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler='nearest+conv',
        resi_connection='1conv'
    )

    checkpoint = torch.load(model_path, map_location=device)

    if 'params_ema' in checkpoint:
        state_dict = checkpoint['params_ema']
    elif 'params' in checkpoint:
        state_dict = checkpoint['params']
    else:
        state_dict = checkpoint

    # 🔥🔥🔥 THIS PART WAS MISSING 🔥🔥🔥
    remove_keys = [k for k in state_dict.keys() if 'attn_mask' in k]
    for k in remove_keys:
        del state_dict[k]

    model.load_state_dict(state_dict, strict=False)

    model.to(device)
    model.eval()
    return model
def preprocess_image(image, target_size=160):
    """Preprocess image for model input"""
    img = np.array(image)
    
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    
    img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
    img = img.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    
    return img_tensor

def postprocess_image(tensor):
    """Convert model output tensor to PIL Image"""
    img = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(img)

def enhance_image(model, image, device):
    """Run super-resolution on input image"""
    import time
    start_time = time.time()
    
    with torch.no_grad():
        img_tensor = preprocess_image(image).to(device)
        output = model(img_tensor)
        enhanced = postprocess_image(output)
    
    processing_time = time.time() - start_time
    return enhanced, processing_time

def calculate_metrics(original, enhanced):
    """Calculate basic image quality metrics"""
    # Resize original to match enhanced for comparison
    orig_resized = original.resize(enhanced.size, Image.BICUBIC)
    
    orig_array = np.array(orig_resized)
    enh_array = np.array(enhanced)
    
    # Calculate sharpness (using Laplacian variance)
    def sharpness(img_array):
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    
    orig_sharp = sharpness(orig_array)
    enh_sharp = sharpness(enh_array)
    
    return {
        'original_sharpness': orig_sharp,
        'enhanced_sharpness': enh_sharp,
        'sharpness_improvement': ((enh_sharp - orig_sharp) / orig_sharp * 100) if orig_sharp > 0 else 0
    }

# App Title
st.title("🛰️ Satellite Image Super-Resolution")
st.markdown("""
<div class="info-box">
    <h3 style="margin:0; color: #00acc1;">🚀 WorldStrat Dataset - 4x Super-Resolution Enhancement</h3>
    <p style="margin: 0.5rem 0 0 0;">Transform low-resolution satellite imagery into stunning high-resolution images using our advanced SwinIR model trained on the WorldStrat dataset.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Model Configuration")
    
    model_path = st.text_input(
        "Model Path",
        value="Best-model.pth",
        help="Path to your trained .pth model checkpoint"
    )
    
    device_option = st.radio(
        "Processing Device",
        ["CPU", "GPU (CUDA)"],
        help="GPU recommended for faster processing"
    )
    device = torch.device("cuda" if device_option == "GPU (CUDA)" and torch.cuda.is_available() else "cpu")
    
    if device_option == "GPU (CUDA)" and not torch.cuda.is_available():
        st.warning("⚠️ CUDA not available. Using CPU instead.")
    
    if st.button("🔄 Load Model", use_container_width=True):
        if os.path.exists(model_path):
            with st.spinner("Loading model... Please wait"):
                try:
                    st.session_state.model = load_model(model_path, device)
                    st.session_state.model_loaded = True
                    st.balloons()
                    st.success("✅ Model loaded successfully!")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        else:
            st.error(f"❌ File not found: {model_path}")
    
    st.markdown("---")
    
    if st.session_state.model_loaded:
        st.markdown("""
        <div class="success-box">
            <strong>✅ Model Status: Ready</strong><br>
            <small>📍 Device: {}</small>
        </div>
        """.format(device), unsafe_allow_html=True)
    else:
        st.warning("⚠️ Load model to start")
    
    st.markdown("---")
    st.markdown("### 📊 Architecture Details")
    with st.expander("View Model Specs"):
        st.markdown("""
        - **Type**: SwinIR Transformer
        - **Upscale**: 4x (160→640)
        - **Depth**: 6 layers × 6 blocks
        - **Embedding**: 180 dimensions
        - **Attention Heads**: 6 per layer
        - **Window Size**: 8×8
        - **Dataset**: WorldStrat
        """)
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 1rem; background-color: #f0f2f6; border-radius: 8px;">
        <strong>Team Alpha</strong><br>
        <small>Klymo Hackathon 2026</small>
    </div>
    """, unsafe_allow_html=True)

# Main content area
tab1, tab2, tab3 = st.tabs(["📸 Enhance Image", "🔍 Comparison View", "📈 Analytics"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📤 Input Image")
        uploaded_file = st.file_uploader(
            "Upload satellite image",
            type=['png', 'jpg', 'jpeg', 'tiff', 'tif'],
            help="Supported: PNG, JPG, JPEG, TIFF"
        )
        
        if uploaded_file is not None:
            input_image = Image.open(uploaded_file)
            st.image(input_image, caption="Low-Resolution Input", use_container_width=True)
            
            col1a, col1b = st.columns(2)
            with col1a:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="stat-number">{input_image.size[0]}×{input_image.size[1]}</div>
                    <div>Input Dimensions</div>
                </div>
                """, unsafe_allow_html=True)
            with col1b:
                file_size = len(uploaded_file.getvalue()) / 1024
                st.markdown(f"""
                <div class="metric-card">
                    <div class="stat-number">{file_size:.1f} KB</div>
                    <div>File Size</div>
                </div>
                """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### ✨ Enhanced Output")
        
        if uploaded_file is not None and st.session_state.model_loaded:
            if st.button("🚀 Enhance Image", type="primary", use_container_width=True):
                with st.spinner("🔮 Enhancing image with trained model..."):
                    try:
                        enhanced_image, proc_time = enhance_image(
                            st.session_state.model,
                            input_image,
                            device
                        )
                        st.session_state.enhanced_image = enhanced_image
                        st.session_state.processing_time = proc_time
                        
                        st.success(f"✅ Enhancement complete in {proc_time:.2f}s")
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
            
            if st.session_state.enhanced_image is not None:
                st.image(
                    st.session_state.enhanced_image,
                    caption="High-Resolution Output (4x)",
                    use_container_width=True
                )
                
                col2a, col2b = st.columns(2)
                with col2a:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="stat-number">{st.session_state.enhanced_image.size[0]}×{st.session_state.enhanced_image.size[1]}</div>
                        <div>Output Dimensions</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col2b:
                    if st.session_state.processing_time:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="stat-number">{st.session_state.processing_time:.2f}s</div>
                            <div>Processing Time</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Download button
                buf = io.BytesIO()
                st.session_state.enhanced_image.save(buf, format='PNG')
                byte_im = buf.getvalue()
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.download_button(
                    label="⬇️ Download Enhanced Image",
                    data=byte_im,
                    file_name=f"enhanced_satellite_{timestamp}.png",
                    mime="image/png",
                    use_container_width=True
                )
        
        elif uploaded_file is not None:
            st.info("👈 Please load the model first")
        else:
            st.info("👈 Upload an image to begin")

with tab2:
    st.markdown("### 🔍 Side-by-Side Comparison")
    
    if uploaded_file is not None and st.session_state.enhanced_image is not None:
        comp_col1, comp_col2 = st.columns(2)
        
        with comp_col1:
            st.markdown('<div class="comparison-label">📥 Original (Low-Res)</div>', unsafe_allow_html=True)
            st.image(input_image, use_container_width=True)
        
        with comp_col2:
            st.markdown('<div class="comparison-label">✨ Enhanced (High-Res 4x)</div>', unsafe_allow_html=True)
            st.image(st.session_state.enhanced_image, use_container_width=True)
        
        # Quality metrics
        st.markdown("### 📊 Quality Metrics")
        metrics = calculate_metrics(input_image, st.session_state.enhanced_image)
        
        met_col1, met_col2, met_col3 = st.columns(3)
        with met_col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="stat-number">{metrics['original_sharpness']:.1f}</div>
                <div>Original Sharpness</div>
            </div>
            """, unsafe_allow_html=True)
        with met_col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="stat-number">{metrics['enhanced_sharpness']:.1f}</div>
                <div>Enhanced Sharpness</div>
            </div>
            """, unsafe_allow_html=True)
        with met_col3:
            improvement = metrics['sharpness_improvement']
            color = "#4caf50" if improvement > 0 else "#f44336"
            st.markdown(f"""
            <div class="metric-card">
                <div class="stat-number" style="color: {color};">+{improvement:.1f}%</div>
                <div>Improvement</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("📸 Enhance an image first to view comparison")

with tab3:
    st.markdown("### 📈 Performance Analytics")
    
    if st.session_state.processing_time and st.session_state.enhanced_image:
        anal_col1, anal_col2, anal_col3, anal_col4 = st.columns(4)
        
        with anal_col1:
            st.metric("Processing Time", f"{st.session_state.processing_time:.3f}s")
        with anal_col2:
            pixels_processed = 160 * 160
            st.metric("Input Pixels", f"{pixels_processed:,}")
        with anal_col3:
            output_pixels = 640 * 640
            st.metric("Output Pixels", f"{output_pixels:,}")
        with anal_col4:
            speedup = pixels_processed / st.session_state.processing_time
            st.metric("Processing Speed", f"{speedup:,.0f} px/s")
        
        st.markdown("---")
        
        # Model info
        st.markdown("### 🧠 Model Information")
        info_col1, info_col2 = st.columns(2)
        
        with info_col1:
            st.markdown("""
            **Architecture Details:**
            - Model: SwinIR
            - Transformer layers: 6
            - Attention heads: 6 per layer
            - Embedding dimension: 180
            - MLP ratio: 2
            """)
        
        with info_col2:
            st.markdown("""
            **Training Details:**
            - Dataset: WorldStrat
            - Input size: 160×160
            - Output size: 640×640
            - Upscale factor: 4x
            - Color channels: RGB (3)
            """)
    else:
        st.info("📊 Process an image to view analytics")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <h4 style="color: #667eea;">🌍 Powered by SwinIR & WorldStrat Dataset</h4>
    <p><strong>Team Alpha</strong> | Klymo Hackathon 2026</p>
    <p><small>For optimal results, use satellite imagery similar to the WorldStrat training dataset</small></p>
</div>
""", unsafe_allow_html=True)
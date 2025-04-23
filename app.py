import streamlit as st
import cv2
import numpy as np
import mahotas
from PIL import Image
import io

# Set page configuration
st.set_page_config(
    page_title="Object Detection using Zernike Moments",
    page_icon="��",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        background-color: #000000;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .result-title {
        color: white;
        text-align: center;
        font-size: 2em;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("About")
    st.markdown("""
    This application uses Zernike moments to detect objects in images.
    
    ### How it works:
    1. Upload a reference image (e.g., a single apple)
    2. Upload a target image (e.g., a group of fruits)
    3. Click 'Detect Object' to find the matching object
    
    ### Technical Details:
    - Uses HSV color space for object isolation
    - Implements Zernike moments for shape matching
    - Provides visual feedback with bounding boxes
    """)

# Main content
st.title("🔍 Object Detection using Zernike Moments")
st.markdown("""
<div class="info-box">
    <h3>Instructions</h3>
    <ol>
        <li>Upload a clear image of the object you want to detect (Reference Image)</li>
        <li>Upload an image containing multiple objects (Target Image)</li>
        <li>Click the 'Detect Object' button to find the matching object</li>
    </ol>
</div>
""", unsafe_allow_html=True)

def color_preprocess_image(bgr_img):
    """Preprocess the image to isolate red regions."""
    hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    
    # Define red color ranges in HSV
    lower_red1 = np.array([0, 100, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 100, 50])
    upper_red2 = np.array([180, 255, 255])
    
    # Create masks for each range and combine them
    mask1 = cv2.inRange(hsv_img, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_img, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    
    # Remove noise and small holes
    kernel = np.ones((5, 5), np.uint8)
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask_clean

def compute_zernike(mask, contour, degree=8):
    """Compute Zernike moments for a given contour."""
    x, y, w, h = cv2.boundingRect(contour)
    crop = mask[y:y+h, x:x+w]
    crop = (crop > 0).astype(np.uint8)
    
    radius = min(w, h) // 2
    if radius <= 0:
        return None, (x, y, w, h)
    
    zernike_vector = mahotas.features.zernike_moments(crop.astype(np.float32), radius, degree=degree)
    return zernike_vector, (x, y, w, h)

def add_found_text(img, bbox):
    """Add 'FOUND!' text above the detected object."""
    x, y, w, h = bbox
    # Create a green background for text
    text = "FOUND!"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    
    # Get text size
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Calculate text position (centered above the object)
    text_x = x + (w - text_width) // 2
    text_y = y - 10  # 10 pixels above the object
    
    # Add green background
    bg_pad = 4
    cv2.rectangle(img, 
                 (text_x - bg_pad, text_y - text_height - bg_pad),
                 (text_x + text_width + bg_pad, text_y + bg_pad),
                 (0, 255, 0),
                 -1)
    
    # Add white text
    cv2.putText(img, text, (text_x, text_y), 
                font, font_scale, (255, 255, 255), thickness)
    return img

def process_images(reference_img, target_img):
    """Process both images and find the matching object."""
    # Process reference image
    ref_mask = color_preprocess_image(reference_img)
    ref_contours, _ = cv2.findContours(ref_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not ref_contours:
        return None, "No object found in reference image"
    
    largest_ref_contour = max(ref_contours, key=cv2.contourArea)
    ref_moments, _ = compute_zernike(ref_mask, largest_ref_contour)
    
    if ref_moments is None:
        return None, "Could not compute Zernike moments for reference object"
    
    # Process target image
    target_mask = color_preprocess_image(target_img)
    target_contours, _ = cv2.findContours(target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    best_distance = float('inf')
    best_bbox = None
    all_bboxes = []
    
    for cnt in target_contours:
        if cv2.contourArea(cnt) < 100:
            continue
        
        zernike_vector, bbox = compute_zernike(target_mask, cnt)
        if zernike_vector is None:
            continue
        
        distance = np.linalg.norm(ref_moments - zernike_vector)
        all_bboxes.append((distance, bbox))
        
        if distance < best_distance:
            best_distance = distance
            best_bbox = bbox
    
    if best_bbox is None:
        return None, "No matching object found in target image"
    
    # Create result image with black background
    result_img = target_img.copy()
    
    # Draw red rectangles for non-target objects
    for distance, bbox in all_bboxes[1:]:  # Skip the best match
        x, y, w, h = bbox
        cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    
    # Draw green rectangle for target object and add FOUND! text
    x, y, w, h = best_bbox
    cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    result_img = add_found_text(result_img, best_bbox)
    
    return result_img, "Target object found!"

# Main content
st.markdown('<h1 class="result-title">RESULT</h1>', unsafe_allow_html=True)

# Create two columns for the images
col1, col2 = st.columns(2)

with col1:
    st.subheader("Input Image")
    ref_file = st.file_uploader("Choose reference image", type=["jpg", "jpeg", "png"], key="ref")
    
    if ref_file is not None:
        ref_img = cv2.imdecode(np.frombuffer(ref_file.read(), np.uint8), cv2.IMREAD_COLOR)
        # Create black background
        bg = np.zeros((ref_img.shape[0] + 40, ref_img.shape[1] + 40, 3), dtype=np.uint8)
        # Place image on black background
        y_offset = 20
        x_offset = 20
        bg[y_offset:y_offset+ref_img.shape[0], x_offset:x_offset+ref_img.shape[1]] = ref_img
        st.image(cv2.cvtColor(bg, cv2.COLOR_BGR2RGB), use_column_width=True)

with col2:
    st.subheader("Detected Objects")
    target_file = st.file_uploader("Choose target image", type=["jpg", "jpeg", "png"], key="target")
    
    if target_file is not None:
        target_img = cv2.imdecode(np.frombuffer(target_file.read(), np.uint8), cv2.IMREAD_COLOR)
        if ref_file is not None:
            result_img, message = process_images(ref_img, target_img)
            if result_img is not None:
                # Create black background
                bg = np.zeros((result_img.shape[0] + 40, result_img.shape[1] + 40, 3), dtype=np.uint8)
                # Place image on black background
                y_offset = 20
                x_offset = 20
                bg[y_offset:y_offset+result_img.shape[0], x_offset:x_offset+result_img.shape[1]] = result_img
                st.image(cv2.cvtColor(bg, cv2.COLOR_BGR2RGB), use_column_width=True)
            else:
                st.error(message)
        else:
            # Display original image with black background
            bg = np.zeros((target_img.shape[0] + 40, target_img.shape[1] + 40, 3), dtype=np.uint8)
            y_offset = 20
            x_offset = 20
            bg[y_offset:y_offset+target_img.shape[0], x_offset:x_offset+target_img.shape[1]] = target_img
            st.image(cv2.cvtColor(bg, cv2.COLOR_BGR2RGB), use_column_width=True)

# Add a footer
st.markdown("---")
st.markdown("""
<div style="text-align: center">
    <p>Created with Streamlit | Using Zernike Moments for Object Detection</p>
</div>
""", unsafe_allow_html=True) 
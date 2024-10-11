import streamlit as st
from PIL import Image
import numpy as np
import cv2
from skimage.util import random_noise
from streamlit_image_comparison import image_comparison

st.title("Lab-03 Image Enhancing")

def make_noise(image, amount=0.05):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    noisy_image = random_noise(image_rgb, mode='gaussian', mean=0, var=amount)
    return (noisy_image * 255).astype(np.uint8)


# Function to sharpen image with adjustable kernel
def sharpeningImage(image_cv, sharpness_kernel):
    sharpened_image = cv2.filter2D(image_cv, -1, sharpness_kernel)
    sharpened_image_rgb = cv2.cvtColor(sharpened_image, cv2.COLOR_BGR2RGB)
    return sharpened_image_rgb


def denoisingImage(image_cv, blur_level, denoise_type="Gaussian"):
    if denoise_type == "Gaussian":
        denoised_image = cv2.GaussianBlur(image_cv, ksize=(blur_level, blur_level), sigmaX=0)
    elif denoise_type == "Median":
        denoised_image = cv2.medianBlur(image_cv, ksize=blur_level)
    elif denoise_type == "Bilateral":
        denoised_image = cv2.bilateralFilter(image_cv, d=9, sigmaColor=75, sigmaSpace=75)

    image_rgb = cv2.cvtColor(denoised_image, cv2.COLOR_BGR2RGB)
    return image_rgb

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

tab1, tab2 = st.tabs(["Enhance image", "Edge Detection"])

# Enhance Image Tab
with tab1:
    st.header("Enhance image")
    left_column, right_column = st.columns([3, 1])
    kernels = {
        "Kernel 1": np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]),
        "Kernel 2": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
    }
    blur_level = 5
    sharpness_kernel = kernels["Kernel 1"]

    with right_column:
        chosen = st.radio('Enhance', ("Denoising", "Sharpening", "Both"))
        st.divider()
        if chosen == "Denoising" or chosen == "Both":
            blur_level = st.slider("Blur Level", min_value=5, max_value=21, step=2, value=5)
            denoise_type = st.radio("Denoise Type", ["Gaussian", "Median", "Bilateral"])
            st.divider()

        if chosen == "Sharpening" or chosen == "Both":
            kernel_choice = st.radio("Choose Sharpening Kernel", list(kernels.keys()))
            st.code(f"{kernels[kernel_choice]}", language="python")
            sharpness_kernel = kernels[kernel_choice]

    with left_column:
        if uploaded_file is not None:
            img = Image.open(uploaded_file)
            img_cv = np.array(img)
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

            if chosen == "Denoising":
                img1 = img
                img2 = denoisingImage(img_cv, blur_level, denoise_type)
                label2 = f"Denoised ({denoise_type})"
            elif chosen == "Sharpening":
                img1 = img
                img2 = sharpeningImage(img_cv, sharpness_kernel)
                label2 = "Sharpened"
            elif chosen == "Both":
                img1 = img
                denoised_image_cv = denoisingImage(img_cv, blur_level, denoise_type)  # Denoise first
                img2 = sharpeningImage(cv2.cvtColor(denoised_image_cv, cv2.COLOR_RGB2BGR), sharpness_kernel)  # Then sharpen
                label2 = f"Denoised & Sharpened ({denoise_type})"

            # Image comparison
            image_comparison(
                label1="Original",
                label2=label2,
                width=500,
                starting_position=50,
                show_labels=True,
                make_responsive=True,
                img1=img1,
                img2=img2,
            )
        else:
            st.error('Please choose an image')

# Sobel Edge Detection function
def sobel_edge_detection(image_cv):
    gray_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)  # Sobel in x direction
    sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)  # Sobel in y direction
    sobel = cv2.sqrt(sobelx**2 + sobely**2)
    return cv2.convertScaleAbs(sobel)

# Prewitt Edge Detection function
def prewitt_edge_detection(image_cv):
    gray_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)

    # Prewitt kernels
    kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32)
    kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)

    # Apply filter2D for Prewitt X and Y directions
    prewittx = cv2.filter2D(gray_image, cv2.CV_32F, kernelx)  # Convert to float for calculations
    prewitty = cv2.filter2D(gray_image, cv2.CV_32F, kernely)

    # Magnitude of gradients
    prewitt = np.sqrt(prewittx**2 + prewitty**2)

    # Convert the result to uint8 to display
    prewitt = cv2.convertScaleAbs(prewitt)

    return prewitt

# Canny Edge Detection function
def canny_edge_detection(image_cv):
    gray_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(gray_image, 100, 200)

# Edge Detection Tab
with tab2:
    st.header("Edge Detection")

    # Columns for layout in Edge Detection tab
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)

    if uploaded_file is not None:
        # Load the image
        img = Image.open(uploaded_file)
        img_cv = np.array(img)
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)  # Convert to OpenCV BGR format

        with col1:
            st.image(img, caption="Original", use_column_width=True)
        with col2:
            sobel_edge_img = sobel_edge_detection(img_cv)
            st.image(sobel_edge_img, caption="Sobel", use_column_width=True)
        with col3:
            canny_edge_img = canny_edge_detection(img_cv)
            st.image(canny_edge_img, caption="Canny", use_column_width=True)
        with col4:
            prewitt_edge_img = prewitt_edge_detection(img_cv)
            st.image(prewitt_edge_img, caption="Prewitt", use_column_width=True)

    else:
        st.error("Please choose an image")

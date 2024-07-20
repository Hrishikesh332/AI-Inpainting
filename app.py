import streamlit as st
from PIL import Image, ImageDraw
import torch
from ultralytics import YOLO
import numpy as np
import io
import os
from PIL import Image
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas
import vertexai
from vertexai.preview.vision_models import Image as VertexImage, ImageGenerationModel
from google.oauth2 import service_account
from utils import resize_image, format_results, point_prompt, show_masks_on_image

if not os.path.exists('temp_images'):
    os.makedirs('temp_images')

page_element = """
<style>
[data-testid="stAppViewContainer"]{
background-image: url("https://img.freepik.com/free-photo/vivid-blurred-colorful-background_58702-2655.jpg");
background-size: cover;
}
[data-testid="stHeader"]{
background-color: rgba(0,0,0,0);
}
[data-testid="stToolbar"]{
right: 2rem;
background-image: url("");
background-size: cover;
}
</style>
"""

st.set_page_config(layout="wide")
st.markdown("""
<style>
.stApp {
    background-color: #f0f2f6;
}
.main {
    padding: 2rem;
}
.stButton>button {
    background-color: #4CAF50;
    color: white;
    border: none;
    padding: 10px 20px;
    text-align: center;
    text-decoration: none;
    display: inline-block;
    font-size: 16px;
    margin: 4px 2px;
    cursor: pointer;
    border-radius: 5px;
}
.point-list {
    background-color: #ffffff;
    padding: 10px;
    border-radius: 5px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.point-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 5px 0;
    border-bottom: 1px solid #e0e0e0;
}
</style>
""", unsafe_allow_html=True)

st.markdown(page_element, unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: black';>🖼️ Interactive Image Segmentation</h1>", unsafe_allow_html=True)
st.markdown("---")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load and display the uploaded image
    raw_image = Image.open(uploaded_file)
    st.image(raw_image, caption="Uploaded Image", use_column_width=True)

    # Resize the image
    resized_image = resize_image(raw_image, input_size=1024)

    # Set up the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    @st.cache_resource
    def load_model():
        return YOLO('./FastSAM.pt')

    model = load_model()

    # Interactive point selection
    st.subheader("🎯 Mark points on the image")
    
    # Initialize session state for points and labels if not already done
    if 'points' not in st.session_state:
        st.session_state.points = []
        st.session_state.labels = []

    # Function to draw larger dots on the image
    def draw_large_dots(image, points, labels):
        draw = ImageDraw.Draw(image)
        for point, label in zip(points, labels):
            color = "green" if label == 1 else "red"
            draw.ellipse([point[0]-7, point[1]-7, point[0]+7, point[1]+7], fill=color, outline=color)
        return image

    # Draw larger dots on the image
    image_with_dots = draw_large_dots(resized_image.copy(), st.session_state.points, st.session_state.labels)

    # Create a canvas component for visualization
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",
        stroke_width=3,
        stroke_color="#000000",
        background_color="#ffffff",
        background_image=image_with_dots,
        update_streamlit=True,
        height=resized_image.height,
        width=resized_image.width,
        drawing_mode="point",
        point_display_radius=5,
        key="canvas",
    )

    # Manual input for point coordinates
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        x = st.number_input("X coordinate", 0, resized_image.width, step=1)
    with col2:
        y = st.number_input("Y coordinate", 0, resized_image.height, step=1)
    with col3:
        point_type = st.radio("Point type", ["✅ Positive", "❌ Negative"])
    with col4:
        if st.button("Add Point"):
            st.session_state.points.append([x, y])
            st.session_state.labels.append(1 if "Positive" in point_type else 0)
            st.experimental_rerun()

    # Display and allow editing of selected points
    st.write("Current Points:")
    st.markdown('<div class="point-list">', unsafe_allow_html=True)
    for i, (point, label) in enumerate(zip(st.session_state.points, st.session_state.labels)):
        st.markdown(f'<div class="point-item">', unsafe_allow_html=True)
        col1, col2, col3 = st.columns([2,2,1])
        with col1:
            st.write(f"Point {i+1}: ({point[0]}, {point[1]})")
        with col2:
            st.write(f"Type: {'✅ Positive' if label == 1 else '❌ Negative'}")
        with col3:
            if st.button(f"🗑️ Remove", key=f"remove_{i}"):
                st.session_state.points.pop(i)
                st.session_state.labels.pop(i)
                st.experimental_rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🎭 Generate Mask"):
        if len(st.session_state.points) > 0:
            results = model(resized_image, device=device, retina_masks=True)
            results = format_results(results[0], 0)
            
            # Generate the masks
            masks, _ = point_prompt(results, st.session_state.points, st.session_state.labels)
            
            # Visualize the generated masks
            masked_image = show_masks_on_image(resized_image, [masks])

            col1, col2 = st.columns(2)
            with col1:
                st.image(masked_image, caption="Masked Image", use_column_width=True)
                
                # Save masked image locally
                masked_image_path = os.path.join('temp_images', 'masked_image.png')
                masked_image.save(masked_image_path)
                
                with open(masked_image_path, "rb") as file:
                    st.download_button(
                        label="📥 Download Masked Image",
                        data=file,
                        file_name="masked_image.png",
                        mime="image/png"
                    )
            
            with col2:
                # Display the mask using plt.imshow
                fig, ax = plt.subplots()
                ax.imshow(masks.squeeze(), cmap='gray')
                ax.axis('off')
                st.pyplot(fig)
                
                # Save grayscale mask locally
                mask_path = os.path.join('temp_images', 'grayscale_mask.png')
                plt.imsave(mask_path, masks.squeeze(), cmap='gray')
                plt.close(fig)
                
                # Provide download option for the grayscale mask
                with open(mask_path, "rb") as file:
                    st.download_button(
                        label="📥 Download Grayscale Mask",
                        data=file,
                        file_name="grayscale_mask.png",
                        mime="image/png"
                    )
            
            # Store paths in session state
            st.session_state.masked_image_path = masked_image_path
            st.session_state.mask_path = mask_path

        else:
            st.warning("⚠️ Please add at least one point before generating the mask.")

    # Image generation
    st.subheader("🖼️ Generate New Image")
    prompt = st.text_input("Enter a prompt for image generation:", "Huge Alien Creature")

    if st.button("Generate New Image"):
        try:
            if 'masked_image_path' in st.session_state and 'mask_path' in st.session_state:
                st.info("Starting image generation process...")

                # Resize images
                input_image = Image.open(st.session_state.masked_image_path)
                mask = Image.open(st.session_state.mask_path)

                # Define the target dimensions
                target_width = 540
                target_height = 960

                # Calculate the scaling factors for both images
                scale_x = target_width / input_image.width
                scale_y = target_height / input_image.height

                # Calculate the new dimensions while maintaining aspect ratio
                new_width = round(input_image.width * min(scale_x, scale_y))
                new_height = round(input_image.height * min(scale_x, scale_y))

                # Resize both images
                resized_input_image = input_image.resize((new_width, new_height))
                resized_mask = mask.resize((new_width, new_height))

                # Save resized images
                resized_input_path = os.path.join('temp_images', 'resized_input.png')
                resized_mask_path = os.path.join('temp_images', 'resized_mask.png')
                resized_input_image.save(resized_input_path)
                resized_mask.save(resized_mask_path)

                st.info("Images resized and saved. Initializing Vertex AI...")

                # Initialize Vertex AI
                credentials = service_account.Credentials.from_service_account_file('<Your Credential .json file from the GCP>')
                vertexai.init(credentials=credentials, project="<Your Project ID from the GCP>", location="us-central1")
                
                model = ImageGenerationModel.from_pretrained("imagegeneration@006")
                base_img = VertexImage.load_from_file(resized_input_path)
                mask_img = VertexImage.load_from_file(resized_mask_path)

                st.info("Generating image...")

                images = model.edit_image(
                    base_image=base_img,
                    mask=mask_img,
                    prompt=prompt,
                    edit_mode="inpainting-insert",
                )

                output_file = os.path.join('temp_images', 'generated_image.png')
                images[0].save(location=output_file, include_generation_parameters=False)

                st.success("Image generated successfully!")

                # Display generated image
                generated_image = Image.open(output_file)
                st.image(generated_image, caption="Generated Image", use_column_width=True)

                # Provide download option for the generated image
                with open(output_file, "rb") as file:
                    st.download_button(
                        label="📥 Download Generated Image",
                        data=file,
                        file_name="generated_image.png",
                        mime="image/png"
                    )
            else:
                st.warning("⚠️ Please generate a mask first before creating a new image.")
        except Exception as e:
            st.error(f"An error occurred during image generation: {str(e)}")
            st.error("Please check your Vertex AI credentials and ensure you have the necessary permissions.")
            st.error("If the problem persists, please contact support with the error message above.")

    if st.button("🧹 Clear All Points"):
        st.session_state.points = []
        st.session_state.labels = []
        st.experimental_rerun()
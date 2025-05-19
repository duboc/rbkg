import streamlit as st
import os
import tempfile
from PIL import Image as PIL_Image 
import io
import numpy as np  # For image blending

# Import the Google GenAI SDK
from google import genai
from google.genai.types import (
    EditImageConfig,
    Image as GenAIImage,
    MaskReferenceConfig,
    MaskReferenceImage,
    RawReferenceImage,
    GenerateContentResponse,
    Part
)
import google.ai.generativelanguage as glm

# Set page config and custom Google-themed styling
st.set_page_config(page_title="Background Remover - Imagen 3 & Gemini", layout="wide")

# --- Google Colors ---
GOOGLE_BLUE = "#4285F4"
GOOGLE_RED = "#EA4335"
GOOGLE_YELLOW = "#FBBC05"
GOOGLE_GREEN = "#34A853"
GOOGLE_GRAY = "#F8F9FA"
GOOGLE_DARK_GRAY = "#5F6368"

# --- Apply custom styling ---
st.markdown("""
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background-color: white;
    }
    h1, h2, h3, p {
        color: #5F6368;
        font-family: 'Google Sans', 'Helvetica Neue', sans-serif; 
    }
    .stButton > button {
        background-color: #4285F4;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        font-family: 'Google Sans', 'Helvetica Neue', sans-serif;
    }
    .stDownloadButton > button {
        background-color: #34A853;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        font-family: 'Google Sans', 'Helvetica Neue', sans-serif;
    }
    .stSelectbox label, .stTextInput label {
        color: #5F6368;
        font-family: 'Google Sans', 'Helvetica Neue', sans-serif;
    }
    .css-8ojfln, .st-cc {
        box-shadow: none !important;
        border: 1px solid #DADCE0 !important;
        border-radius: 8px !important;
    }
    .stAlert {
        border-radius: 8px;
    }
    footer {
        visibility: hidden;
    }
    .ViewerBanner {
        visibility: hidden;
    }
    </style>
""", unsafe_allow_html=True)

# --- Constants for Model Names ---
IMAGEN_MODEL_NAME = "imagen-3.0-capability-001"
GEMINI_MODEL_NAME = "gemini-2.0-flash-preview-image-generation"

# --- Client Initialization ---
def get_genai_client(current_project_id: str, current_location: str):
    if (
        "genai_client" in st.session_state
        and st.session_state.genai_client is not None
        and st.session_state.get("genai_client_project_id") == current_project_id
        and st.session_state.get("genai_client_location") == current_location
    ):
        return st.session_state.genai_client
    
    try:
        client = genai.Client(vertexai=True, project=current_project_id, location=current_location)
        st.session_state.genai_client = client
        st.session_state.genai_client_project_id = current_project_id
        st.session_state.genai_client_location = current_location
        return client
    except Exception as e:
        st.error(f"Failed to initialize GenAI client: {e}")
        st.session_state.genai_client = None
        st.session_state.genai_client_project_id = None
        st.session_state.genai_client_location = None
        return None

# --- Initialize session state for configurations ---
if 'imagen_edit_mode' not in st.session_state:
    st.session_state.imagen_edit_mode = "EDIT_MODE_BGSWAP"
if 'imagen_mask_mode' not in st.session_state:
    st.session_state.imagen_mask_mode = "MASK_MODE_BACKGROUND"
if 'imagen_prompt' not in st.session_state:
    st.session_state.imagen_prompt = ""
if 'imagen_number_of_images' not in st.session_state:
    st.session_state.imagen_number_of_images = 1
if 'gemini_prompt' not in st.session_state:
    st.session_state.gemini_prompt = "Please remove the background from this image, keeping the main subject in full detail. The background should become transparent."
if 'gemini_temperature' not in st.session_state:
    st.session_state.gemini_temperature = 0.4
if 'show_advanced' not in st.session_state:
    st.session_state.show_advanced = False

# --- Sidebar Configuration with Google styling ---
st.sidebar.markdown(f"""
    <div style="text-align: center; margin-bottom: 20px;">
        <h2 style="color: {GOOGLE_BLUE}; font-family: 'Google Sans', 'Helvetica Neue', sans-serif; font-weight: 500;">AI Background Remover</h2>
    </div>
""", unsafe_allow_html=True)

with st.sidebar:
    # Connection settings
    st.subheader("Connection Settings")
    project_id = st.text_input("Google Cloud Project ID", os.environ.get("GOOGLE_CLOUD_PROJECT", "your-project-id"))
    location = st.selectbox("Location", ["us-central1", "europe-west4"], index=0) 
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Model selection
    st.subheader("Model Selection")
    selected_model_option = st.selectbox(
        "Choose AI Model:",
        ("Gemini 2.0 Flash", "Imagen 3"),
        help="Select the AI model to use for background removal"
    )
    
    # Model information with Google color-coded icons
    if selected_model_option == "Imagen 3":
        st.markdown(f"""
            <div style="background-color: #F8F9FA; padding: 10px; border-radius: 8px; margin-top: 10px;">
                <div style="display: flex; align-items: center;">
                    <div style="background-color: {GOOGLE_RED}; width: 12px; height: 12px; border-radius: 50%; margin-right: 8px;"></div>
                    <span style="color: #5F6368; font-size: 0.9em;">Specialized image editing</span>
                </div>
                <div style="display: flex; align-items: center; margin-top: 5px;">
                    <div style="background-color: {GOOGLE_GREEN}; width: 12px; height: 12px; border-radius: 50%; margin-right: 8px;"></div>
                    <span style="color: #5F6368; font-size: 0.9em;">Background swap technology</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
    elif selected_model_option == "Gemini 2.0 Flash":
        st.markdown(f"""
            <div style="background-color: #F8F9FA; padding: 10px; border-radius: 8px; margin-top: 10px;">
                <div style="display: flex; align-items: center;">
                    <div style="background-color: {GOOGLE_BLUE}; width: 12px; height: 12px; border-radius: 50%; margin-right: 8px;"></div>
                    <span style="color: #5F6368; font-size: 0.9em;">Multimodal capabilities</span>
                </div>
                <div style="display: flex; align-items: center; margin-top: 5px;">
                    <div style="background-color: {GOOGLE_YELLOW}; width: 12px; height: 12px; border-radius: 50%; margin-right: 8px;"></div>
                    <span style="color: #5F6368; font-size: 0.9em;">Text & image understanding</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<hr>", unsafe_allow_html=True)
    
    # Advanced Configuration Section
    st.subheader("Model Configuration")
    
    st.session_state.show_advanced = st.checkbox("Show Advanced Options", st.session_state.show_advanced)
    
    # Model-specific configuration
    if selected_model_option == "Imagen 3":
        # Simple configuration options always shown
        st.session_state.imagen_edit_mode = st.selectbox(
            "Edit Mode:",
            ["EDIT_MODE_BGSWAP", "EDIT_MODE_INPAINT_REMOVAL"],
            index=0 if st.session_state.imagen_edit_mode == "EDIT_MODE_BGSWAP" else 1,
            help="EDIT_MODE_BGSWAP: Replaces background with a new one (empty for transparency). EDIT_MODE_INPAINT_REMOVAL: Removes parts of the image."
        )
        
        st.session_state.imagen_mask_mode = st.selectbox(
            "Mask Mode:",
            ["MASK_MODE_BACKGROUND", "MASK_MODE_FOREGROUND"],
            index=0 if st.session_state.imagen_mask_mode == "MASK_MODE_BACKGROUND" else 1,
            help="Determines whether to mask the background or foreground of the image."
        )
        
        # Advanced options
        if st.session_state.show_advanced:
            st.session_state.imagen_prompt = st.text_area(
                "Background Prompt:",
                st.session_state.imagen_prompt,
                help="Empty for transparent background. Enter a description for custom background."
            )
            
            st.session_state.imagen_number_of_images = st.slider(
                "Number of Images:",
                min_value=1,
                max_value=4,
                value=st.session_state.imagen_number_of_images,
                help="Number of variations to generate. Only the first will be displayed."
            )
            
            # Display current settings in a code block for transparent view of what's happening
            with st.expander("View API Configuration"):
                st.code(f"""
# Imagen 3 API Configuration
edit_config = EditImageConfig(
    edit_mode="{st.session_state.imagen_edit_mode}",
    number_of_images={st.session_state.imagen_number_of_images},
)

mask_ref_config = MaskReferenceConfig(
    mask_mode="{st.session_state.imagen_mask_mode}"
)

prompt="{st.session_state.imagen_prompt}"
                """)
    
    elif selected_model_option == "Gemini 2.0 Flash":
        # Standard configuration for Gemini
        st.session_state.gemini_prompt = st.text_area(
            "Instruction Prompt:",
            st.session_state.gemini_prompt,
            help="Instructions to Gemini on how to modify the image."
        )
        
        # Advanced options
        if st.session_state.show_advanced:
            st.session_state.gemini_temperature = st.slider(
                "Temperature:",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.gemini_temperature,
                step=0.1,
                help="Controls creativity. Lower values are more deterministic, higher values more creative."
            )
            
            # Display current settings in a code block
            with st.expander("View API Configuration"):
                st.code(f"""
# Gemini 2.0 Flash API Configuration
gemini_config = genai.types.GenerateContentConfig(
    temperature={st.session_state.gemini_temperature},
    response_modalities=['TEXT', 'IMAGE']
)

prompt="{st.session_state.gemini_prompt}"
                """)
    
    # Preset templates section
    if st.session_state.show_advanced:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.subheader("Prompt Templates")
        
        if selected_model_option == "Imagen 3":
            template_option = st.selectbox(
                "Select a Template:",
                ["Transparent Background", "Solid White Background", "Natural Shadow", "Studio Light", "Text Banner (Custom)"]
            )
            
            if template_option == "Transparent Background":
                if st.button("Apply Template"):
                    st.session_state.imagen_prompt = ""
                    st.session_state.imagen_edit_mode = "EDIT_MODE_BGSWAP"
                    st.session_state.imagen_mask_mode = "MASK_MODE_BACKGROUND"
                    st.rerun()
            elif template_option == "Solid White Background":
                if st.button("Apply Template"):
                    st.session_state.imagen_prompt = "solid white"
                    st.session_state.imagen_edit_mode = "EDIT_MODE_BGSWAP"
                    st.session_state.imagen_mask_mode = "MASK_MODE_BACKGROUND"
                    st.rerun()
            elif template_option == "Natural Shadow":
                if st.button("Apply Template"):
                    st.session_state.imagen_prompt = "soft natural shadow beneath the object, otherwise transparent"
                    st.session_state.imagen_edit_mode = "EDIT_MODE_BGSWAP"
                    st.session_state.imagen_mask_mode = "MASK_MODE_BACKGROUND"
                    st.rerun()
            elif template_option == "Studio Light":
                if st.button("Apply Template"):
                    st.session_state.imagen_prompt = "professional studio lighting with soft gradient background"
                    st.session_state.imagen_edit_mode = "EDIT_MODE_BGSWAP"
                    st.session_state.imagen_mask_mode = "MASK_MODE_BACKGROUND" 
                    st.rerun()
            elif template_option == "Text Banner (Custom)":
                # Initialize text banner variables in session state if they don't exist
                if 'banner_text' not in st.session_state:
                    st.session_state.banner_text = "PRODUCT NAME"
                if 'banner_color' not in st.session_state:
                    st.session_state.banner_color = "blue"
                
                # Banner text input
                st.session_state.banner_text = st.text_input("Banner Text:", st.session_state.banner_text, 
                                                     help="Text to display on the banner below the product")
                
                # Banner color selection with visual indicators
                color_options = ["blue", "red", "green", "yellow", "purple", "teal", "orange", "white", "black"]
                color_cols = st.columns(3)
                
                # Display color options in a grid with visual indicators
                selected_color_index = 0
                for i, color in enumerate(color_options):
                    if color == st.session_state.banner_color:
                        selected_color_index = i
                
                st.session_state.banner_color = st.selectbox(
                    "Banner Color:",
                    options=color_options,
                    index=selected_color_index,
                    help="Color of the banner background"
                )
                
                # Preview of what the prompt will look like
                st.caption(f"Preview: '{st.session_state.banner_text}' on {st.session_state.banner_color} background")
                
                if st.button("Apply Banner Template"):
                    st.session_state.imagen_prompt = f"transparent background with a {st.session_state.banner_color} rectangular banner at the bottom containing the text '{st.session_state.banner_text}' in contrasting color"
                    st.session_state.imagen_edit_mode = "EDIT_MODE_BGSWAP"
                    st.session_state.imagen_mask_mode = "MASK_MODE_BACKGROUND"
                    st.rerun()
                    
        elif selected_model_option == "Gemini 2.0 Flash":
            template_option = st.selectbox(
                "Select a Template:",
                ["Standard Background Removal", "High Detail Preservation", "Product Photography", "Add Natural Shadow", "Text Banner (Custom)"]
            )
            
            if template_option == "Standard Background Removal":
                if st.button("Apply Template"):
                    st.session_state.gemini_prompt = "Please remove the background from this image, keeping the main subject in full detail. The background should become transparent."
                    st.session_state.gemini_temperature = 0.4
                    st.rerun()
            elif template_option == "High Detail Preservation":
                if st.button("Apply Template"):
                    st.session_state.gemini_prompt = "Remove the background completely with extreme attention to fine details like hair, fur, and transparent elements. Preserve all fine details of the subject. Make the background pure alpha transparency."
                    st.session_state.gemini_temperature = 0.3
                    st.rerun()
            elif template_option == "Product Photography":
                if st.button("Apply Template"):
                    st.session_state.gemini_prompt = "This is a product image. Create a professional product photo by removing the background completely. Keep perfect detail on product edges. The background should be pure white or transparent."
                    st.session_state.gemini_temperature = 0.2
                    st.rerun()
            elif template_option == "Add Natural Shadow":
                if st.button("Apply Template"):
                    st.session_state.gemini_prompt = "Remove the background from this image, but maintain a subtle, natural drop shadow beneath the subject. The background should be transparent except for the soft shadow."
                    st.session_state.gemini_temperature = 0.5
                    st.rerun()
            elif template_option == "Text Banner (Custom)":
                # Initialize text banner variables in session state if they don't exist
                if 'banner_text' not in st.session_state:
                    st.session_state.banner_text = "PRODUCT NAME"
                if 'banner_color' not in st.session_state:
                    st.session_state.banner_color = "blue"
                
                # Banner text input
                st.session_state.banner_text = st.text_input("Banner Text:", st.session_state.banner_text, 
                                                     help="Text to display on the banner below the product")
                
                # Banner color selection with visual indicators
                color_options = ["blue", "red", "green", "yellow", "purple", "teal", "orange", "white", "black"]
                color_cols = st.columns(3)
                
                # Display color options in a grid with visual indicators
                selected_color_index = 0
                for i, color in enumerate(color_options):
                    if color == st.session_state.banner_color:
                        selected_color_index = i
                
                st.session_state.banner_color = st.selectbox(
                    "Banner Color:",
                    options=color_options,
                    index=selected_color_index,
                    help="Color of the banner background"
                )
                
                # Preview of what the prompt will look like
                st.caption(f"Preview: '{st.session_state.banner_text}' on {st.session_state.banner_color} banner")
                
                if st.button("Apply Banner Template"):
                    st.session_state.gemini_prompt = f"Remove the background from this image, making it transparent. Then add a {st.session_state.banner_color} rectangular banner at the bottom of the image containing the text '{st.session_state.banner_text}' in a contrasting color that is easy to read. Make sure the banner width matches the width of the product and looks professional."
                    st.session_state.gemini_temperature = 0.4
                    st.rerun()

# --- Main UI ---
st.markdown(f"""
    <h1 style="text-align: center; color: {GOOGLE_BLUE}; font-family: 'Google Sans', sans-serif; font-weight: 500; margin-bottom: 1.5rem;">
        Background Remover
        <span style="font-size: 0.7em; color: {GOOGLE_DARK_GRAY}; display: block; margin-top: 5px;">
            Powered by {selected_model_option}
        </span>
    </h1>
""", unsafe_allow_html=True)

st.markdown(f"""
    <p style="text-align: center; color: {GOOGLE_DARK_GRAY}; margin-bottom: 2rem;">
        Upload your image, and our AI will remove the background while preserving details.
    </p>
""", unsafe_allow_html=True)

# Initialize session state for processed image and UI settings
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None
    st.session_state.original_image = None
    st.session_state.temp_output_path = None
    st.session_state.show_transparency_grid = True
    st.session_state.image_metadata = {}

# Helper function to create a checkboard pattern for transparency visualization
def create_checkboard_background(width, height, square_size=10):
    """Create a checkboard pattern for visualizing transparency"""
    checkboard = PIL_Image.new("RGBA", (width, height), (255, 255, 255, 0))
    draw = PIL_Image.new("RGBA", (width, height), (255, 255, 255, 0))
    
    for y in range(0, height, square_size):
        for x in range(0, width, square_size):
            if (x // square_size + y // square_size) % 2 == 0:
                draw.paste((204, 204, 204, 255), (x, y, x + square_size, y + square_size))
            else:
                draw.paste((255, 255, 255, 255), (x, y, x + square_size, y + square_size))
    
    return draw

# Helper function to compose a transparent image over a checkboard background
def display_with_transparency_grid(image, use_grid=True):
    """Display a transparent image over a checkboard grid if needed"""
    if not use_grid or image.mode != "RGBA":
        return image
    
    # Create checkboard of the same size
    checkboard = create_checkboard_background(image.width, image.height)
    # Paste the image with alpha over the checkboard
    result = checkboard.copy()
    result.alpha_composite(image)
    return result

def reset_session():
    """Reset the processing results but keep configuration"""
    st.session_state.processed_image = None
    st.session_state.original_image = None
    st.session_state.temp_output_path = None
    st.session_state.image_metadata = {}
    if st.session_state.temp_output_path and os.path.exists(st.session_state.temp_output_path):
        try:
            os.unlink(st.session_state.temp_output_path)
        except:
            pass  # Ignore error if file cannot be deleted
    st.rerun()

# Main UI for file upload area
upload_col1, upload_col2 = st.columns([3, 1])

with upload_col1:
    # File uploader with Google styling
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

with upload_col2:
    st.markdown("<br>", unsafe_allow_html=True)  # Add spacing to align with the file uploader
    # Toggle for transparency grid
    st.session_state.show_transparency_grid = st.checkbox(
        "Show Transparency Grid", 
        st.session_state.show_transparency_grid,
        help="Display a checkboard pattern underneath transparent areas"
    )

if uploaded_file is not None:
    # Store the original image in session state
    pil_original_image = PIL_Image.open(uploaded_file)
    st.session_state.original_image = pil_original_image
    
    # Create columns for the display
    display_container = st.container()
    
    with display_container:
        if st.session_state.processed_image is None:
            # Show only the original image initially
            st.markdown(f"<p style='color: {GOOGLE_DARK_GRAY}; font-weight: 500;'>Original Image</p>", unsafe_allow_html=True)
            st.image(pil_original_image, use_container_width=True)
    
    # Google-styled button
    if st.button("Remove Background"):
        if not project_id or project_id == "your-project-id":
            st.error("Please enter your Google Cloud Project ID in the sidebar.")
        else:
            client_to_use = get_genai_client(project_id, location)
            if client_to_use is not None:
                try:
                    image_bytes = uploaded_file.getvalue()
                    base_genai_image = GenAIImage(image_bytes=image_bytes)
                    processed_pil_image = None

                    with st.spinner(f"Processing image with {selected_model_option}..."):
                        # Display configuration summary to show what's being used
                        if selected_model_option == "Imagen 3":
                            st.info(f"""
                            **Using Imagen 3 with:**
                            - Edit Mode: {st.session_state.imagen_edit_mode}
                            - Mask Mode: {st.session_state.imagen_mask_mode}
                            - Prompt: {"(empty)" if not st.session_state.imagen_prompt else f'"{st.session_state.imagen_prompt}"'}
                            - Number of Generated Images: {st.session_state.imagen_number_of_images}
                            """)
                            
                            # Configure with session state values
                            raw_ref_image = RawReferenceImage(reference_image=base_genai_image, reference_id=0)
                            mask_ref_image = MaskReferenceImage(
                                reference_id=1, 
                                reference_image=None, 
                                config=MaskReferenceConfig(mask_mode=st.session_state.imagen_mask_mode)
                            )
                            edit_config = EditImageConfig(
                                edit_mode=st.session_state.imagen_edit_mode, 
                                number_of_images=st.session_state.imagen_number_of_images,
                            )
                            
                            edited_response = client_to_use.models.edit_image(
                                model=IMAGEN_MODEL_NAME,
                                prompt=st.session_state.imagen_prompt,
                                reference_images=[raw_ref_image, mask_ref_image],
                                config=edit_config
                            )
                            if edited_response.generated_images:
                                processed_genai_image = edited_response.generated_images[0].image
                                processed_pil_image = processed_genai_image._pil_image
                            else:
                                st.error("Imagen 3 did not return an image.")

                        elif selected_model_option == "Gemini 2.0 Flash":
                            st.info(f"""
                            **Using Gemini 2.0 Flash with:**
                            - Prompt: "{st.session_state.gemini_prompt[:50]}..." (truncated for display)
                            - Temperature: {st.session_state.gemini_temperature}
                            """)
                            
                            pil_image_for_gemini = PIL_Image.open(io.BytesIO(image_bytes))
                            
                            gemini_config = genai.types.GenerateContentConfig(
                                temperature=st.session_state.gemini_temperature,
                                response_modalities=['TEXT', 'IMAGE']
                            )

                            response = client_to_use.models.generate_content(
                                model=GEMINI_MODEL_NAME,
                                contents=[st.session_state.gemini_prompt, pil_image_for_gemini],
                                config=gemini_config
                            )
                            
                            if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                                found_image = False
                                for part in response.candidates[0].content.parts:
                                    if part.inline_data and part.inline_data.data:
                                        processed_pil_image = PIL_Image.open(io.BytesIO(part.inline_data.data))
                                        found_image = True
                                        break
                                if not found_image:
                                    st.error("Gemini 2.0 Flash did not return an image. Response text: " + response.text if hasattr(response, 'text') else "No text part.")
                            else:
                                st.error("Gemini 2.0 Flash returned an empty or invalid response.")
                                if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                                    st.warning(f"Gemini Prompt Feedback: {response.prompt_feedback}")

                    # --- Common display and download logic ---
                    if processed_pil_image:
                        # Save the processed image to a temporary file for download
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_output_file:
                            processed_pil_image.save(tmp_output_file, format="PNG")
                            st.session_state.temp_output_path = tmp_output_file.name
                        
                        # Store the processed image in session state
                        st.session_state.processed_image = processed_pil_image
                        
                        # Force Streamlit to rerun to show the comparison UI
                        st.rerun()

                except Exception as e:
                    st.error(f"An error occurred during processing: {str(e)}")
                    st.error("Please check your GCP settings, model availability, and permissions.")

# Show the comparison UI if we have both images
if st.session_state.original_image is not None and st.session_state.processed_image is not None:
    # Clear any previous containers
    display_container = st.container()
    
    with display_container:
        # Add a reset button and comparison header
        header_col1, header_col2, header_col3 = st.columns([1, 4, 1])
        
        with header_col1:
            if st.button("↺ Reset", help="Clear results and start over"):
                reset_session()
                
        with header_col2:
            st.markdown(f"""
                <div style="text-align: center; margin-bottom: 1rem;">
                    <p style="color: {GOOGLE_DARK_GRAY}; font-weight: 500; margin-bottom: 0.5rem;">
                        Before & After Comparison
                    </p>
                </div>
            """, unsafe_allow_html=True)
        
        # Use columns to display the images side by side
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"<p style='color: {GOOGLE_DARK_GRAY}; font-weight: 500; text-align: center;'>Original</p>", unsafe_allow_html=True)
            st.image(st.session_state.original_image, use_container_width=True)
            
        with col2:
            st.markdown(f"<p style='color: {GOOGLE_DARK_GRAY}; font-weight: 500; text-align: center;'>Processed</p>", unsafe_allow_html=True)
            
            # Apply transparency grid if enabled and image has alpha channel
            display_image = st.session_state.processed_image
            if st.session_state.show_transparency_grid and display_image.mode == 'RGBA':
                display_image = display_with_transparency_grid(display_image, st.session_state.show_transparency_grid)
            
            st.image(display_image, use_container_width=True)
        
        # Add download section with better file name
        st.markdown("<hr style='margin: 1rem 0;'>", unsafe_allow_html=True)
        dl_col1, dl_col2, dl_col3 = st.columns([1, 2, 1])
        
        with dl_col2:
            if st.session_state.temp_output_path:
                # Create a more descriptive filename
                model_short = "imagen" if selected_model_option == "Imagen 3" else "gemini"
                
                with open(st.session_state.temp_output_path, "rb") as file_bytes_dl:
                    st.download_button(
                        label="Download Processed Image",
                        data=file_bytes_dl,
                        file_name=f"bg_removed_{model_short}.png",
                        mime="image/png",
                        use_container_width=True
                    )

# --- Footer Information with Google styling ---
st.markdown("<hr style='margin-top: 2rem;'>", unsafe_allow_html=True)
st.markdown(f"""
<div style="background-color: {GOOGLE_GRAY}; padding: 1rem; border-radius: 8px;">
    <h3 style="color: {GOOGLE_DARK_GRAY}; font-size: 1.1rem; font-weight: 500;">Requirements</h3>
    <ul style="color: {GOOGLE_DARK_GRAY};">
        <li>A Google Cloud account with Vertex AI API enabled</li>
        <li>API access to selected model: {IMAGEN_MODEL_NAME if selected_model_option == "Imagen 3" else GEMINI_MODEL_NAME}</li>
        <li>Appropriate IAM permissions for Vertex AI</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Add Google-themed credits
st.markdown(f"""
<div style="text-align: center; margin-top: 2rem; color: {GOOGLE_DARK_GRAY}; font-size: 0.8rem;">
    Built with 🔥 Google GenAI SDK • Imagen 3 • Gemini 2.0 Flash
</div>
""", unsafe_allow_html=True)

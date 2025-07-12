# Story-to-Comic Generator with Clean Progress Notifications and Speech Bubbles
import os
import requests
import time
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import streamlit as st
from dotenv import load_dotenv
import base64
import re
import textwrap

load_dotenv()

# Step 1: Setup 
COMIC_FOLDER = "comic_panels"
os.makedirs(COMIC_FOLDER, exist_ok=True)

# Hugging Face API endpoints and token
IMAGE_API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-dev"
SUMMARIZE_API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
HF_TOKEN = os.getenv("HF_API_TOKEN")
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

# Step 2: Dialogue extraction functions 
def extract_dialogue_from_scene(scene_text):
    """Extract dialogue from scene text"""
    dialogues = []
    
    # Pattern to match quoted text
    dialogue_pattern = r'"([^"]*)"'
    matches = re.findall(dialogue_pattern, scene_text)
    
    for match in matches:
        if match.strip():
            dialogues.append(match.strip())
    
    # Also check for single quotes
    single_quote_pattern = r"'([^']*)'"
    single_matches = re.findall(single_quote_pattern, scene_text)
    
    for match in single_matches:
        if match.strip() and match not in dialogues:
            dialogues.append(match.strip())
    
    return dialogues

def get_default_font():
    """Get default font for text bubbles"""
    try:
        # Try to use a system font
        return ImageFont.truetype("arial.ttf", 16)
    except:
        try:
            return ImageFont.truetype("Arial.ttf", 16)
        except:
            try:
                return ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 16)
            except:
                # Fallback to default font
                return ImageFont.load_default()

def draw_speech_bubble(image, text, position, bubble_type="speech"):
    """Draw a speech bubble on the image"""
    if not text.strip():
        return image
    
    draw = ImageDraw.Draw(image)
    font = get_default_font()
    
    # Wrap text to fit in bubble
    wrapped_text = textwrap.fill(text, width=20)
    lines = wrapped_text.split('\n')
    
    # Calculate text dimensions
    line_height = 20
    max_line_width = 0
    
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_width = bbox[2] - bbox[0]
        if line_width > max_line_width:
            max_line_width = line_width
    
    # Bubble dimensions
    bubble_width = max_line_width + 20
    bubble_height = len(lines) * line_height + 20
    
    # Position bubble (avoid edges)
    x, y = position
    if x + bubble_width > image.width - 10:
        x = image.width - bubble_width - 10
    if y + bubble_height > image.height - 10:
        y = image.height - bubble_height - 10
    if x < 10:
        x = 10
    if y < 10:
        y = 10
    
    # Draw bubble background
    bubble_color = (255, 255, 255, 220)  # White with slight transparency
    border_color = (0, 0, 0)
    
    # Create bubble shape
    bubble_rect = [x, y, x + bubble_width, y + bubble_height]
    draw.rounded_rectangle(bubble_rect, radius=10, fill=bubble_color, outline=border_color, width=2)
    
    # Draw speech bubble tail (small triangle)
    if bubble_type == "speech":
        tail_points = [
            (x + bubble_width // 2 - 5, y + bubble_height),
            (x + bubble_width // 2 + 5, y + bubble_height),
            (x + bubble_width // 2, y + bubble_height + 15)
        ]
        draw.polygon(tail_points, fill=bubble_color, outline=border_color)
    
    # Draw text
    text_y = y + 10
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        text_width = bbox[2] - bbox[0]
        text_x = x + (bubble_width - text_width) // 2
        draw.text((text_x, text_y), line, fill=(0, 0, 0), font=font)
        text_y += line_height
    
    return image

def add_speech_bubbles_to_image(image, dialogues):
    """Add speech bubbles to the generated image"""
    if not dialogues:
        return image
    
    # Create positions for speech bubbles
    positions = []
    bubble_spacing = 60
    
    for i, dialogue in enumerate(dialogues):
        if i == 0:
            # First bubble in upper area
            pos = (50, 30)
        elif i == 1:
            # Second bubble in middle-right
            pos = (image.width - 200, image.height // 2 - 50)
        elif i == 2:
            # Third bubble in lower-left
            pos = (50, image.height - 100)
        else:
            # Additional bubbles distributed
            pos = (50 + (i % 3) * 150, 30 + (i // 3) * bubble_spacing)
        
        positions.append(pos)
    
    # Add speech bubbles
    for dialogue, position in zip(dialogues, positions):
        image = draw_speech_bubble(image, dialogue, position)
    
    return image

# Step 3: Break story into scenes 
def break_story_into_scenes(story, max_scenes):
    """Break story into scenes based on user's panel choice (2-10 panels)"""
    sentences = [s.strip() for s in story.split('.') if s.strip()]
    
    if len(sentences) <= max_scenes:
        return sentences
    
    # Group sentences into scenes
    scenes = []
    sentences_per_scene = max(1, len(sentences) // max_scenes)
    
    for i in range(0, len(sentences), sentences_per_scene):
        scene = '. '.join(sentences[i:i + sentences_per_scene])
        if scene:
            scenes.append(scene)
        if len(scenes) >= max_scenes:
            break
    
    return scenes

# Step 4: Summarize scenes 
def summarize_scene(scene_text, progress_container, panel_num, total_panels):
    """Summarize a scene with fallback to simple truncation"""
    if progress_container:
        progress_container.info(f"📝 Processing panel {panel_num}/{total_panels}: Summarizing scene...")
    
    # Remove dialogue from scene for image generation (we'll add it back as bubbles)
    scene_for_image = re.sub(r'"[^"]*"', '', scene_text)
    scene_for_image = re.sub(r"'[^']*'", '', scene_for_image)
    scene_for_image = re.sub(r'\s+', ' ', scene_for_image).strip()
    
    payload = {
        "inputs": scene_for_image,
        "parameters": {
            "max_length": 50,
            "min_length": 10,
            "do_sample": False
        }
    }
    
    try:
        response = requests.post(SUMMARIZE_API_URL, headers=HEADERS, json=payload, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0]['summary_text']
    except:
        pass
    
    # Fallback: simple truncation
    return scene_for_image[:60] + "..." if len(scene_for_image) > 60 else scene_for_image

# Step 5: Try multiple image generation APIs
def generate_image_from_prompt(prompt, progress_container, panel_num, total_panels):
    """Generate image using FLUX.1-dev model"""
    # FLUX-optimized prompt for comic style
    flux_prompt = f"comic book panel: {prompt}, cartoon style, colorful illustration, detailed drawing, clear background for speech bubbles"
    
    payload = {
        "inputs": flux_prompt,
        "parameters": {
            "num_inference_steps": 30,
            "guidance_scale": 3.5,
            "width": 512,
            "height": 512
        }
    }
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            if progress_container:
                progress_container.info(f"🎨 Generating panel {panel_num}/{total_panels} with FLUX.1-dev... (attempt {attempt + 1}/{max_retries})")
            
            response = requests.post(IMAGE_API_URL, headers=HEADERS, json=payload, timeout=60)
            
            if response.status_code == 200:
                if progress_container:
                    progress_container.success(f"✅ Panel {panel_num}/{total_panels} image generated!")
                return Image.open(BytesIO(response.content))
            elif response.status_code == 503:
                if progress_container:
                    progress_container.warning(f"⏳ Panel {panel_num}/{total_panels}: FLUX.1-dev is loading, please wait...")
                time.sleep(20)
                continue
            else:
                if progress_container:
                    progress_container.warning(f"❌ Panel {panel_num}/{total_panels}: Attempt {attempt + 1} failed: {response.status_code}")
                time.sleep(5)
        except Exception as e:
            if progress_container:
                progress_container.warning(f"❌ Panel {panel_num}/{total_panels}: Attempt {attempt + 1} error: {str(e)}")
            time.sleep(5)
    
    # If all attempts fail, raise an exception
    raise Exception(f"FLUX.1-dev failed to generate image for panel {panel_num} after {max_retries} attempts")

# Step 6: Generate comic panels 
def generate_comic_panels(story, max_panels, progress_container=None):
    """Generate comic panels from story"""
    scenes = break_story_into_scenes(story, max_panels)
    
    images = []
    for idx, scene in enumerate(scenes):
        panel_num = idx + 1
        total_panels = len(scenes)
        
        if progress_container:
            progress_container.info(f"🔄 Processing panel {panel_num}/{total_panels}: Analyzing scene...")
        
        # Extract dialogues from scene
        dialogues = extract_dialogue_from_scene(scene)
        
        # Summarize scene (without dialogue for image generation)
        summary = summarize_scene(scene, progress_container, panel_num, total_panels)
        
        # Generate base image
        image = generate_image_from_prompt(summary, progress_container, panel_num, total_panels)
        
        # Add speech bubbles if dialogues exist
        if dialogues:
            if progress_container:
                progress_container.info(f"💬 Adding speech bubbles to panel {panel_num}/{total_panels}...")
            image = add_speech_bubbles_to_image(image, dialogues)
        
        # Save image
        image_path = os.path.join(COMIC_FOLDER, f"panel_{idx+1}.png")
        image.save(image_path)
        images.append(image_path)
        
        if progress_container:
            progress_container.success(f"✅ Panel {panel_num}/{total_panels} completed!")
            time.sleep(0.5)  # Brief pause to show completion
    
    return images

# Step 7: Create comic strip 
def create_comic_strip(image_paths, output_path=None, progress_container=None):
    """Stitch panels into a comic strip with smart layout for multiple panels"""
    if progress_container:
        progress_container.info("🔗 Creating final comic strip...")
    
    if output_path is None:
        output_path = os.path.join(COMIC_FOLDER, "final_comic.png")
    
    images = [Image.open(p).resize((512, 512)) for p in image_paths]
    num_panels = len(images)
    
    # Smart layout based on number of panels
    if num_panels <= 4:
        # Horizontal layout for 2-4 panels
        total_width = 512 * num_panels
        total_height = 512
        comic_strip = Image.new("RGB", (total_width, total_height), color="white")
        
        for i, img in enumerate(images):
            comic_strip.paste(img, (i * 512, 0))
    
    elif num_panels <= 6:
        # 2 rows layout for 5-6 panels
        panels_per_row = 3
        rows = (num_panels + panels_per_row - 1) // panels_per_row
        total_width = 512 * min(panels_per_row, num_panels)
        total_height = 512 * rows
        comic_strip = Image.new("RGB", (total_width, total_height), color="white")
        
        for i, img in enumerate(images):
            row = i // panels_per_row
            col = i % panels_per_row
            comic_strip.paste(img, (col * 512, row * 512))
    
    elif num_panels <= 9:
        # 3x3 grid layout for 7-9 panels
        panels_per_row = 3
        rows = (num_panels + panels_per_row - 1) // panels_per_row
        total_width = 512 * panels_per_row
        total_height = 512 * rows
        comic_strip = Image.new("RGB", (total_width, total_height), color="white")
        
        for i, img in enumerate(images):
            row = i // panels_per_row
            col = i % panels_per_row
            comic_strip.paste(img, (col * 512, row * 512))
    
    else:
        # 5x2 grid layout for 10 panels
        panels_per_row = 5
        rows = 2
        total_width = 512 * panels_per_row
        total_height = 512 * rows
        comic_strip = Image.new("RGB", (total_width, total_height), color="white")
        
        for i, img in enumerate(images):
            row = i // panels_per_row
            col = i % panels_per_row
            comic_strip.paste(img, (col * 512, row * 512))
    
    comic_strip.save(output_path)
    
    if progress_container:
        progress_container.success("🎉 Comic strip completed!")
    
    return output_path

# Alternative: Use local fallback 
def use_local_fallback():
    """Provide instructions if FLUX.1-dev fails"""
    st.error("🚫 FLUX.1-dev is currently unavailable")
    st.info("🔄 This might be temporary. Try again in a few minutes.")

# Streamlit Frontend
st.set_page_config(page_title="Story to Comic Generator", layout="centered")
st.title("📚🖼️ PlotToon: AI-Powered Story to Comic Converter")
st.markdown("Convert your story into a comic strip with dialogue bubbles using AI!")

# Check token
if not HF_TOKEN:
    st.error("⚠️ Hugging Face API token not found! Please set HF_API_TOKEN in your .env file.")
    st.stop()

# Input section
story = st.text_area(
    "Enter your story (include dialogue in quotes):", 
    height=200, 
    placeholder='Write a story with dialogue like: The knight said "I will defeat the dragon!" The dragon roared "You cannot defeat me!" They began to fight...'
)

num_panels = st.selectbox("Number of comic panels:", options=[2, 3, 4, 5, 6, 7, 8, 9, 10], index=1)

# Generate button
if st.button("🎨 Generate Comic Strip with Speech Bubbles"):
    if story.strip():
        if len(story.strip()) < 20:
            st.warning("Please enter a longer story (at least 20 characters).")
        else:
            # Create a single progress container that will update
            progress_container = st.empty()
            
            try:
                progress_container.info("🚀 Starting comic generation process...")
                
                panel_paths = generate_comic_panels(story, num_panels, progress_container)
                final_comic = create_comic_strip(panel_paths, progress_container=progress_container)
                
                # Clear progress and show results
                progress_container.empty()
                
                st.success("🎉 Comic strip with speech bubbles created successfully!")
                st.image(final_comic, caption="Your Comic Strip", use_container_width=True)
                
                # Show individual panels
                st.subheader("Individual Panels:")
                cols = st.columns(len(panel_paths))
                for i, panel_path in enumerate(panel_paths):
                    with cols[i]:
                        st.image(panel_path, caption=f"Panel {i+1}", use_container_width=True)
                
                # Download option
                with open(final_comic, "rb") as file:
                    st.download_button(
                        label="📥 Download Comic Strip",
                        data=file.read(),
                        file_name="my_comic_strip.png",
                        mime="image/png"
                    )
                    
            except Exception as e:
                progress_container.error(f"❌ Error: {str(e)}")
                use_local_fallback()
    else:
        st.warning("Please enter a story to generate comic panels.")

# Instructions
with st.expander("ℹ️ How to use"):
    st.markdown("""
    1. **Enter your story** in the text area above
    2. **Include dialogue in quotes** - Example: "Hello there!" or 'How are you?'
    3. **Select number of panels** (2-10)
    4. **Click "Generate Comic Strip"**
    5. **Wait patiently** - AI image generation takes time!
    
    **Tips for better results with speech bubbles:**
    - Use quotation marks for dialogue: "Hello!" or 'Hi there!'
    - Write clear, descriptive action sequences
    - Include character names for context
    - Keep dialogue concise for better bubble appearance
    - Example: 'The hero said "I will save the day!" The villain laughed "Never!" They began their epic battle.'
    
    **Speech Bubble Features:**
    - Automatically detects quoted dialogue
    - Places bubbles in optimal positions
    - Supports multiple speakers per panel
    - Wraps long text automatically
    """)

st.markdown("---")
st.markdown("*Powered by Hugging Face AI Models (FLUX.1-dev & BART) with Custom Speech Bubble Technology*")
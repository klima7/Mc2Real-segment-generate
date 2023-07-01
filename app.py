import tempfile
from glob import glob
from pathlib import Path
import random

import cv2
import numpy as np

import streamlit as st

from converter import Mc2RealConverter


@st.cache_resource
def get_model():
    return Mc2RealConverter()


def get_uploaded_image(file):
    if file is None:
        return None
    
    temp_dir_path = Path(tempfile.mkdtemp())
    path = str(temp_dir_path / file.name)
    with open(path, 'wb') as f:
        f.write(file.read())
    image = cv2.imread(path)
    return image[..., ::-1]


def preprocess_image(mc_image):
    return mc_image.astype(np.float32) / 255


def get_random_image():
    paths = list(glob('sample_images/*.png'))
    path = random.choice(paths)
    image = cv2.imread(str(path))[..., ::-1]
    return image


# global state
if 'mc_image' not in st.session_state:
    st.session_state.mc_image = get_random_image()


# sidebar
st.sidebar.title('Options')
opt_noise = st.sidebar.checkbox('Optimize noise', value=True)
opt_steps = st.sidebar.number_input('Optimization steps', value=100, min_value=0, max_value=1_000, step=100, format='%d', disabled=not opt_noise)
opt_steps = opt_steps if opt_noise else 0
use_seed = st.sidebar.checkbox('Use seed', value=False)
seed = st.sidebar.number_input('Seed', value=0, min_value=0, step=1, format='%d', disabled=not use_seed)
seed = seed if use_seed else None

# main content
st.title('Minecraft to real converter')

with st.expander('Input Minecraft image', expanded=True):

    mc_file = st.file_uploader('Upload')

    if mc_file:
        st.session_state.mc_image = get_uploaded_image(mc_file)
    
    if st.button('Random'):
        st.session_state.mc_image = get_random_image()

convert = st.button('Convert', use_container_width=True)

if opt_noise:
    progress = st.progress(0)

def progress_callback(progress_percent):
    progress.progress(progress_percent, text='Optimizing noise')
        
col_mc, col_real = st.columns(2)

model = get_model()
real_image = None

with col_mc:
    st.image(st.session_state.mc_image, use_column_width='always')

with col_real:
    if convert:
        mc_image_pre = preprocess_image(st.session_state.mc_image)
        with st.spinner('converting'):
            real_image = model(mc_image_pre, opt_steps=opt_steps, seed=seed, progress_callback=progress_callback if opt_noise else None)
        st.image(real_image, use_column_width='always')

if model.last_segmentator_mask is not None:
    with st.expander('Details'):
        st.image(model.last_segmentator_mask, caption='Segmentator mask')
        st.image(model.last_raw_generator_mask, caption='Raw generator mask')
        st.image(model.last_filtered_generator_mask, caption='Filtered generator mask')
        st.image(model.last_img_without_opt, caption='Image without optimization')
        if model.last_img_with_opt is not None:
            st.image(model.last_img_with_opt, caption='Image with optimization')

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


st.title('Minecraft to real converter')

with st.expander('Input Minecraft image', expanded=True):

    mc_file = st.file_uploader('Upload')

    mc_image = get_uploaded_image(mc_file) if mc_file else get_random_image()
    
    if st.button('Random'):
        mc_image = get_random_image()
        
    show_mask = st.checkbox('Show segmentation mask')

col_mc, col_real = st.columns(2)

with col_mc:
    st.image(mc_image)

with col_real:
    model = get_model()
    mc_image_pre = preprocess_image(mc_image)
    real_image = model(mc_image_pre)
    st.image(real_image)

with st.expander('Segmentation masks'):
    masks = model.get_last_masks()
    cols = st.columns(len(masks))
    for col, mask in zip(cols, masks):
        with col:
            st.image(mask)

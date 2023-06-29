import cv2
import tempfile
import numpy as np
from pathlib import Path

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


st.title('Minecraft to real converter')

mc_file = st.file_uploader('Minecraft image')
mc_image = get_uploaded_image(mc_file)

if mc_image is not None:
    st.image(mc_image)

if st.button('Convert'):
    model = get_model()
    mc_image_pre = preprocess_image(mc_image)
    real_image = model(mc_image_pre)
    st.image(real_image)

import streamlit as st
import requests
from PIL import Image
from io import BytesIO


st.markdown('''
            # Lung Segmentation App
            Select a model, then upload your **CXR** image and 
            choose a pre-process for your input image
            then hit the `submit` button to get your **segmented** mask.    
            For **VAE** just select to numbers between `1` and `30`, 
            hit the `submit` button and get your **generated** image.
            
            ------------------------------------------------------
            ''')
url = "http://0.0.0.0:8000/process"


def post_request(image, model_name, url):
    
    byte_io = BytesIO()
    image.save(byte_io, 'png')
    byte_io.seek(0)

    response = requests.post(url,
                        files={
                            'image': (
                                model_name,
                                byte_io,
                                'image/png'
                            )},)
    return response


with st.form(key='classification'):
    with st.sidebar:
        model_name = st.sidebar.selectbox(
            'Select model',
            [None, "ResNet34", "MobileNet_V2",])

        if model_name == "ResNet34":
            file = st.sidebar.file_uploader("Please upload an image file", type=["jpg", "png"])
            if file is not None:
                image = Image.open(file).convert('RGB')
                st.image(image, use_column_width=True)
                submit_button = st.form_submit_button(label='Submit')
                if submit_button:
                    r = post_request(image, 'resnet', url)
                    st.write(r.text)

        elif model_name == "MobileNet_V2":
            file = st.sidebar.file_uploader("Please upload an image file", type=["jpg", "png"])
            if file is not None:
                image = Image.open(file).convert('RGB')
                st.image(image, use_column_width=True)
                submit_button = st.form_submit_button(label='Submit')
                if submit_button:
                    r = post_request(image, 'mobilenet', url)
                    st.write(r.content)
                    # if responsed.status_code == 200:
                    #     output_ready = True

import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import json
import sys
from streamlit import cli as stcli


st.markdown('''
            # Covid 19 Detection APP
            Select a model, then upload your **CT** image,
            then hit the `submit` button to get your **Prediction**.    

            ------------------------------------------------------
            ''')
url = "http://0.0.0.0:8000/process"

status_code = 400

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
                    preds = json.loads(r.content.decode("utf-8").replace("'",'"'))
                    if r.status_code == 200:
                        status_code = 200

        elif model_name == "MobileNet_V2":
            file = st.sidebar.file_uploader("Please upload an image file", type=["jpg", "png"])
            if file is not None:
                image = Image.open(file).convert('RGB')
                st.image(image, use_column_width=True)
                submit_button = st.form_submit_button(label='Submit')
                if submit_button:
                    r = post_request(image, 'mobilenet', url)
                    preds = json.loads(r.content.decode("utf-8").replace("'",'"'))
                    if r.status_code == 200:
                        status_code = 200

if status_code == 200:
    st.write(f"### `{preds['Class']}`")
    st.image('../images/results.png')


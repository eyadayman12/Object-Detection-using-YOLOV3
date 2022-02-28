import streamlit as st
import numpy as np
from PIL import Image
import helper
import tempfile

st.title("Object Detection")

selected_option =   st.selectbox("Choose an Option", ['Image', 'Video', 'Webcam'])

if selected_option == 'Image':
    image_file = st.file_uploader('Upload Image', type=["png","jpg","jpeg"])
    if image_file is not None:
        img = Image.open(image_file)
        st.image(img, width = 750)
        if st.button("Detect Objects"):
            detect = helper.detect_image(img)
            st.header("Result:")
            st.image(detect, width=750)


elif selected_option == 'Video':
    f = st.file_uploader("Upload Video")
    if f is not None:
        st.text("Note: Press q button to stop streaming")
        if st.button("Show Video"):
            try:
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(f.read())
                vf = tfile.name
                helper.detect_video(vf)
            except:
                pass


else:
    st.text("Note: Press q button to stop streaming")
    if st.button("Open the Camera"):
        helper.detect_webcame()


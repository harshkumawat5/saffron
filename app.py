# import PIL
# from PIL import Image
# import streamlit as st
# from ultralytics import YOLO
# import numpy as np

# # Replace the relative path to your weight file
# model_path = 'weights/best.pt'

# # Specify the desired size for displayed object images
# object_img_size = (100, 100)

# # Setting page layout
# st.set_page_config(
#     page_title="Object Detection using YOLOv8",
#     page_icon="🤖",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Creating sidebar
# with st.sidebar:
#     st.header("Image")
#     source_img = st.file_uploader(
#         "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

#     confidence = float(st.slider(
#         "Select Model Confidence", 25, 100, 40)) / 100

# # Creating main page heading
# st.title("Object Detection using YOLOv8")

# # Creating two columns on the main page
# col1, col2 = st.columns(2)

# # Adding image to the first column if an image is uploaded
# with col1:
#     if source_img:
#         uploaded_image = PIL.Image.open(source_img)
#         st.image(source_img, caption="Uploaded Image", use_column_width=True)

# try:
#     model = YOLO(model_path)
# except Exception as ex:
#     st.error(f"Unable to load the model. Check the specified path: {model_path}")
#     st.error(ex)

# if st.sidebar.button('Detect Objects'):
#     res = model.predict(uploaded_image, conf=confidence)
#     boxes = res[0].boxes
#     res_plotted = res[0].plot()[:, :, ::-1]
#     with col2:
#         st.image(res_plotted, caption='Detected Image', use_column_width=True)

#     # Display detected objects separately at the bottom of the app
#     st.subheader("Detected Objects")

#     st.write(f'<p style="font-size:50px;">Total number of Saffron Flowers:{len(boxes)}</p>',unsafe_allow_html=True)

#     for i, box in enumerate(boxes):
#         st.write(f"Object {i + 1} - Coordinates: {box.xyxy}")
#         # Extract and resize the object image using the coordinates
#         xmin, ymin, xmax, ymax = box.xyxy[0]
#         object_img = np.array(uploaded_image)[int(ymin):int(ymax), int(xmin):int(xmax)]
        
#         # Resize the object image to the specified size with anti-aliasing
#         object_img = Image.fromarray(object_img).resize(object_img_size,Image.BILINEAR)

#         # Display the resized object image
#         st.image(object_img, caption=f"Object {i + 1}")

import PIL
from PIL import Image
import streamlit as st
from ultralytics import YOLO
import numpy as np
import csv

# Replace the relative path to your weight file
model_path = 'weights/best.pt'

# Specify the desired size for displayed object images
object_img_size = (100, 100)

# Setting page layout
st.set_page_config(
    page_title="Object Detection and Pixel Values",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Creating sidebar
with st.sidebar:
    st.header("Image Config")
    source_img = st.file_uploader("Choose an image...", type=("jpg", "jpeg", "png", "bmp", "webp"))
    confidence = st.slider("Select Model Confidence", 0.0, 1.0, 0.25)


# # Creating main page heading
st.title("Object Detection using YOLOv8")

# # Creating two columns on the main page
col1, col2 = st.columns(2)

# # Adding image to the first column if an image is uploaded
with col1:
    if source_img:
        uploaded_image = PIL.Image.open(source_img)
        st.image(source_img, caption="Uploaded Image", use_column_width=True)

# Create a placeholder for displaying the extracted objects

try:
    model = YOLO(model_path)
    # model = model_path
except Exception as ex:
    st.error(f"Unable to load the model. Check the specified path: {model_path}")
    st.error(ex)

if st.sidebar.button('Detect Objects'):
    res = model.predict(uploaded_image, conf=confidence)
    boxes = res[0].boxes
    res_plotted = res[0].plot()[:, :, ::-1]
    with col2:
        st.image(res_plotted, caption='Detected Image', use_column_width=True)

    # Display detected objects separately at the bottom of the app
    st.subheader("Detected Objects")

    st.write(f'<p style="font-size:50px;">Total number of Saffron Flowers:{len(boxes)}</p>',unsafe_allow_html=True)
    objects_placeholder = st.empty()

    for i, box in enumerate(boxes):
        st.write(f"Object {i + 1} - Coordinates: {box.xyxy}")
        # Extract and resize the object image using the coordinates
        xmin, ymin, xmax, ymax = box.xyxy[0]
        object_img = np.array(uploaded_image)[int(ymin):int(ymax), int(xmin):int(xmax)]
        
        # Resize the object image to the specified size with anti-aliasing
        object_img = Image.fromarray(object_img).resize(object_img_size,Image.BILINEAR)

        # Display the resized object image
        st.image(object_img, caption=f"Object {i + 1}")

        # Create and open the CSV file for writing
        with open('object_data.csv', 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['Object', 'X', 'Y', 'R', 'G', 'B'])

            unique_values = set()

            for i, box in enumerate(boxes):
                x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
                object_img = np.array(uploaded_image)[y_min:y_max, x_min:x_max]

                for x in range(x_min, x_max):
                    for y in range(y_min, y_max):
                        pixel_value = (object_img[y - y_min, x - x_min]) 
                        r, g, b = pixel_value
                        unique_values.add((x, y, r,g,b))
                        csv_writer.writerow([i + 1, x, y, r, g, b])
                    
        unique_count = len(unique_values)
        ratio = unique_count / (uploaded_image.width * uploaded_image.height)

        # Display the count and ratio

        objects_placeholder.write(f'<p style="font-size:27px;font-style:italic;">Percentage of Unique Values to Total Values: {ratio*100:.2f}%</p>',unsafe_allow_html=True)
import streamlit as st #success this version does not support linear regression
# To make things easier later, we're also importing numpy and pandas for
# working with sample data. Need to upload this version to an ipad but we need to cut down on the face landmarks to save processing power
import cv2
import av
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import altair as alt
import plotly.express as px
import pandas as pd
from gsheetsdb import connect
# from streamlit_webrtc import (
#     AudioProcessorBase,
#     ClientSettings,
#     VideoProcessorBase,
#     WebRtcMode,
#     webrtc_streamer,
# )

# WEBRTC_CLIENT_SETTINGS = ClientSettings(
#     rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
#     media_stream_constraints={"video": True, "audio": False},
# )


EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}


# Optional if you are using a GPU
# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)


# Create a connection object.
conn = connect()

# Perform SQL query on the Google Sheet.
# Uses st.cache to only rerun when the query changes or after 10 min.
@st.cache(ttl=600)
def run_query(query):
    rows = conn.execute(query, headers=1)
    return rows


def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]
        
        if (c1 > confidence_threshold) & (c2 > confidence_threshold):      
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 4)


def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    
    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 6, (0,255,0), -1)



# Function to loop through each person detected and render
def loop_through_people(frame, keypoints_with_scores, edges, confidence_threshold):
    for person in keypoints_with_scores:
        draw_connections(frame, person, edges, confidence_threshold)
        draw_keypoints(frame, person, confidence_threshold)

def radar_chart(row1, row2, row3):  
    df = pd.DataFrame(dict(
    r=[row1,
       row2,
       row3],
    theta=['Preparation','Swing','Finished',
           ]))
    fig = px.line_polar(df, r='r', theta='theta', line_close=True)
    st.write(fig)

def row_style(row):
    if row.Name != 'Total':
        if row.Status == 'Intervention':
            return pd.Series('background-color: red', row.index)
        else:
            return pd.Series('background-color: green', row.index)
    else:
        return pd.Series('', row.index)

#load model 

# model = hub.load('https://tfhub.dev/google/movenet/multipose/lightning/1')
# movenet = model.signatures['serving_default']


st.title('MOVE Teacher Dashboard Prototype V1')

#access the database and send the data to google sheet




class_df = pd.DataFrame({
    'sec column': ['Sec 1', 'Sec 2', 'Sec 3', 'Sec 4', 'Sec 5'],
    'third column': ['1', '2', '3', '4', '5'],
    })


class_df = pd.DataFrame({
    'sec column': ['Sec 1', 'Sec 2', 'Sec 3', 'Sec 4', 'Sec 5'],
    'third column': ['1', '2', '3', '4', '5'],
    })

stream_df = pd.DataFrame({
  'str column': ['Express', 'Normal Academic', 'Normal Tech'],
  })


with st.sidebar.form("Post PE Class Analysis"):
    code = st.text_input('Class Code')
    name = st.text_input('Name')
    age = st.slider('Age', min_value = 12, max_value = 17, value = 15, step=1)
    level = st.selectbox('Select your level:',class_df['sec column'])
    stream = st.selectbox('Select your stream:', stream_df['str column'])
    class_no = st.selectbox('Select your class:',class_df['third column'])
    submit_button = st.form_submit_button()


live_code = st.text_input('Please enter the current close code for live feedback:')


st.subheader('Class Analysis for class: ' + str(live_code))


# class OpenCVVideoProcessor(VideoProcessorBase):
#     def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
#         in_image = frame.to_ndarray(format="bgr24")

#         # Resize image
#         img = in_image.copy()
#         img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 384,640)
#         input_img = tf.cast(img, dtype=tf.int32)
        
#         # Detection section
#         results = movenet(input_img)
#         keypoints_with_scores = results['output_0'].numpy()[:,:,:51].reshape((6,17,3))
        
#         # Render keypoints 
#         loop_through_people(in_image, keypoints_with_scores, EDGES, 0.1)

#         return av.VideoFrame.from_ndarray(in_image, format="bgr24")

# webrtc_ctx = webrtc_streamer(
#         key="opencv-filter",
#         mode=WebRtcMode.SENDRECV,
#         client_settings=WEBRTC_CLIENT_SETTINGS,
#         video_processor_factory=OpenCVVideoProcessor,
#         async_transform=True,
#     )



sheet_url = st.secrets["public_gsheets_url"]
# rows = run_query(f'SELECT * FROM "{sheet_url}"')

# # Print results.
# for row in rows:
#     st.write(f"{row.name} has a :{row.pet}:")

class_data = pd.read_excel(sheet_url)

df = class_data.loc[:,'Class':'Status']

#st.dataframe(df,1000,700)

st.dataframe(df.style.apply(row_style, axis=1),1000,700)

no_prep = (df['Preparation'] > 0.7).sum()
no_swing = (df['Swing'] > 0.7).sum()
no_finished = (df['Finished'] > 0.7).sum()

#print(gpus)


st.subheader('Post Class Analysis')

#Sample Data 

df = pd.DataFrame(
	np.random.randn(200, 3),
	columns=['a', 'b', 'c'])
c = alt.Chart(df).mark_circle().encode(
	x='a', y='b', size='c', color='c', tooltip=['a', 'b', 'c'])

# val = st.slider('Select value',0,10,1)


if st.button("Analyse"):
	#st.write(c)
    radar_chart(no_prep, no_swing, no_finished)
    

import pandas as pd
import numpy as np


xyz=pd.read_csv("initial_model\asl-handsigns\xyz_df.csv")  # give your own path hai

import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import PIL.Image

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

# THIS CODE IS COMBINATION OF ALL THE CODES..
# !pip install tflite-runtime
import tflite_runtime.interpreter as tflite


import pandas as pd
import numpy as np
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import PIL.Image
import tflite_runtime.interpreter as tflite

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

ROWS_PER_FRAME = 543  # number of landmarks per frame

def load_relevant_data_subset(pq_path):
    data_columns = ['x', 'y', 'z']
    data = pd.read_parquet(pq_path, columns=data_columns)
    n_frames = int(len(data) / ROWS_PER_FRAME)
    data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
    return data.astype(np.float32)

def load_tflite_model(model_path):
    interpreter = tflite.Interpreter(model_path)
    interpreter.allocate_tensors()
    return interpreter



def preprocess_data(xyz_np):
    # Add any necessary data preprocessing steps here
    return xyz_np

def pred_fn():
#     input_tensor_index = interpreter.get_input_details()[0]['index']
#     output = np.zeros(interpreter.get_output_details()[0]['shape']).astype(np.float32)

#     for frame_data in input_data:
#         interpreter.set_tensor(input_tensor_index, frame_data)
#         interpreter.invoke()
#         output += interpreter.get_tensor(output_index)

    interpreter = tflite.Interpreter("initial_model\asl-handsigns\model.tflite") # give your own path hai
    found_signatures = list(interpreter.get_signature_list().keys())
    prediction_fn = interpreter.get_signature_runner("serving_default")
    return prediction_fn




def create_frame_landmark_df(results,frame,xyz_skel):

          # where
          # results = results of mediapipe
          # frame = frame number
          # xyz = dataset for the xyz of example data


    face = pd.DataFrame()
    if results.face_landmarks is not None:
      for i,point in enumerate(results.face_landmarks.landmark):
        face.loc[i,'x'] = point.x
        face.loc[i,'y'] = point.y
        face.loc[i,'z'] = point.z

      face['y'] = -face['y']


    pose = pd.DataFrame()
    if results.pose_landmarks is not None:
      for i,point in enumerate(results.pose_landmarks.landmark):
        pose.loc[i,'x'] = point.x
        pose.loc[i,'y'] = point.y
        pose.loc[i,'z'] = point.z

    # pose['y'] = -pose['y']

    left_hand = pd.DataFrame()
    if results.left_hand_landmarks is not None:
      for i,point in enumerate(results.left_hand_landmarks.landmark):
        left_hand.loc[i,'x'] = point.x
        left_hand.loc[i,'y'] = point.y
        left_hand.loc[i,'z'] = point.z

    # left_hand['y'] = -left_hand['y']

    right_hand = pd.DataFrame()
    if results.right_hand_landmarks is not None:
      for i,point in enumerate(results.right_hand_landmarks.landmark):
        right_hand.loc[i,'x'] = point.x
        right_hand.loc[i,'y'] = point.y
        right_hand.loc[i,'z'] = point.z

    # right_hand['y'] = -right_hand['y']

    face = face.reset_index() \
                .rename(columns={'index': 'landmark_index'}) \
                .assign(type='face')
    pose = pose.reset_index() \
                .rename(columns={'index': 'landmark_index'}) \
                .assign(type='pose')
    left_hand = left_hand.reset_index() \
                .rename(columns={'index': 'landmark_index'}) \
                .assign(type='left_hand')
    right_hand = right_hand.reset_index() \
                .rename(columns={'index': 'landmark_index'}) \
                .assign(type='right_hand')
    face_reset = face.reset_index(drop=True)
    pose_reset = pose.reset_index(drop=True)
    left_hand_reset = left_hand.reset_index(drop=True)
    right_hand_reset = right_hand.reset_index(drop=True)

    # Concatenating the DataFrames
    landmarks = pd.concat([face_reset, pose_reset, left_hand_reset, right_hand_reset]).reset_index(drop=True)
    landmarks=xyz_skel.merge(landmarks,on=['type','landmark_index'],how='left')
    landmarks = landmarks.assign(frame=frame)

    return landmarks

def caturing_video(xyz):

    prediction_fn = pred_fn()
    all_landmarks = []
    video_path = 'initial_model\asl-handsigns\demo.mp4' # give your own path hai
    cap = cv2.VideoCapture(video_path)
    xyz_skel = xyz[['type','landmark_index']].drop_duplicates().drop_duplicates().reset_index(drop=True).copy()

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        frame = 0
        while cap.isOpened():
            frame += 1
            success, image = cap.read()
            if not success:
                print("End of video.")
                break

            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)

            # Create landmark dataframe
            landmarks = create_frame_landmark_df(results, frame, xyz_skel)
            all_landmarks.append(landmarks)


        # Save the DataFrame to a parquet file
        output_path = 'output.parquet'
        all_landmarks_df = pd.concat(all_landmarks, ignore_index=True)
        all_landmarks_df.to_parquet(output_path, index=False)
        print(f"Landmarks saved to {output_path}")


        # Make predictions using the loaded tflite model
        xyz_np = load_relevant_data_subset('output.parquet')
        output = prediction_fn(inputs=xyz_np)
        sign = output['outputs'].argmax()
        print(f"Predicted Sign: {sign}")


    cap.release()



if __name__ == "__main__":
    xyz = pd.read_csv("initial_model\asl-handsigns\xyz_df.csv")  # give your own path hai

    # Capture video and make predictions
    caturing_video(xyz)


# Dictionaries to translate sign <-> ordinal encoded sign
SIGN2ORD = train[['sign', 'sign_ord']].set_index('sign').squeeze().to_dict()
ORD2SIGN = train[['sign_ord', 'sign']].set_index('sign_ord').squeeze().to_dict()
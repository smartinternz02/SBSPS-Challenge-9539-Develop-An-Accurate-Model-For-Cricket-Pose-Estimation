import uvicorn
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from io import BytesIO
import onnx
import onnxruntime as ort
import mediapipe as mp
import numpy as np
import cv2

mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

results_dict = {
    0: 'drive',
    1: 'legglance-flick',
    2: 'sweep',
    3: 'pullshot'
}

desc = """
This API will be able to predict the cricket shots based on the pose estimates of the given image.
"""

app = FastAPI(
    title="Cricket Shot Prediction using Pose Estimation",
    description=desc, 
    version="1.0.0",
    contact={
        "name": "Aneesh",
        "email": "aneeshaparajit.g2002@gmail.com"
    }, 
    openapi_tags=[
        {
            'name': 'Predict the Shot!',
            'description': 'Upload an image and we\'ll tell you what shot was played!'
        }
    ]
)

def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    return image


def get_pose(image: Image.Image):
    image = np.array(image)
    image_height, image_width, _ = image.shape
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=True,
        min_detection_confidence=0.4) as pose:
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            return None
        pts = np.array([ [data_point.x, data_point.y, data_point.z, data_point.visibility] for data_point in results.pose_landmarks.landmark ])
    return pts


@app.post("/predict/image", tags=['Predict the Shot!'])
async def predict_shot(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    image = read_imagefile(await file.read())
    inputs = get_pose(image)
    if inputs is not None:
        ort_session = ort.InferenceSession("../model.onnx")
        outputs = ort_session.run(
            None,
            {'x': np.random.randn(3, 132).astype(np.float32)},
        )
        ix = np.argmax(outputs[0])
        if ix == 0:
            return "The shot played in the input was a Drive!"
        elif ix == 1:
            return "The shot played in the input was either the Flick or a Leg Glance!"
        elif ix == 2:
            return "The shot played in the input was the Sweep!"
        else:
            return "The shot played in the input was the Pull Shot!"
    return "Unfortunately, our model was not able to capture enough information from the image to predict the shot. Please try again with some other picture."

if __name__ == "__main__":
    uvicorn.run(app, debug=True)
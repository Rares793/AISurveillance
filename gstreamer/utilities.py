import cv2
import numpy as np
import os
import mediapipe as mp
import common

def align_face(image, right_eye, left_eye):
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx)) - 180

    eye_center = (int((left_eye[0] + right_eye[0]) // 2),  int((left_eye[1] + right_eye[1]) // 2))
    M = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
    aligned_face = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), flags = cv2.INTER_CUBIC)
    return aligned_face
    
def detect_and_crop_face_haar(image, face_detector):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))

    if len(faces) == 0:
        return None

    largest_face = max(faces, key=lambda f: f[2]*f[3])
    x, y, w, h = largest_face
    cropped_face = image[y:y+h, x:x+w]
    return cropped_face

def detect_and_crop_face_blazeface(image, face_detector):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detector.process(image_rgb)

    if not results.detections:
        return None
        
    detection  = results.detections[0]
    h, w, _ = image.shape

    # Extract Bounding Box
    bboxC = detection.location_data.relative_bounding_box
    x, y = int(bboxC.xmin * w), int(bboxC.ymin * h)
    box_w, box_h = int(bboxC.width * w), int(bboxC.height * h)

    # Extract Eye Landmarks
    keypoints = detection.location_data.relative_keypoints
    right_eye = (int(keypoints[0].x * w), int(keypoints[0].y * h))
    left_eye = (int(keypoints[1].x * w), int(keypoints[1].y * h))

    aligned_image = align_face(image, right_eye, left_eye)

    x = max(0, x)
    y = max(0, y)
    aligned_crop = aligned_image[y:y+box_h, x:x+box_w]

    return aligned_crop #, (x, y, box_w, box_h)
    
def detect_and_crop_face(image, face_detector):
    if type(face_detector).__name__ == "CascadeClassifier":
        return detect_and_crop_face_haar(image, face_detector)
    return detect_and_crop_face_blazeface(image, face_detector)

def get_body_crop(td, interpreter):
    # coordinates between 0 and 1, relative to input tensor size
    x0, y0, x1, y1 = td[0].item(), td[1].item(), td[2].item(), td[3].item()

    # get current frame as a NumPy array
    camera_img = common.input_tensor(interpreter)
    img_h, img_w, _ =  camera_img.shape

    # Convert decimal coordinates to absolute pixel coordinates
    px0 = max(0, int(x0 * img_w))
    py0 = max(0, int(y0 * img_h))
    px1 = min(img_w, int(x1 * img_w))
    py1 = min(img_h, int(y1 * img_h))

    body_crop = camera_img[py0:py1, px0:px1]

    if body_crop.size == 0:
        return None
    return body_crop

def pre_process_face(face_crop):
    face_resized = cv2.resize(face_crop, (112, 112))

    # the original paper normalised its data as such, the internet sometimes just divides by 255
    face_normalised = (np.float32(face_resized) - 127.5) / 128
    face_input = np.expand_dims(face_normalised, axis=0)

    return face_input
    
def print_face_fingerprint(name, embedding):
    print(f"Name: {name}")
    num_features = embedding.shape[0] if len(embedding.shape) > 0 else len(embedding)
    print(f"Array Shape: {embedding.shape} (It has {num_features} unique features)")
    print(f"Data Type: {embedding.dtype}")
    print(f"{num_features} dimensional coordinates:")

    formatted_nums = [f"{num: 6.4f}" for num in embedding]

    for i in range(0, len(formatted_nums), 8):
        row = " | ".join(formatted_nums[i:i+8])
        print(f" [ {row} ]")

def load_face_detector(detector):
    if detector is None:
        return None
        
    detector = detector.lower()
    if detector == "haar":
        print("Loading OpenCV's Haar Cascade Face Detector")
        cascade_path = "haarcascade_frontalface_default.xml"
        if not os.path.exists(cascade_path):
            cascade_path = "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml"

        return cv2.CascadeClassifier(cascade_path)

    if detector == "blazeface":
        mp_face_detection = mp.solutions.face_detection
        return mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

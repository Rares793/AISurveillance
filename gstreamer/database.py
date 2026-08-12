# New file not present in the original repository

import os
import json
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import utilities as ut

# --- Configuration ---
DB_FILE = 'face_database.json'
NEW_FACES_DIR = 'new_faces'
MODEL_PATH = '../models/mobilefacenet.tflite'
MATCH_THRESHOLD = 0.8  # Euclidean distance threshold

def load_database():
    if not os.path.exists(DB_FILE):
        return {}
    
    with open(DB_FILE, 'r') as f:
        raw_db = json.load(f)
        
    return {name: np.array(embedding) for name, embedding in raw_db.items()}

def save_database(db):
    raw_db = {name: embedding.tolist() for name, embedding in db.items()}
    
    with open(DB_FILE, 'w') as f:
        json.dump(raw_db, f, indent=4)

# well, it was a TEMPORARY solution, but eh 
def get_embedding(image_path, interpreter, input_details, output_details, detector):
    # Reads an image, formats it, and extracts the 192-number fingerprint
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return None

    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    # MobileFaceNet wants 112x112 

    if detector is not None:
        face_crop = ut.detect_and_crop_face(img, detector)
    
        if face_crop is None:
            print("[Debug] No face found in picture")
            return None
    
        img_input = ut.pre_process_face(face_crop)
    else:
        img_input = ut.pre_process_face(img)
        
    # Run the model
    interpreter.set_tensor(input_details[0]['index'], img_input)
    interpreter.invoke()
    
    # Return the 192-number array
    return interpreter.get_tensor(output_details[0]['index'])[0]

def compare_with_database(new_embedding, db, threshold=MATCH_THRESHOLD):
    # Compares a fingerprint against the database
    best_match_name = None
    lowest_distance = float('inf')

    for name, db_embedding in db.items():
        # Calculate Euclidean distance
        distance = np.linalg.norm(db_embedding - new_embedding)
        
        if distance < lowest_distance:
            lowest_distance = distance
            best_match_name = name

    if lowest_distance < threshold:
        return best_match_name, lowest_distance
        
    return None, lowest_distance

# It will update the face_db if it adds something new so no need to reinitialize it after every scan
def scan_folder(face_db, detector):
    # Main function to scan the folder, update the D and delete images to not waste space
    # Create the folder if it doesn't exist
    if not os.path.exists(NEW_FACES_DIR):
        os.makedirs(NEW_FACES_DIR)
        print(f"Created directory '{NEW_FACES_DIR}'. Drop headshot images here.")
        return
    
    # Load the TFLite model on the CPU
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Iterate through all files in the folder
    for filename in os.listdir(NEW_FACES_DIR):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        filepath = os.path.join(NEW_FACES_DIR, filename)
        
        # Use the filename (minus the extension) as the person's name
        name = os.path.splitext(filename)[0] 
        
        print(f"\nProcessing {filename}...")
        embedding = get_embedding(filepath, interpreter, input_details, output_details, detector)
        
        if embedding is None:
            print(f"  -> Error: Could not read image. Skipping.")
            continue

        # Check if they are already in the system
        match_name, distance = compare_with_database(embedding, face_db)
        
        if match_name:
            print(f"  -> Match found! This looks like {match_name} (Distance: {distance:.2f}).")
            print(f"  -> Deleting duplicate image.")
        else:
            print(f"  -> New face detected. Adding '{name}' to database.")
            face_db[name] = embedding
            save_database(face_db)
            
        # We process the image successfully, so we delete it
        os.remove(filepath)
        print(f"  -> Deleted {filepath}.")
        
    print("\nScan complete. Database is up to date.")

if __name__ == "__main__":
    scan_folder()

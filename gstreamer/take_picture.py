import argparse
import os
import time
import signal
import cv2
import numpy as np
import svgwrite
import gstreamer

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

def extract_image_from_buffer(buff, width, height):
    # Takes the image from Gstreamer and converts it into numpy array
    result, mapinfo = buff.map(Gst.MapFlags.READ)
    if result:
        img_array = np.ndarray((height, width, 3), buffer=mapinfo.data, dtype = np.uint8)
        buff.unmap(mapinfo)
        return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    return None

def main():
    parser = argparse.ArgumentParser()  
    parser.add_argument('--videosrc', help='Which video source to use', default='/dev/video0')
    parser.add_argument('--videofmt', help='Input video format', default='raw', choices=['raw', 'h264', 'jpeg'])
    args = parser.parse_args()

    print("Stand in front of the camera\n")
    print("Taking a picture in 10 seconds\n")

    start_time = time.monotonic()
    last_second = 10
    picture_taken = False
    captured_img = None

    def user_callback(input_tensor, src_size, inference_box, mot_tracker):
        nonlocal start_time, last_second, picture_taken, captured_img

        if picture_taken:
            return None

        elapsed = time.monotonic() - start_time
        remaining = 10 - int(elapsed)

        if remaining < last_second and remaining > 0:
            print(f"Taking picture in {remaining}...\n")
            last_second = remaining

        dw = svgwrite.Drawing('', size=src_size)

        if remaining <= 0:
            print("Capturing image\n")
            picture_taken = True
            captured_img = extract_image_from_buffer(input_tensor, src_size[0], src_size[1])
            os.kill(os.getpid(), signal.SIGINT)

        if not picture_taken:
            countdown_text = f"Snapshot in {remaining}s"
            dw.add(dw.text(countdown_text, insert=(22,52), fill="black", font_size='40', font_weight='bold'))
            dw.add(dw.text(countdown_text, insert=(20,50), fill="red", font_size='40', font_weight='bold'))

            center_x, center_y = src_size[0] / 2, src_size[1] / 2
            box_size = 200
            dw.add(dw.rect(insert=(center_x - box_size/2, center_y - box_size/2), size=(box_size, box_size), fill='none', stroke='yellow', stroke_width='3', stroke_dasharray="10,10"))
        return dw.tostring()

    gstreamer.run_pipeline(user_callback, src_size=(640,480), appsink_size=(640,480), trackerName=None, videosrc= args.videosrc, videofmt = args.videofmt)

    if captured_img is not None:
        print("\n" + "="*36)
        name = input("Enter the person's name: ").strip()

        if not name:
            name = "Unknown"

        save_dir = 'new_faces'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        filepath = os.path.join(save_dir, f"{name}.jpg")
        cv2.imwrite(filepath, captured_img)

        print(f"Picture saved to: {filepath}")

if __name__ == '__main__':
    main()

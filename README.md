# AISurveillance

A face-recognition camera that runs on Google's Coral Dev Board. It detects and tracks people using the camera's feed and if it identifies them, it also tags them with the name associated in the database.

This project is a fork of Google's (https://github.com/google-coral/example-object-tracker). The camera pipeline, the SSD detection setup and the SORT integration come from that repo whereas the database and face identification are new.

- **Person detection** runs on the Edge TPU using the COCO SSD MobileNet v2 model, filtered to the `person` class.
- **Tracking** uses [SORT](https://github.com/abewley/sort). Each person gets a track ID, so recognition only has to succeed once per track. After a match, the name stays attached to that track.
- **Face recognition** uses MobileFaceNet (`models/mobilefacenet.tflite`), which runs on the CPU through the TFLite runtime. It turns a face into a 192-number embedding that is compared with the enrolled ones.
- **Face detection** (to find the face inside a person's box) is selectable: OpenCV's Haar cascade, or MediaPipe BlazeFace, which also gives eye positions used to straighten the face before it is embedded.
- **Enrollment** needs only one frontal photo per person. Only the embedding is stored, and the photo is deleted after processing.

## Requirements

- Google Coral Dev Board with the Coral camera (a USB webcam also works, see `--videosrc` which was done by Google so you could also check their repo for more detailed instructions)
- The Google Coral setup completed
- A microSD card for the Board's initial setup ( to install the Board's OS, Mendel)


## Installation

```bash
git clone https://github.com/Rares793/AISurveillance.git
cd AISurveillance
sh download_models.sh          # downloads the SSD model and COCO labels into models/
cd gstreamer
bash install_requirements.sh   # GStreamer, svgwrite and the SORT tracker
```

You will have to install OpenCV seperately whereas MediaPipe will have to be compiled for the board which uses python 3.7.9 I believe and some older C++ versions too. Good luck.

## Usage

Either drop a .png in the new_faces folder or use the script provided to take a picture using the Coral's camera so that you have something in the databse to work with.

Run the camera

```bash
python3 detect.py --tracker sort --detector haar
```

On startup, any images in the new_faces folder are given to MobileFaceNet to add the processed result to the face_database (a json file for simplicity). If the person is already in the database, the face isn't added. The image will always be deleted if it is identified, if not, it shall remain in the folder until time immemorial.

### Options

| Flag | Description |
|---|---|
| `--tracker sort` | Enable SORT tracking (required for recognition) |
| `--detector {haar,blazeface}` | Face detector used inside each person box |
| `--blur` | Apply a 3×3 Gaussian blur to person crops to reduce camera noise |
| `--debug_messages` | Print debug output, including embeddings |
| `--videosrc` | Video device, default `/dev/video0` |
| `--videofmt {raw,h264,jpeg}` | Input video format |
| `--threshold`, `--top_k` | Detection score threshold and number of detections kept |

The overlay shows inference time and FPS in the top-left corner.

## Limitations

- The pictures provided for the database should be frontal (like the ones on drivers licenses and whatnot).
- A name is tied to a SORT track ID. As such, if the person disappears for a very small amount of time, they will have to be reidentified
- Face recognition runs on the CPU, not the Edge TPU. It was done to offload some of the work from the TPU

## Licenses and credits

- Original example code: Copyright 2019 Google LLC, Apache License 2.0 (see `LICENSE`).
- [SORT](https://github.com/abewley/sort) by Alex Bewley, GPLv3, in `third_party/`.
- MobileFaceNet is used for face embeddings.

And a very special thank you to the mentor who helped me out with this project, Jeong Pangyu, who was more patient than he should have been with me throughout the duration of my summer practice at the company he worked at.

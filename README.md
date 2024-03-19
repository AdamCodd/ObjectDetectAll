# ObjectDetectAll
**ObjectDetectAll** is a comprehensive toolkit for object detection across various media types, including images, GIFs, and videos. Utilizing state-of-the-art object detection models (YOLOS by default), this project allows users to detect objects and draw bounding boxes with labels across different media formats seamlessly.

![Cat detection](https://github.com/AdamCodd/ObjectDetectAll/blob/main/cat_demo.gif)

## Features
* Support for multiple media types: Process any kind of images, GIFs, and videos with a single toolkit.
* Object Detection configuration: Configure labels, detection thresholds.
* Comprehensive FFmpeg settings (batch_size, audio or not, bitrate, duration, etc)
* Auto-download your detection model and convert it into ONNX and quantized ONNX for faster processing
* Handle local and remote files. When URLs are provided (CLI or from a text file) the medias are downloaded automatically.

## Requirements
This project requires:
* Python 3.6+
* [FFmpeg](https://ffmpeg.org/download.html) for video processing. Ensure FFmpeg is installed and accessible in your system's PATH.
* [Gifsicle](https://www.lcdf.org/gifsicle/) for GIF optimization. If not installed, the script will output an non-optimized GIF instead (i.e the output size >> input size).


## Installation
Clone the Repository
```
git clone https://github.com/AdamCodd/ObjectDetectAll.git
cd ObjectDetectAll
```
Ensure you have Python 3.6+ installed, then run:
```
pip install -r requirements.txt
pip install -r requirements-convert.txt
```

## Usage
Basic usage examples for processing different media types:

**Images** (local)
```
python main.py --input path/to/image.jpg --output path/to/output/directory
```
**Images** (remote)
```
python main.py --input https://upload.wikimedia.org/wikipedia/commons/3/3f/JPEG_example_flower.jpg --output path/to/output/directory
```
**GIFs** (local or remote)
```
python main.py --input path/to/animation.gif --output path/to/output/directory
```
**Videos** (local or remote, without audio by default)
```
python main.py --input path/to/video.mp4 --output path/to/output/directory
```

Replace path/to/input and path/to/output/directory with your specific paths. Use the --help flag to see all available options:
```
python main.py --help
```

## Customizing Detection
To specify object labels for detection (others labels will be ignored):
```
python main.py --input path/to/media --output path/to/output --labels person car
```
Adjust detection sensitivity using the --threshold option (default 0.9):
```
python main.py --input path/to/media --output path/to/output --threshold 0.5
```
NB: If the threshold is decreased (from 0.9), there will be an increase in false positives.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing
If you'd like to contribute, please fork the repository and use a feature branch. Pull requests are warmly welcome.

## Acknowledgments
This project utilizes [ONNX Runtime](https://github.com/microsoft/onnxruntime), the [Transformers](https://github.com/huggingface/transformers) library from Hugging Face for object detection models and a slightly modified version of the convert.py script from [Transformers.js](https://github.com/xenova/transformers.js).

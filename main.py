import argparse
import json
import os
import requests
import subprocess
import cv2
import time
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont, ImageSequence
from concurrent.futures import ThreadPoolExecutor, as_completed
import onnxruntime as ort
from transformers import AutoFeatureExtractor
import torch
import numpy as np
import pathlib
import shutil
import tempfile
from typing import List, Tuple, Union, Any, Dict, Optional
import mimetypes
from convert import ConversionArguments, main as quantize_main

def convert_and_quantize_model(model_id, output_dir, quantize=True, task='auto', device='cpu') -> None:
    """
    Converts a model from the Hugging Face Hub to ONNX and optionally quantizes it.

    Args:
    - model_id: The ID of the model on HuggingFace Hub.
    - output_dir: The directory where the converted model will be saved.
    - quantize: Whether to quantize the model. Defaults to True.
    - task: The task for which the model is optimized. Defaults to 'auto'.
    - device: The device to perform the conversion and quantization on. Defaults to 'cpu'.
    """
    # Create a ConversionArguments instance with the necessary settings
    conversion_args = ConversionArguments(
        model_id=model_id,
        output_parent_dir=output_dir,
        quantize=quantize,
        task=task,
        device=device
    )
    quantize_main(conversion_args)

def get_media_type(path_or_url: str) -> Optional[str]:
    mime_type, _ = mimetypes.guess_type(path_or_url)
    if mime_type:
        if mime_type.startswith('image/'):
            if 'gif' in mime_type:
                return 'gif'
            return 'image'
        elif mime_type.startswith('video/'):
            return 'video'
    return None

def generate_media_output_path(base_output_dir: pathlib.Path, media_source: str, prefix: str = 'out', media_type: str = None) -> pathlib.Path:
    """
    Generates an output path for media files based on the media type.
    """
    if media_type is None:
        media_type = get_media_type(media_source)
    
    # Extract the filename without extension and the extension itself
    source_filename = pathlib.Path(media_source).stem
    source_extension = pathlib.Path(media_source).suffix

    # Determine the output file name based on the media type
    if media_type == 'gif':
        output_filename = f"{prefix}_{source_filename}.gif"
    elif media_type == 'video':
        # Preserving the original extension for videos
        output_filename = f"{prefix}_{source_filename}{source_extension}"
    else:  # Default to image
        output_filename = f"{prefix}_{source_filename}.jpg"

    # Construct the full output path
    output_path = base_output_dir / output_filename

    return output_path

class Drawer:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Drawer, cls).__new__(cls)
            # Initialize the instance once
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, id2label, font_path="arial.ttf", font_size=15) -> None:
        # Check if the instance has been initialized
        if not self._initialized:
            self.id2label = id2label
            self.font_size = font_size
            self.font = self.load_font(font_path, font_size)
            # Mark the instance as initialized
            self._initialized = True

    def load_font(self, font_path: str, font_size: int) -> ImageFont.ImageFont:
        """Attempts to load a TrueType font and falls back to default font if not found."""
        try:
            font = ImageFont.truetype(font_path, font_size)
        except IOError:
            print("TrueType font not found; using default bitmap font.")
            font = ImageFont.load_default()
        return font

    def draw_boxes(self, image: Image.Image, outputs: List[torch.Tensor], threshold: float = 0.9) -> Image.Image:
        logits, pred_boxes = outputs
        probas = logits.softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > threshold
        bboxes_scaled = self.rescale_bboxes(pred_boxes[0, keep].cpu(), image.size)

        draw = ImageDraw.Draw(image, "RGBA")
        box_colors = ["red", "green", "blue", "cyan", "purple", "orange"]

        # Use a sample text to pre-compute text size; actual text dimensions might vary slightly
        text_bbox = draw.textbbox((0, 0), "Sample Text", font=self.font)
        text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]

        for i, (p, box) in enumerate(zip(probas[keep], bboxes_scaled)):
            cl = p.argmax()
            label = self.id2label.get(str(cl.item()))
            
            if label is None:
                continue
            
            text = f'{label}: {p[cl]:0.2f}'
            draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline=box_colors[i % len(box_colors)], width=3)
            text_background = [box[0], box[1], box[0] + text_width, box[1] + text_height]
            draw.rectangle(text_background, fill=(255, 255, 0, 128))  # Semi-transparent background
            draw.text((box[0], box[1]), text, fill="black", font=self.font)

        return image

    @staticmethod
    def rescale_bboxes(out_bbox: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        img_w, img_h = size
        b = [(out_bbox[:, 0] - 0.5 * out_bbox[:, 2]) * img_w, (out_bbox[:, 1] - 0.5 * out_bbox[:, 3]) * img_h,
             (out_bbox[:, 0] + 0.5 * out_bbox[:, 2]) * img_w, (out_bbox[:, 1] + 0.5 * out_bbox[:, 3]) * img_h]
        return torch.stack(b, dim=1)

class Visualizer:
    def __init__(self, config_path: pathlib.Path, model_path: pathlib.Path, model_base_path: pathlib.Path, selected_labels: Optional[List[str]] = None) -> None:
        with config_path.open() as f:
            self.config = json.load(f)
        self.id2label = self.config["id2label"]
        self.model_path = str(model_path) # ONNX file
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.ort_session = ort.InferenceSession(self.model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_base_path)
        
        if selected_labels is not None:
            # Filter id2label to include only selected labels if any
            filtered_id2label = {id: label for id, label in self.id2label.items() if label in selected_labels}
            self.drawer = Drawer(filtered_id2label)
        else:
            self.drawer = Drawer(self.config["id2label"])

    def load_gif(self, file: str, image_path_or_url: str, filename_prefix: str, output_dir: pathlib.Path, threshold: float, chunk_size: int) -> None:
        """
        Loads a GIF from a file or URL, processes each frame to detect objects, and saves the processed frames as a new GIF.
        """
        gif = Image.open(file)
        frames = [frame.copy().convert("RGB") for frame in ImageSequence.Iterator(gif)]
        processed_frames = self.process_media_batch(frames, None, filename_prefix, image_path_or_url, threshold, batch_size=chunk_size, is_video=True)

        # Prepare output path
        output_path = generate_media_output_path(output_dir, image_path_or_url, filename_prefix, 'gif')

        # Optimize GIF processing by using in-memory operations
        with BytesIO() as gif_bytes:
            processed_frames[0].save(gif_bytes, format='GIF', save_all=True, append_images=processed_frames[1:], loop=0, duration=gif.info['duration'], disposal=2)
            gif_bytes.seek(0)  # Rewind to the start of the BytesIO object

            if shutil.which("gifsicle"):
                # Run gifsicle to optimize the GIF directly from memory and handle subprocess output and errors
                process = subprocess.run(["gifsicle", "-O3", "--colors", "256", "-o", "-"], input=gif_bytes.read(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if process.returncode == 0:
                    with open(output_path, "wb") as output_file:
                        output_file.write(process.stdout)
                    print(f"Optimized GIF saved to {output_path}")
                else:
                    print(f"gifsicle error: {process.stderr.decode()}")
            else:
                # Fallback: Save the GIF without optimization if gifsicle is not found
                # No need to read gif_bytes again, just use the existing buffer
                with open(output_path, "wb") as output_file:
                    output_file.write(gif_bytes.getvalue())
                print("gifsicle not found. Saving unoptimized GIF.")

    def preprocess_images_batch(self, images: List[Image.Image], is_video: bool = False) -> torch.Tensor:
        """Preprocess a batch of images without resizing if it's video content or if all images are of the same size."""
        if is_video:
            # For gif/video frames, we bypass the resizing logic entirely.
            resized_images = images
        else:
            # Determine if all images are of the same size
            all_same_size = all(image.size == images[0].size for image in images)

            if not all_same_size:
                # Find the smallest width and height in the batch.
                min_width = min(image.width for image in images)
                min_height = min(image.height for image in images)
                # Resize images to the smallest dimensions found, if necessary.
                resized_images = [image.resize((min_width, min_height), Image.ANTIALIAS) for image in images]
            else:
                resized_images = images

        pixel_values_list = [self.feature_extractor(images=image, return_tensors="pt").pixel_values for image in resized_images]
        pixel_values_batch = torch.cat(pixel_values_list, dim=0)
        return pixel_values_batch.to(self.device)

    def predict_batch(self, pixel_values_batch: torch.Tensor) -> List[np.ndarray]:
        """Predict a batch of images."""
        inputs = {self.ort_session.get_inputs()[0].name: pixel_values_batch.cpu().numpy()}
        outputs = self.ort_session.run(None, inputs)
        return [np.split(output, output.shape[0], axis=0) for output in outputs] 

    def process_media_batch(self, media_batch: List[Image.Image], output_path: Optional[str], prefix: str, batch_paths: Optional[List[str]], threshold: float, batch_size: int, is_video: bool = False) -> List[Image.Image]:
        """
        Processes a batch of media (images or video frames), applying object detection and optionally saving the processed media.
        """
        processed_media = []

        # Iterate over media_batch in chunks of batch_size
        for start_idx in range(0, len(media_batch), batch_size):
            end_idx = min(start_idx + batch_size, len(media_batch))
            chunk = media_batch[start_idx:end_idx]
            
            # Process each chunk as a separate batch
            pixel_values_batch = self.preprocess_images_batch(chunk, is_video=is_video)
            outputs = self.predict_batch(pixel_values_batch)

            for idx, media in enumerate(chunk):
                image_outputs = [torch.from_numpy(output[idx]).to(self.device) for output in outputs]
                processed_image = self.drawer.draw_boxes(media, image_outputs, threshold)

                # Handling for saving images
                if not is_video:
                    media_source = batch_paths[start_idx + idx] if batch_paths else f'image_{start_idx + idx}'
                    full_path = generate_media_output_path(pathlib.Path(output_path), media_source, prefix, 'image')
                    processed_image.save(full_path)
                    print(f"Image saved to {full_path}")

                processed_media.append(processed_image)

        return processed_media

    def process_video_frames(self, frames_batch: List[Image.Image], ffmpeg_process, threshold: float) -> None:
        """
        Processes a batch of video frames for object detection and writes the processed frames to an FFmpeg subprocess.
        """
        # Process each frame and write to FFmpeg
        for frame in frames_batch:
            pixel_values_batch = self.preprocess_images_batch([frame], is_video=True)
            outputs = self.predict_batch(pixel_values_batch)
            image_outputs = [torch.from_numpy(output[0]).to(self.device) for output in outputs]
            processed_frame = self.drawer.draw_boxes(frame, image_outputs, threshold)

            # Convert processed PIL image back to BGR format for FFmpeg
            bgr_image = cv2.cvtColor(np.array(processed_frame), cv2.COLOR_RGB2BGR)
            ffmpeg_process.stdin.write(bgr_image.tobytes())

    def process_video(
        self,
        video_path: str,
        output_path: str,
        vcodec: Optional[str] = None,
        acodec: Optional[str] = None,
        hwaccel: Optional[str] = None,
        preset: str = 'medium',
        threads: Optional[int] = None,
        fps: Optional[float] = None,
        batch_size: int = 10,
        threshold: float = 0.5,
        bitrate: str = 'auto',
        duration: Optional[float] = None,
        include_audio: bool = False
    ) -> None:
        """
        Processes a video file for object detection, applying specified codecs, hardware acceleration, and additional settings, and optionally includes original audio in the output.
        """
        if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
            raise EnvironmentError("FFmpeg or FFprobe is not available or is not installed correctly.")
        
        # Define default video codecs for various formats
        default_video_codecs = {
            '.mp4': 'libx264',
            '.avi': 'mpeg4',
            '.mov': 'libx264',
            '.mkv': 'libx264',
            '.webm': 'libvpx',
            '.flv': 'flv',
            '.wmv': 'wmv2',
            '.m4v': 'libx264',
            '.mpg': 'mpeg2video',
            '.mpeg': 'mpeg2video',
            '.3gp': 'libx264',
            '.vob': 'mpeg2video',
        }

        output_ext = pathlib.Path(output_path).suffix.lower()
        extra_ffmpeg_options = []
        vcodec = vcodec or default_video_codecs.get(output_ext, 'libx264')

        # Extract original audio codec if include_audio is True and acodec is not specified
        original_acodec = None
        if include_audio and acodec is None:
            probe_cmd = ['ffprobe', '-loglevel', 'error', '-select_streams', 'a:0', '-show_entries', 'stream=codec_name', '-of', 'default=noprint_wrappers=1:nokey=1', video_path]
            result = subprocess.run(probe_cmd, stdout=subprocess.PIPE, text=True)
            original_acodec = result.stdout.strip()

        # Adjust options based on the video codec
        if vcodec in ['libx264', 'libx265']:
            extra_ffmpeg_options += ['-preset', preset]
        elif vcodec == 'libvpx-vp9':
            extra_ffmpeg_options += ['-speed', '8', '-b:v', bitrate if bitrate != 'auto' else '1000K', '-row-mt', '1']

        # Custom headers setup
        headers = "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36\\r\\n" \
                "Accept: image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8\\r\\n" \
                "Accept-Language: en-US,en;q=0.9\\r\\n" \
                "Accept-Encoding: gzip, deflate, br\\r\\n" \
                "Referer: https://www.google.com/\\r\\n" \
                "Connection: keep-alive\\r\\n"

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Error opening video file {video_path}")
        
        if fps is None:
            fps = cap.get(cv2.CAP_PROP_FPS)

        frame_limit = float('inf')
        if duration:
            frame_limit = int(fps * duration)

        ffmpeg_cmd = ['ffmpeg', '-y']
        if hwaccel:
            ffmpeg_cmd += ['-hwaccel', hwaccel]
        ffmpeg_cmd += ['-f', 'rawvideo', '-pixel_format', 'bgr24', '-video_size', f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}", '-framerate', str(fps), '-i', '-']
        ffmpeg_cmd += ['-vcodec', vcodec] + extra_ffmpeg_options
        ffmpeg_cmd += ['-headers', headers]
        if duration:
            ffmpeg_cmd += ['-t', str(duration)]
        if threads is not None:
            ffmpeg_cmd += ['-threads', str(threads)]
        if bitrate != 'auto':
            ffmpeg_cmd += ['-b:v', bitrate]
        ffmpeg_cmd += [output_path]

        ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
        
        frames_batch = []
        frame_count = 0
        while frame_count < frame_limit:
            ret, frame = cap.read()
            if ret:
                pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frames_batch.append(pil_frame)
                frame_count += 1  
                
                if len(frames_batch) == batch_size:
                    self.process_video_frames(frames_batch, ffmpeg_process, threshold)
                    frames_batch = []  # Clear the batch after processing

            else: # Process the remaining frames
                if frames_batch:
                    self.process_video_frames(frames_batch, ffmpeg_process, threshold)
                break

        cap.release()
        ffmpeg_process.stdin.close()
        ffmpeg_process.wait()

        if ffmpeg_process.returncode != 0:
            raise ValueError("FFmpeg encountered an error during processing.")

        # Merge audio back if include_audio is True
        if include_audio and original_acodec:
            original_acodec = '.ogg' if original_acodec == 'vorbis' else original_acodec
            audio_temp_file_path = tempfile.mktemp(suffix=f'.{original_acodec}')
            audio_extract_cmd = ['ffmpeg', '-y', '-i', video_path, '-vn', '-acodec', 'copy', audio_temp_file_path]
            subprocess.run(audio_extract_cmd, check=True)

            output_with_audio_path = tempfile.mktemp(suffix=output_ext)
            merge_cmd = ['ffmpeg', '-y', '-i', output_path, '-i', audio_temp_file_path, '-c', 'copy', '-map', '0:v:0', '-map', '1:a:0', '-shortest', output_with_audio_path]
            subprocess.run(merge_cmd, check=True)
            shutil.move(output_with_audio_path, output_path)  # Replace the processed video without audio with the new one with audio
            os.remove(audio_temp_file_path)  # Clean up temporary audio file

class FileHandler:
    def __init__(self, visualizer: Any, output_dir: Union[str, pathlib.Path], filename_prefix: str) -> None:
        self.visualizer = visualizer
        self.output_dir = pathlib.Path(output_dir).resolve()
        self.filename_prefix = filename_prefix if filename_prefix else "out"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process_input(self, input_path: Union[str, pathlib.Path], **kwargs: Dict[str, Any]) -> None:
        start_time = time.time()
        paths = self._resolve_paths(input_path)
        self._process_batch(paths, **kwargs)
        print(f"Total processing time: {time.time() - start_time:.2f} seconds")

    def _resolve_paths(self, input_path: Union[str, pathlib.Path]) -> List[str]:
        if os.path.isdir(input_path):
            return [os.path.join(input_path, file_name) for file_name in os.listdir(input_path) if get_media_type(file_name) in ['image', 'video', 'gif']]
        elif os.path.isfile(input_path) and input_path.endswith('.txt'):
            with open(input_path, 'r', encoding='utf8') as file:
                return [line.strip() for line in file if line.strip()]
        # For direct remote files
        elif get_media_type(input_path) in ['image', 'video', 'gif']:
            return [input_path]
        else:
            raise ValueError("Unsupported input format.")

    def _load_image(self, path_or_url: str, threshold: float, batch_size: int) -> Union[List[Image.Image], None]:
        media_type = get_media_type(path_or_url)
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36',
            'Accept': 'image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Referer': 'https://www.google.com/',
            'Connection': 'keep-alive'
        }
        if path_or_url.startswith('http'):
            # Handle remote files (both GIFs and images) by loading them into BytesIO
            with requests.get(path_or_url, stream=True, headers=headers) as r:
                r.raise_for_status()
                img_data = BytesIO(r.content)

            if media_type == 'gif':
                # For GIFs, use the BytesIO object directly
                return self.visualizer.load_gif(img_data, path_or_url, self.filename_prefix, self.output_dir, threshold, batch_size)
            else:
                # For other image types, load the image from BytesIO and convert to RGB
                img = Image.open(img_data).convert("RGB")
                return [img]
        elif media_type in ['image', 'gif']:
            # For local images and GIFs, load them directly from the file path
            if media_type == 'gif':
                return self.visualizer.load_gif(path_or_url, path_or_url, self.filename_prefix, self.output_dir, threshold, batch_size)
            else:
                return [Image.open(path_or_url).convert("RGB")]
        return None

    def _process_batch(self, paths: List[str], **kwargs: Dict[str, Any]) -> None:
        images_to_process, paths_for_batch = [], []
        batch_size = kwargs.get('batch_size', 10)

        # Separate paths by media type for targeted processing
        image_paths = [path for path in paths if get_media_type(path) == 'image']
        gif_paths = [path for path in paths if get_media_type(path) == 'gif']
        video_paths = [path for path in paths if get_media_type(path) == 'video']

        # Process standard images in parallel first
        with ThreadPoolExecutor() as executor:
            future_to_path_image = {executor.submit(self._load_image, path, kwargs.get('threshold', 0.9), batch_size): path for path in image_paths}
            
            for future in as_completed(future_to_path_image):
                path = future_to_path_image[future]
                try:
                    result = future.result()
                    if result:
                        images_to_process.extend(result)
                        paths_for_batch.append(path)
                        if len(images_to_process) >= batch_size:
                            self._process_images(images_to_process, paths_for_batch, **kwargs)
                            images_to_process, paths_for_batch = [], []
                except Exception as e:
                    print(f"Error loading media from {path}: {e}")

        # Process remaining images if any
        if images_to_process:
            self._process_images(images_to_process, paths_for_batch, **kwargs)

        # Process GIFs sequentially
        for path in gif_paths:
            try:
                result = self._load_image(path, kwargs.get('threshold', 0.9), batch_size)
                if result:
                    self._process_images(result, [path], **kwargs)
            except Exception as e:
                print(f"Error processing GIF from {path}: {e}")

        # Process videos sequentially
        for path in video_paths:
            try:
                output_path = generate_media_output_path(self.output_dir, path, self.filename_prefix, 'video')
                self.visualizer.process_video(path, output_path, **kwargs)
            except Exception as e:
                print(f"Error processing video from {path}: {e}")

    def _process_images(self, images: List[Image.Image], paths: List[str], batch_size: int, **kwargs: Dict[str, Any]) -> None:
        self.visualizer.process_media_batch(images, self.output_dir, self.filename_prefix, paths, threshold=kwargs.get('threshold', 0.9), batch_size=batch_size, is_video=False)

def main():
    parser = argparse.ArgumentParser(description="Process local or remote media with an ONNX model.")
    parser.add_argument("--input", required=True, help="URL, path, text file, or folder.")
    parser.add_argument("--output", required=True, help="Output directory.")
    parser.add_argument("--labels", nargs='*', help="Specific object labels to draw (draws all if omitted).")
    parser.add_argument("--filename", default="out", help="Output filename prefix.")
    parser.add_argument("--vcodec", help="Video codec (defaults based on output format).")
    parser.add_argument("--acodec", help="Audio codec (defaults based on output format).")
    parser.add_argument("--include_audio", action='store_true', help="Include original audio (slows processing).")
    parser.add_argument("--duration", type=float, help="Video duration to process (seconds).")
    parser.add_argument("--fps", type=int, help="Video FPS (defaults to source FPS).")
    parser.add_argument("--hwaccel", choices=['cuda', 'dxva2', 'qsv', 'd3d11va', 'opencl', 'vulkan'], help="Hardware acceleration.")
    parser.add_argument("--preset", default='ultrafast', help="FFmpeg encoding preset.")
    parser.add_argument("--bitrate", default='auto', help="Output video bitrate ('auto' for FFmpeg default).")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size for processing.")
    parser.add_argument("--threads", type=int, help="FFmpeg thread count (0 for auto).")
    parser.add_argument("--threshold", type=float, default=0.9, help="Object detection sensitivity.")
    parser.add_argument("--model", default="hustvl/yolos-tiny", help="Hugging Face model repository. Defaults to 'hustvl/yolos-tiny'.")
    parser.add_argument("--unquantized", action='store_true', help="Use the unquantized ONNX model if present (more accurate but slower).")
    args = parser.parse_args()

    # Determine the directory of the current script    
    script_dir = pathlib.Path(__file__).parent
    repository_name, model_name = args.model.split('/')
    model_folder_path = script_dir / "models"
    model_base_path = model_folder_path / repository_name / model_name
    
    onnx_path = model_base_path / 'onnx'
    quantized_model_path = onnx_path / "model_quantized.onnx"
    unquantized_model_path = onnx_path / "model.onnx"

    if args.unquantized:
        model_path = unquantized_model_path
    else:
        model_path = quantized_model_path

    # Check if the specified ONNX file exists or fallback is necessary
    if args.unquantized and not unquantized_model_path.exists():
        print("No suitable unquantized ONNX model file found. Attempting to download the model...")
        convert_and_quantize_model(args.model, model_folder_path, quantize=True, task='auto', device='cpu')
        if not model_path.exists():
            print("Failed to download the model.")
            exit(1)
    elif not args.unquantized:
        if not quantized_model_path.exists():
            print("Quantized model not found, switching to unquantized ONNX model...")
            if not unquantized_model_path.exists():
                print("No suitable ONNX model file found. Attempting to download and quantize...")
                convert_and_quantize_model(args.model, model_folder_path, quantize=True, task='auto', device='cpu')
            model_path = quantized_model_path if quantized_model_path.exists() else unquantized_model_path
        if not model_path.exists():
            print("Failed to download or quantize model.")
            exit(1)

    # Load the configuration to access id2label
    config_path = model_base_path / "config.json"
    try:
        with open(model_base_path / "config.json", 'r', encoding='utf8') as f:
            config = json.load(f)
    except FileNotFoundError:
        print("Error: Configuration file not found.")
        exit(1)
    except json.JSONDecodeError:
        print("Error: Configuration file is not valid JSON.")
        exit(1)

    # Before initializing the Visualizer, validate the labels if provided
    if args.labels:
        valid_labels = set(config["id2label"].values())
        invalid_labels = set(args.labels) - valid_labels
        if invalid_labels:
            print(f"Invalid label(s) specified: {', '.join(invalid_labels)}")
            print("Valid labels are:")
            print(', '.join(sorted(valid_labels)))
            exit(1)  # Exit if there are invalid labels

    # Initialize Visualizer and FileHandler
    visualizer = Visualizer(config_path, model_path, model_base_path, selected_labels=args.labels)
    file_handler = FileHandler(visualizer, args.output, args.filename)
    
    try:
        file_handler.process_input(
            args.input,
            vcodec=args.vcodec,
            acodec=args.acodec,
            hwaccel=args.hwaccel,
            preset=args.preset,
            threads=args.threads,
            fps=args.fps,
            batch_size=args.batch_size,
            threshold=args.threshold,
            bitrate=args.bitrate,
            duration=args.duration,
            include_audio=args.include_audio
        )
    except Exception as e:
        print(f"Error during file processing: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()

import cv2
import os
from PIL import Image
from tqdm import tqdm
import shutil

class VideoConverter:
    def __init__(self):
        pass
    
    def clean_temp_folders(self):
        """Remove temporary frame folders after processing"""
        folders_to_clean = ["frames", "temp_frames"]
        
        for folder in folders_to_clean:
            if os.path.exists(folder):
                try:
                    shutil.rmtree(folder)
                    print(f"Successfully deleted {folder} folder")
                except Exception as e:
                    print(f"Error deleting {folder}: {e}")
    
    def extract_frames(self, video_path, output_folder="frames", fps=None, resize=None):
        """
        Extract frames from a video file.
        
        Args:
            video_path: Path to the video file
            output_folder: Folder to save extracted frames
            fps: If specified, extract frames at this rate (otherwise extract all frames)
            resize: Optional tuple (width, height) to resize frames
        
        Returns:
            Number of frames extracted
        """
        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
        # Open the video file
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        original_fps = video.get(cv2.CAP_PROP_FPS)
        
        # Check for rotation metadata
        rotation = int(video.get(cv2.CAP_PROP_ORIENTATION_META))
        
        # Determine frame extraction rate
        if fps is None:
            frame_interval = 1
        else:
            frame_interval = max(1, round(original_fps / fps))
        
        # Extract frames
        count = 0
        frame_number = 0
        
        print(f"Extracting frames from {video_path} (total: {total_frames}, FPS: {original_fps}, rotation: {rotation}°)")
        
        with tqdm(total=total_frames) as pbar:
            while True:
                ret, frame = video.read()
                if not ret:
                    break
                    
                if frame_number % frame_interval == 0:
                    # Apply rotation if needed
                    if rotation == 90:
                        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                    elif rotation == 180:
                        frame = cv2.rotate(frame, cv2.ROTATE_180)
                    elif rotation == 270:
                        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    
                    # Resize if needed
                    if resize:
                        frame = cv2.resize(frame, resize)
                    
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Save the frame
                    img = Image.fromarray(frame_rgb)
                    img.save(f"{output_folder}/frame_{count:05d}.png")
                    count += 1
                
                frame_number += 1
                pbar.update(1)
        
        video.release()
        print(f"Extracted {count} frames to {output_folder}")
        return count


    def frames_to_video(self, frames_folder, output_path, fps=24, frame_pattern="frame_%05d.png", codec="mp4v"):
        """
        Create a video from a sequence of frames.
        
        Args:
            frames_folder: Folder containing the frame images
            output_path: Path to save the output video
            fps: Frames per second for the output video
            frame_pattern: Pattern for frame filenames
            codec: FourCC codec code
        
        Returns:
            Path to the created video file
        """
        # List all frames
        frames = []
        for filename in os.listdir(frames_folder):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                frames.append(os.path.join(frames_folder, filename))
        
        if not frames:
            raise ValueError(f"No frames found in {frames_folder}")
        
        # Sort frames by frame number using numeric extraction from filenames
        def extract_number(filename):
            # Extract numbers from the filename using the expected pattern
            try:
                # This assumes the format is like "frame_00001.png"
                return int(''.join(filter(str.isdigit, os.path.basename(filename))))
            except:
                return 0
        
        # Sort frames by extracted number
        frames.sort(key=extract_number)
        
        # Get dimensions from the first frame
        sample = cv2.imread(frames[0])
        h, w, _ = sample.shape
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*codec)
        video = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        
        if not video.isOpened():
            raise ValueError("Could not create video writer")
        
        # Add each frame to the video
        print(f"Creating video from {len(frames)} frames at {fps} FPS")
        for frame_path in tqdm(frames):
            frame = cv2.imread(frame_path)
            video.write(frame)
        
        video.release()
        print(f"Video saved to {output_path}")
        return output_path

    # Resize images before processing
    def resize_for_processing(self, image, max_size=512):
        w, h = image.size
        scale = max_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        return image.resize((new_w, new_h), Image.LANCZOIN)
    
# Example usage:

    # Extract frames from a video
    # num_frames = extract_frames("input/video.mp4", "frames", fps=10)
    
    # Create video from frames
    # frames_to_video("processed_frames", "output/processed_video.mp4", fps=24)
    
    # Process a video through Stable Diffusion
    # 1. Extract frames
    # 2. Process each frame with your StableDiffusionImg2ImgPipeline
    # 3. Compile processed frames back into a video

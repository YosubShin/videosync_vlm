import cv2  # We're using OpenCV to read video, to install !pip install opencv-python
import base64
from openai import OpenAI
import os
import numpy as np

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY",
                "<your OpenAI API key if not set as env var>"))


def extract_frames(video_path):
    """Extracts frames from a video file.
    Returns a list of base64 encoded frames."""
    video = cv2.VideoCapture(video_path)
    frames = []
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        frames.append(base64.b64encode(buffer).decode("utf-8"))
    video.release()
    return frames


def create_frame_visualization(frames1, frames2):
    """Creates a visualization of two sets of frames side by side.
    Returns a single image with two rows of frames."""
    # Decode base64 frames
    def decode_frame(b64_frame):
        img_bytes = base64.b64decode(b64_frame)
        img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
        return cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

    # Decode all frames
    decoded_frames1 = [decode_frame(f) for f in frames1]
    decoded_frames2 = [decode_frame(f) for f in frames2]

    # Get dimensions
    h, w, _ = decoded_frames1[0].shape
    n_frames = len(frames1)  # Assuming both have same length after sampling

    # Create a large canvas
    canvas = np.zeros((2 * h, n_frames * w, 3), dtype=np.uint8)

    # Place frames in the canvas
    for i, (f1, f2) in enumerate(zip(decoded_frames1, decoded_frames2)):
        canvas[0:h, i*w:(i+1)*w] = f1  # Top row
        canvas[h:2*h, i*w:(i+1)*w] = f2  # Bottom row

    return canvas


base_path = "/Users/yosub/Downloads/ntu_videos/"
# Extract frames from both videos
frames1 = extract_frames(base_path + "S016C001P019R002A011.mp4")
frames2 = extract_frames(base_path + "S016C002P019R002A011.mp4")

print(f"Extracted {len(frames1)} frames from video 1")
print(f"Extracted {len(frames2)} frames from video 2")

# Sample frames to reduce the number (taking every 3rd frame, up to 20 frames)
frames1 = frames1[::4][:15]
frames2 = frames2[::4][:15]

print(f"Frames1: {len(frames1)}")
print(f"Frames2: {len(frames2)}")

# After sampling frames, add visualization
visualization = create_frame_visualization(frames1, frames2)
cv2.imwrite('frame_comparison.jpg', visualization)
print("Visualization saved as 'frame_comparison.jpg'")

PROMPT_MESSAGES = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": f"Here are the frames from video 1 with length {len(frames1)}",
            },
            *map(lambda x: {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{x}",
                },
            }, frames1),
            {
                "type": "text",
                "text": f"Here are the frames from video 2 with length {len(frames2)}",
            },
            *map(lambda x: {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{x}",
                },
            }, frames2),
            {
                "type": "text",
                "text": f"""Following are two sequences of frames:
                First sequence: {len(frames1)} frames from video 1
                Second sequence: {len(frames2)} frames from video 2
                These are from different camera angles of the same event,
                with an unknown offset of frames in range [-30, 30].
                Return the number of frames video 1 is offset from video 2.
                If video 1's frame 7 is equal to video 2's frame 4, then the offset is 3 frames. 
                If video 1's frame 5 is equal to video 2's frame 6, then the offset is -1 frames.""",
            },
        ],
    },
]

params = {
    # "model": "gpt-4o",
    "model": "o1-mini",
    "messages": PROMPT_MESSAGES,
    # "max_tokens": 200,
}

result = client.chat.completions.create(**params)
print(result.choices[0].message.content)

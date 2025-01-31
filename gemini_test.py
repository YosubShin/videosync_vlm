import os
import time
import cv2
import PIL.Image
import google.generativeai as genai

genai.configure(api_key=os.environ["GEMINI_API_KEY"])


def extract_frames(video_path):
    """Extracts frames from a video file.
    Returns a list of PIL Images."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Convert to PIL Image
        pil_image = PIL.Image.fromarray(frame_rgb)
        frames.append(pil_image)
    cap.release()
    return frames


def upload_to_gemini(path, mime_type=None):
    """Uploads the given file to Gemini.

    See https://ai.google.dev/gemini-api/docs/prompting_with_media
    """
    file = genai.upload_file(path, mime_type=mime_type)
    print(f"Uploaded file '{file.display_name}' as: {file.uri}")
    return file


def wait_for_files_active(files):
    """Waits for the given files to be active.

    Some files uploaded to the Gemini API need to be processed before they can be
    used as prompt inputs. The status can be seen by querying the file's "state"
    field.

    This implementation uses a simple blocking polling loop. Production code
    should probably employ a more sophisticated approach.
    """
    print("Waiting for file processing...")
    for name in (file.name for file in files):
        file = genai.get_file(name)
        while file.state.name == "PROCESSING":
            print(".", end="", flush=True)
            time.sleep(10)
            file = genai.get_file(name)
        if file.state.name != "ACTIVE":
            raise Exception(f"File {file.name} failed to process")
    print("...all files ready")
    print()


base_path = "/Users/yosub/Downloads/ntu_videos/"
# Replace the file upload section with frame extraction
frames1 = extract_frames(base_path + "S016C001P019R002A011.mp4")
frames2 = extract_frames(base_path + "S016C002P019R002A011.mp4")

print(f"Extracted {len(frames1)} frames from video 1")
print(f"Extracted {len(frames2)} frames from video 2")

# Create the model with the same config as before
generation_config = {
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 65536,
    "response_mime_type": "text/plain",
}

model_name = "gemini-2.0-flash-thinking-exp-01-21"
# model_name = "gemini-2.0-flash-exp"

model = genai.GenerativeModel(
    model_name=model_name,
    generation_config=generation_config,
)

frames1 = frames1[::3][:20]
frames2 = frames2[::3][:20]

print(f"Frames1: {len(frames1)}")
print(f"Frames2: {len(frames2)}")

response = model.generate_content([
    f"""Following are two sequences of frames:
    First sequence: {len(frames1)} frames from video 1
    Second sequence: {len(frames2)} frames from video 2
    These are from different camera angles of the same event,
    with an unknown offset of frames in range [-30, 30].
    Return the number of frames video 1 is offset from video 2.
    If video 1's frame 7 is equal to video 2's frame 4, then the offset is 3 frames. 
    If video 1's frame 5 is equal to video 2's frame 6, then the offset is -1 frames.""",
    "Here are the frames from video 1 with length " + str(len(frames1)),
    *frames1,
    "Here are the frames from video 2 with length " + str(len(frames2)),
    *frames2,
])

print(response.text)

# Start chat with frames instead of videos
# Note: You might want to select specific frames rather than sending all
# as there might be limits on how many images you can send at once
# chat_session = model.start_chat(
#     history=[
#         {
#             "role": "model",
#             "parts": ["Here are the frames from video 1 with length " + str(len(frames1))],  # separator
#         },
#         {
#             "role": "user",
#             "parts": [*frames1],  # first video frames
#         },
#         {
#             "role": "model",
#             "parts": ["Here are the frames from video 2 with length " + str(len(frames2))],  # separator
#         },
#         {
#             "role": "user",
#             "parts": [*frames2],  # second video frames
#         },
#         {
#             "role": "user",
#             "parts": [
#                 f"""I've sent you two sequences of frames:
#                 First sequence: {len(frames1)} frames from video 1
#                 Second sequence: {len(frames2)} frames from video 2
#                 These are from different camera angles of the same event,
#                 with an unknown offset of frames in range [-30, 30].
#                 Return the number of frames video 1 is offset from video 2.
#                 If video 1 is 3 frames after video 2, then the offset is 3 frames.
#                 If video 1 is 2 frames before video 2, then the offset is -2 frames.""",
#             ],
#         },
#     ]
# )

# response = chat_session.send_message("INSERT_INPUT_HERE")
# print(response.text)

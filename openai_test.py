from typing import Dict, Any
import logging
from tqdm import tqdm
import json
from datetime import datetime
import sys
from pathlib import Path
import pandas as pd
import random
import pickle
from pydantic import BaseModel
import numpy as np
import os
from openai import OpenAI
import base64
import cv2

# fmt: off
sys.path.append('../scripts')
from calculate_margin_of_error import calculate_margin_of_error
# fmt: on


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


class SyncOffset(BaseModel):
    offset: int


def process_video_pair(video_pair, base_path, model_name, downsample_factor, num_frames):
    """Process a single video pair and return the offset prediction"""
    try:
        # Extract frames from both videos
        frames1 = extract_frames(
            str(base_path / video_pair['video_file_0'].split('/')[-1]))
        frames2 = extract_frames(
            str(base_path / video_pair['video_file_1'].split('/')[-1]))

        # Sample frames
        frames1 = frames1[::downsample_factor][:num_frames]
        frames2 = frames2[::downsample_factor][:num_frames]

        PROMPT_MESSAGES = [
            {
                "role": "system",
                "content": "You must respond with a single integer representing the frame offset between the two videos."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""You are given two sequences of frames:
- First sequence: {len(frames1)} frames from video 1
- Second sequence: {len(frames2)} frames from video 2

Both sequences capture the same event from different camera angles, with an unknown temporal offset in the range [-30, 30] frames.

Your task is to determine the offset of video 1 relative to video 2.
- A positive offset (N) means video 1 is ahead of video 2 by N frames.
- A negative offset (-N) means video 1 lags behind video 2 by N frames.

Example cases:
- If frame 7 of video 1 matches frame 4 of video 2, the offset is +3 (7 - 4 = 3).
- If frame 5 of video 1 matches frame 6 of video 2, the offset is -1 (5 - 6 = -1).

Return the computed frame offset as a single integer.""",
                    },
                    {
                        "type": "text",
                        "text": f"Frames from video 1:",
                    },
                    *map(lambda x: {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{x}",
                        },
                    }, frames1),
                    {
                        "type": "text",
                        "text": f"Frames from video 2:",
                    },
                    *map(lambda x: {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{x}",
                        },
                    }, frames2),
                ],
            },
        ]

        params = {
            "model": model_name,
            "messages": PROMPT_MESSAGES,
            "response_format": SyncOffset,
        }

        result = client.beta.chat.completions.parse(**params)

        return {
            'offset': result.choices[0].message.parsed.offset,
            'error': False,
            'error_msg': None,
            'tokens_used': result.usage.total_tokens
        }
    except Exception as e:
        return {
            'offset': None,
            'error': True,
            'error_msg': str(e),
            'tokens_used': 0
        }


def setup_logging(timestamp: str) -> None:
    """Setup logging configuration"""
    log_filename = f'sync_analysis_{timestamp}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()  # This will also print to console
        ]
    )


def save_config(config: Dict[str, Any]) -> None:
    """Save configuration to a JSON file"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    config_file = f'config_{timestamp}.json'
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=4, default=str)
    logging.info(f"Configuration saved to {config_file}")


def main(start_idx=5, end_idx=50):
    # Setup logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    setup_logging(timestamp)

    # Load validation data
    val_path = Path('~/co/datasets/ntu/val.pkl').expanduser()
    base_path = val_path.parent / 'processed_videos'

    # Save configuration
    config = {
        'start_idx': start_idx,
        'end_idx': end_idx,
        'val_path': str(val_path),
        'base_path': str(base_path),
        'timestamp': timestamp,
        'model_name': 'gpt-4o-mini',
        'downsample_factor': 4,
        'num_frames': 30
    }
    save_config(config)
    logging.info(f"Configuration:\n{json.dumps(config, indent=4)}")
    logging.info("Starting video processing")

    with open(val_path, 'rb') as f:
        val_data = pickle.load(f)

    # Set random seed and shuffle data
    random.seed(42)
    random.shuffle(val_data)

    # Select subset of data
    val_data = val_data[start_idx:end_idx]

    # Process each video pair
    results = []
    total_tokens = 0

    for video_pair in tqdm(val_data):
        result = process_video_pair(
            video_pair,
            base_path,
            model_name=config['model_name'],
            downsample_factor=config['downsample_factor'],
            num_frames=config['num_frames']
        )
        total_tokens += result['tokens_used']

        # Log results
        log_entry = {
            'video_pair': video_pair,
            'result': result
        }
        logging.info(f"Processed video pair: {json.dumps(log_entry)}")

        # Add to results
        results.append({
            **video_pair,
            'predicted_offset': result['offset'],
            'error_occurred': result['error'],
            'error_message': result['error_msg']
        })

    # Save results to CSV
    results_file = f'sync_results_{timestamp}.csv'
    df = pd.DataFrame(results)
    df.to_csv(results_file, index=False)
    logging.info(f"Results saved to {results_file}")

    # Calculate error metrics for successful predictions
    successful_preds = df[~df['error_occurred']]
    if len(successful_preds) > 0:
        # Multiply predicted offset by downsample factor before calculating error
        abs_errors = abs((successful_preds['predicted_offset'] * config['downsample_factor']) -
                         (successful_preds['label_0'] - successful_preds['label_1']))

        mean_error = abs_errors.mean()
        margin_error = calculate_margin_of_error(abs_errors.tolist())

        logging.info("\nResults Summary:")
        logging.info(f"Total samples processed: {len(df)}")
        logging.info(f"Successful predictions: {len(successful_preds)}")
        logging.info(
            f"Mean absolute error: {mean_error:.2f} ± {margin_error:.2f} frames")
        logging.info(f"Total tokens used: {total_tokens}")


if __name__ == "__main__":
    main()

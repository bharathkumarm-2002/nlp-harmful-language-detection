import os
import cv2

input_dir = "video_data"
output_dir = "datasets/video_frames"
os.makedirs(output_dir, exist_ok=True)

for label in ["toxic", "non_toxic"]:
    input_path = os.path.join(input_dir, label)
    output_path = os.path.join(output_dir, label)
    os.makedirs(output_path, exist_ok=True)

    for video_file in os.listdir(input_path):
        video_path = os.path.join(input_path, video_file)
        cap = cv2.VideoCapture(video_path)
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % 10 == 0:
                frame_filename = f"{os.path.splitext(video_file)[0]}_f{frame_count}.jpg"
                frame_output = os.path.join(output_path, frame_filename)
                cv2.imwrite(frame_output, frame)
            frame_count += 1

        cap.release()
        print(f"✔️ Extracted frames from: {video_file}")

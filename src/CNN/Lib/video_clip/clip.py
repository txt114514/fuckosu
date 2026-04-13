import cv2

video_path = "input.mp4"
out_path = "cropped.mp4"

left, top, right, bottom = 101, 199, 1848, 1197
width = right - left
height = bottom - top

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    crop = frame[top:bottom, left:right]
    out.write(crop)

cap.release()
out.release()
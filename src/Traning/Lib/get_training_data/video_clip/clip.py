import cv2


class VideoCropper:
    def __init__(self, video_path, out_path, left, top, right, bottom):
        self.video_path = video_path
        self.out_path = out_path
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom
    def intrface_crop(self, frame, out):
        '''显示裁剪界面'''
        cv2.rectangle(frame, (self.left, self.top), (self.right, self.bottom), (0, 255, 0), 2)
        cv2.imshow("Crop Interface", frame)
        cv2.waitKey(1)  # 等待1毫秒以更新界面
    def crop_video(self):
        width = self.right - self.left
        height = self.bottom - self.top

        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(self.out_path, fourcc, fps, (width, height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            deskop = frame[self.top:self.bottom, self.left:self.right]
            self.interface_crop(frame, out)
            
            crop = frame[self.top:self.bottom, self.left:self.right]
            out.write(crop)

        cap.release()
        out.release()
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
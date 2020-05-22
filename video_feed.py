import cv2
from imutils.video import FPS
import imutils


class VideoFeed:

    def __init__(self, url, width=450, bw=True) -> None:
        super().__init__()
        self.url = url
        self.feed = cv2.VideoCapture(url)
        self.fps = FPS().start()
        self.width = width
        self.bw = bw
        print("Is local video: " + str(self.local_video()))

    def next_frame(self):
        (grabbed, frame) = self.feed.read()
        if not grabbed:
            return None
        frame = imutils.resize(frame, width=self.width)
        if self.bw:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.fps.update()
        return frame

    def close(self):
        self.fps.stop()
        self.feed.release()
        cv2.destroyAllWindows()
        print("fps: " + str(self.fps.fps()))

    def local_video(self):
        return "rtsp://" not in self.url
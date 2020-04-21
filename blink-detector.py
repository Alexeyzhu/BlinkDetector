# import the necessary packages
import argparse
import datetime
import time
from threading import Thread

import cv2
import dlib
import imutils
from imutils import face_utils
from imutils.video import FPS
from scipy.spatial import distance as dist


# The main goal of this class is to speed up reading frames
# from web camera
class WebcamStream:
    def __init__(self, src=0):
        # initialize the video camera stream and read the first frame from the stream
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        # initialize the variable used to indicate if the thread should be stopped
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                self.stream.release()
                return

            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # return the frame most recently read
        return self.grabbed, self.frame

    def release(self):
        # indicate that the thread should be stopped
        self.stopped = True

    def get(self, param):
        return self.stream.get(param)


def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])
    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    # return the eye aspect ratio
    return ear


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
                help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="",
                help="path to input video file")
args = vars(ap.parse_args())

# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 3
ALLOWED_EYE_CLOSED_TIME = 2  # sec

COUNTER = 0  # total number of successive frames that have an eye aspect ratio less than EYE_AR_THRESH
TOTAL = 0  # total number of blinks that have taken place while the script has been running
TIME_STARTED_CLOSED = 0
ALREADY_CLOSED = False
ALERT = False

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

if args["video"] != "":
    print("[INFO] starting video stream...")
    fileStream = True
    vs = cv2.VideoCapture(args["video"])
    fps = vs.get(cv2.CAP_PROP_FPS)
    print(f'[INFO] video FPS: {fps}')
    duration = int(vs.get(cv2.CAP_PROP_FRAME_COUNT)) / fps
    print(f'[INFO] duration: {duration}')
else:
    print("[INFO] starting web camera stream...")
    fileStream = False
    vs = WebcamStream().start()
    fps = vs.get(cv2.CAP_PROP_FPS)
    print(f'[INFO] camera FPS: {fps}')

time.sleep(1.0)
estimated_fps = FPS().start()
frame_count = 0

# loop over frames from the video stream
while True:
    ret, frame = vs.read()

    # if this is a file video stream, then we need to check if
    # there any more frames left in the buffer to process
    if fileStream and not ret:
        break

    # grab the frame from the threaded video file stream, resize
    # it, and convert it to grayscale
    # channels)
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale frame
    rects = detector(gray, 0)
    # loop over the face detections
    for rect in rects:
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        # extract the left and right eye coordinates, then use the
        # coordinates to compute the eye aspect ratio for both eyes
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        # average the eye aspect ratio together for both eyes
        ear = (leftEAR + rightEAR) / 2.0
        # compute the convex hull for the left and right eye, then
        # visualize each of the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # check to see if the eye aspect ratio is below the blink
        # threshold, and if so, increment the blink frame counter
        if ear < EYE_AR_THRESH:
            COUNTER += 1

            # if precess a video, then compute time till alert
            # using FPS of the video
            # If process a stream from webcam, then use a timer
            if fileStream:
                if ALREADY_CLOSED:
                    if (frame_count - TIME_STARTED_CLOSED) / fps > ALLOWED_EYE_CLOSED_TIME:
                        ALERT = True
                else:
                    ALREADY_CLOSED = True
                    TIME_STARTED_CLOSED = frame_count
            else:
                if ALREADY_CLOSED:
                    now = datetime.datetime.now().timestamp()
                    if now - TIME_STARTED_CLOSED > ALLOWED_EYE_CLOSED_TIME:
                        ALERT = True
                else:
                    ALREADY_CLOSED = True
                    TIME_STARTED_CLOSED = datetime.datetime.now().timestamp()

        # otherwise, the eye aspect ratio is not below the blink
        # threshold
        else:
            # if the eyes were closed for a sufficient number of
            # then increment the total number of blinks
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                TOTAL += 1
            # reset the eye frame counter
            COUNTER = 0
            ALREADY_CLOSED = False
            ALERT = False

        # draw the total number of blinks on the frame along with
        # the computed eye aspect ratio for the frame
        cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if ALERT:
            cv2.putText(frame, "ALERT!", (70, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)

    # show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

    frame_count += 1
    estimated_fps.update()

estimated_fps.stop()
print("[INFO] elasped time: {:.2f}".format(estimated_fps.elapsed()))
print("[INFO] Approx. processing FPS: {:.2f}".format(estimated_fps.fps()))

# do a bit of cleanup
vs.release()
cv2.destroyAllWindows()

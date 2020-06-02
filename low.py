from video_feed import VideoFeed
import cv2
import numpy as np
from object_detector import ObjectDetector
import time
from stream_credentials import get_authenticated_stream_url
import sys
import os
import logging

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', filename='out.log',level=logging.DEBUG)
logging.debug('Starting app..')

url = "data/videos/v0_fast.mp4"
# url = "data/videos/phone.mov"
url = "data/videos/phone_fast.mp4"
# url = "data/videos/v0.mp4"
# url = get_authenticated_stream_url()
WIDTH = 700
feed = VideoFeed(url, width=WIDTH, bw=False)
detector = ObjectDetector()

COLORS = {}
COLORS["cat"] = np.random.uniform(0, 255, size=(3))
COLORS["waterbowl"] = np.random.uniform(0, 255, size=(3))
COLORS["foodbowl"] = np.random.uniform(0, 255, size=(3))


def draw_bounding_box(img, label, confidence, x, y, x_plus_w, y_plus_h):
    color = [0, 0, 0]
    color[1] = confidence*255 # green
    color[2] = (1 - confidence) * 255 # red
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    # cv2.putText(img, str(confidence), (x - 10, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def print_boxes(frame, result):
    for type in result.keys():
        for b in result[type]:
            draw_bounding_box(frame,
                              type,
                              b["confidence"],
                              round(b["x"]),
                              round(b["y"]),
                              round(b["x"] + b["w"]),
                              round(b["y"] + b["h"])
            )
    cv2.imshow("low", frame)


def overlap(bowls, cats):
    # any overlap between cat and bowl
    for bowl in bowls:
        for cat in cats:
            bowl_top_left_x = bowl["x"]
            bowl_top_left_y = bowl["y"]
            bowl_bottom_right_x = bowl["x"] + bowl["w"]
            bowl_bottom_right_y = bowl["y"] + bowl["h"]

            cat_top_left_x = cat["x"]
            cat_top_left_y = cat["y"]
            cat_bottom_right_x = cat["x"] + cat["w"]
            cat_bottom_right_y = cat["y"] + cat["h"]

            if bowl_top_left_x >= cat_bottom_right_x or cat_top_left_x >= bowl_bottom_right_x:
                continue
            if bowl_top_left_y >= cat_bottom_right_y or cat_top_left_y >= bowl_bottom_right_y:
                continue
            return True
    return False

show_video = True

no_foodbowl_wait = 0
no_cat_wait = 0
no_cat_session_end = 10

eating_detection_time = 5 # if one of the eating indicators been in place for this time, we consider eating
food_bowl_missing_time = 2 # if food bowl is missing for this time, we consider eating
wait_till_session_end_sleep_time = 5

# Session tracking
session = False
session_start_time = sys.maxsize

# Session info tracking
ate_food = False
cat_food_overlap_start_time = sys.maxsize
food_bowl_missing_start_time = sys.maxsize
cat_last_seen = sys.maxsize
eating_images = []

while True:

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    start = time.time()
    frame = feed.next_frame()
    # print("Reading: " + str(time.time() - start))
    if frame is None:
        break
    start = time.time()
    result = detector.get_objects(frame)
    # print("Detection: " + str(time.time() - start))
    if show_video:
        print_boxes(frame, result)
        # continue
    '''
    * If no active session, then check if there's a food bowl. If no food bowl, sleep
    * If no active session, and there is a foodbowl and cat, start session
    '''
    if not session:
        if "foodbowl" not in result:
            logging.debug("no foodbowl, sleeping..")
            time.sleep(no_foodbowl_wait)
            continue

        if "cat" not in result:
            # logging.debug("no cat, sleeping..")
            time.sleep(no_cat_wait)
            continue
        session = True
        session_start_time = time.time()
        eating_images += [frame]

    '''
    * If active session, and no cat, wait for a bit to handle model inconsistencies. End session after wait.
    '''
    if "cat" not in result:
        if time.time() - cat_last_seen > no_cat_session_end:
            eating_images += [frame]
            path = "runtime/" + str(session_start_time) + "/"
            os.makedirs(path)
            c = 0
            for image in eating_images:
                c += 1
                cv2.imwrite(path + str(c) + ".jpg", image)

            logging.debug("Session ended at " + str(time.time() - no_cat_session_end))
            logging.debug("Total session time: " + str(time.time() - session_start_time - no_cat_session_end))
            logging.debug("Ate food: " + str(ate_food))
            session = False
            ate_food = False
            cat_food_overlap_start_time = sys.maxsize
            food_bowl_missing_start_time = sys.maxsize
            cat_last_seen = sys.maxsize
            session_start_time = sys.maxsize
            eating_images = []
        continue

    cat_last_seen = time.time()

    if ate_food:
        # No need to check again for a session
        if feed.local_video():
            break
        logging.debug("Already eaten food, waiting for session to end")
        time.sleep(wait_till_session_end_sleep_time)
        continue
    '''
    * If active session and there is a cat, check if cat is eating. Cat is potentially eating if:
    * there is an overlap of bounding boxes
    * foodbowl is missing for over X seconds
    '''

    if "foodbowl" in result:
        food_bowl_missing_start_time = sys.maxsize
        overlapped = overlap(result["foodbowl"], result["cat"])
        if not overlapped:
            # if both cat and bowl are in frame but not overlapping then we reset potential eating check
            cat_food_overlap_start_time = sys.maxsize
        else:
            # if there is overlap, check if it's been for long enough
            cat_food_overlap_start_time = min(cat_food_overlap_start_time, time.time())
            if time.time() - cat_food_overlap_start_time > eating_detection_time:
                logging.debug("Ate food because of overlap")
                eating_images += [frame]
                ate_food = True
                continue

    if "foodbowl" not in result:
        food_bowl_missing_start_time = min(time.time(), food_bowl_missing_start_time)
        if time.time() - food_bowl_missing_start_time > eating_detection_time:
            logging.debug("Ate food because of missing bowl")
            eating_images += [frame]
            ate_food = True
            continue
feed.close()

if feed.local_video() and session_start_time < sys.maxsize:
    path = "runtime/" + str(session_start_time) + "/"
    os.makedirs(path)
    c = 0
    for image in eating_images:
        c += 1
        cv2.imwrite(path + str(c) + ".jpg", image)
    print("======== Last state ==========")
    print("Session ended at " + str(time.time()))
    print("Total session time: " + str(time.time() - session_start_time))
    print("Ate food: " + str(ate_food))



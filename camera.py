import sys
from Detector.MtcnnDetector import MtcnnDetector
from Detector.detector import Detector
from Detector.fcn_detector import FcnDetector
from Train_Model.mtcnn_model import P_Net, R_Net, O_Net
import cv2
import numpy as np

# Configure
model_path = ['MTCNN_Model/PNet/PNet-30', 'MTCNN_Model/RNet/RNet-22', 'MTCNN_Model/ONet/ONet-22']
min_face_size = 24
thresh = [0.9, 0.8, 0.9]


stride = 2
slide_window = False
shuffle = False

detectors = [None, None, None]

PNet = FcnDetector(P_Net, model_path[0])
detectors[0] = PNet
RNet = Detector(R_Net, 24, 1, model_path[1])
detectors[1] = RNet
ONet = Detector(O_Net, 48, 1, model_path[2])
detectors[2] = ONet

mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=min_face_size,
                               stride=stride, threshold=thresh, slide_window=slide_window)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

shot_photo = False
Real_time = True

img_list = []
sample_amount = 5
count = 0
if (shot_photo):
    while count < sample_amount:
        __, frame = cap.read()
        cv2.imshow("camera", frame)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            img = frame
            img_list.append(img)
            cv2.imshow("taken", img)
            cv2.waitKey(0)
            count += 1
    cap.release()
    cv2.destroyAllWindows()


corpbbox = None

while (shot_photo):
    # fps = video_capture.get(cv2.CAP_PROP_FPS)
    for frame in img_list:
        ret = True
        if ret:
            image = np.array(frame)
            boxes_c,_ = mtcnn_detector.detect(image)

            for i in range(boxes_c.shape[0]):
                bbox = boxes_c[i, :4]
                score = boxes_c[i, 4]
                corpbbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
                # if score > thresh:
                cv2.rectangle(frame, (corpbbox[0], corpbbox[1]),
                              (corpbbox[2], corpbbox[3]), (255, 0, 0), 1)
                cv2.putText(frame, '{:.3f}'.format(score), (corpbbox[0], corpbbox[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 255), 2)

            # time end
            cv2.imshow("", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                continue
        else:
            print('img not find')
            break
    break

while (Real_time):
    # fps = video_capture.get(cv2.CAP_PROP_FPS)
    t1 = cv2.getTickCount()
    ret, frame = cap.read()
    if ret:
        image = np.array(frame)
        boxes_c,_ = mtcnn_detector.detect(image)

        t2 = cv2.getTickCount()
        t = (t2 - t1) / cv2.getTickFrequency()
        fps = 1.0 / t
        for i in range(boxes_c.shape[0]):
            bbox = boxes_c[i, :4]
            score = boxes_c[i, 4]
            corpbbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
            # if score > thresh:
            cv2.rectangle(frame, (corpbbox[0], corpbbox[1]),
                          (corpbbox[2], corpbbox[3]), (255, 0, 0), 1)
            cv2.putText(frame, '{:.3f}'.format(score), (corpbbox[0], corpbbox[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 2)
        cv2.putText(frame, '{:.4f}'.format(t) + " " + '{:.3f}'.format(fps), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 0, 255), 2)

        # time end
        cv2.imshow("", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print('device not find')
        break

cap.release()
cv2.destroyAllWindows()
from boxmot import DeepOCSORT
from pathlib import Path
import cv2
import numpy as np

tracker = DeepOCSORT(
    model_weights=Path('osnet_x0_25_msmt17.pt'),  # which ReID model to use
    device='cuda:0',
    fp16=True,
)

vid = cv2.VideoCapture(0)
color = (0, 0, 255)  # BGR
thickness = 2
fontscale = 0.5

while True:
    ret, im = vid.read()

    # output has to be N X (x, y, x, y, conf, cls)
    dets = yolov7_inference(im)

    ts = tracker.update(dets, im)  # --> (x, y, x, y, id, conf, cls)

    xyxys = ts[:, 0:4].astype('int')  # float64 to int
    ids = ts[:, 4].astype('int')  # float64 to int
    confs = ts[:, 5]
    clss = ts[:, 6]

    # print bboxes with their associated id, cls and conf
    if ts.shape[0] != 0:
        for xyxy, id, conf, cls in zip(xyxys, ids, confs, clss):
            im = cv2.rectangle(
                im,
                (xyxy[0], xyxy[1]),
                (xyxy[2], xyxy[3]),
                color,
                thickness
            )
            cv2.putText(
                im,
                f'id: {id}, conf: {conf}, c: {cls}',
                (xyxy[0], xyxy[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontscale,
                color,
                thickness
            )

    # show image with bboxes, ids, classes and confidences
    cv2.imshow('frame', im)

    # break on pressing q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
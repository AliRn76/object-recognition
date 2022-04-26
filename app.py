import cv2

thres = 0.5  # threshold to detect object


cap = cv2.VideoCapture(0)
cap.set(3, 00)
cap.set(4, 00)


classNames = []
classfile = "coco.names"
with open(classfile, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

    configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
    weightsPath = "frozen_inference_graph.pb"

    net = cv2.dnn_DetectionModel(weightsPath, configPath)
    net.setInputSize(320, 320)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)

    while True:
        success, img = cap.read()
        classids, confs, bbox = net.detect(img, confThreshold=thres)

        print(classids, bbox)

        if len(classids) != 0:
            for classid, confidence, bbox in zip(classids.flatten(), confs.flatten(), bbox):
                cv2.rectangle(img, bbox, color=(0, 255, 0), thickness=3)
                cv2.putText(
                    img,
                    classNames[classid - 1].upper(),
                    (bbox[0] + 10, bbox[1] + 30),
                    cv2.FONT_HERSHEY_PLAIN,
                    2,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    img,
                    str(round(confidence * 100, 2)),
                    (bbox[0] + 200, bbox[1] + 30),
                    cv2.FONT_HERSHEY_PLAIN,
                    2,
                    (0, 255, 0),
                    2,
                )

        cv2.imshow("output", img)
        cv2.waitKey(1)

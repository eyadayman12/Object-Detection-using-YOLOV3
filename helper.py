import numpy as np
import cv2
from PIL import Image


def detect_image(img):
    net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
    classes = []
    with open('coco.names.txt', 'r') as obj_f:
        classes = obj_f.read().splitlines()

    img = np.array(img.convert('RGB'))

    height, width, channels = img.shape
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)  # set the input from the blob into the network
    output_layers_names = net.getUnconnectedOutLayersNames()  # get the output layers names
    layersOutput = net.forward(
        output_layers_names)  # passing output layers names to forward network function we will get the output from this funciton
    boundary_boxes = []
    probabilities = []
    predicted_classes = []
    for output in layersOutput:  # extract all the information from the layers output
        for detection in output:  # extract the information from each of the outputs
            scores = detection[5:]  # store all the acting classes predictions
            class_id = np.argmax(scores)  # store the locations that contains the higher scores
            probability = scores[class_id]  # extract the higher scores,
            # bec. we want to make sure that thier their predictions has a confidence that is high enough to consider that the object has been detected
            if probability > 0.5:
                center_x = int(detection[0] * width)  # scale it back
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # bec. yolo predicts the results with the center of the bounding boxes
                # extract the upper left cornor position
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boundary_boxes.append([x, y, w, h])
                probabilities.append((float(probability)))
                predicted_classes.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boundary_boxes, probabilities, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(boundary_boxes), 3))
    for i in indexes.flatten():
        x, y, w, h = boundary_boxes[i]
        label = str(classes[predicted_classes[i]])
        probability = str(round(probabilities[i], 2))
        color = colors[i]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label + " " + probability, (x, y + 20), font, 2, (0, 255, 0), 2)
    return img


def detect_video(video):
    net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
    classes = []
    with open('coco.names.txt', 'r') as obj_f:
        classes = obj_f.read().splitlines()
    vid = cv2.VideoCapture(video)

    while True:
        _, img = vid.read()
        height, width, channels = img.shape
        blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
        net.setInput(blob)  # set the input from the blob into the network
        output_layers_names = net.getUnconnectedOutLayersNames()  # get the output layers names
        layersOutput = net.forward(
            output_layers_names)  # passing output layers names to forward network function we will get the output from this funciton
        boundary_boxes = []
        probabilities = []
        predicted_classes = []
        for output in layersOutput:  # extract all the information from the layers output
            for detection in output:  # extract the information from each of the outputs
                scores = detection[5:]  # store all the acting classes predictions
                class_id = np.argmax(scores)  # store the locations that contains the higher scores
                probability = scores[class_id]  # extract the higher scores,
                # bec. we want to make sure that thier their predictions has a confidence that is high enough to consider that the object has been detected
                if probability > 0.5:
                    center_x = int(detection[0] * width)  # scale it back
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    # bec. yolo predicts the results with the center of the bounding boxes
                    # extract the upper left cornor position
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boundary_boxes.append([x, y, w, h])
                    probabilities.append((float(probability)))
                    predicted_classes.append(class_id)
        indexes = cv2.dnn.NMSBoxes(boundary_boxes, probabilities, 0.5, 0.4)
        font = cv2.FONT_HERSHEY_PLAIN
        colors = np.random.uniform(0, 255, size=(len(boundary_boxes), 3))
        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boundary_boxes[i]
                label = str(classes[predicted_classes[i]])
                probability = str(round(probabilities[i], 2))
                color = colors[i]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label + " " + probability, (x, y + 20), font, 2, (255, 255, 255), 2)

        cv2.imshow('Video', img)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    vid.release()
    cv2.destroyAllWindows()


def detect_webcame():
    net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
    classes = []
    with open('coco.names.txt', 'r') as obj_f:
        classes = obj_f.read().splitlines()
    vid = cv2.VideoCapture(0)
    while True:
        _, img = vid.read()
        height, width, channels = img.shape
        blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
        net.setInput(blob)  # set the input from the blob into the network
        output_layers_names = net.getUnconnectedOutLayersNames()  # get the output layers names
        layersOutput = net.forward(
            output_layers_names)  # passing output layers names to forward network function we will get the output from this funciton
        boundary_boxes = []
        probabilities = []
        predicted_classes = []
        for output in layersOutput:  # extract all the information from the layers output
            for detection in output:  # extract the information from each of the outputs
                scores = detection[5:]  # store all the acting classes predictions
                class_id = np.argmax(scores)  # store the locations that contains the higher scores
                probability = scores[class_id]  # extract the higher scores,
                # bec. we want to make sure that thier their predictions has a confidence that is high enough to consider that the object has been detected
                if probability > 0.5:
                    center_x = int(detection[0] * width)  # scale it back
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    # bec. yolo predicts the results with the center of the bounding boxes
                    # extract the upper left cornor position
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boundary_boxes.append([x, y, w, h])
                    probabilities.append((float(probability)))
                    predicted_classes.append(class_id)
        indexes = cv2.dnn.NMSBoxes(boundary_boxes, probabilities, 0.5, 0.4)
        font = cv2.FONT_HERSHEY_PLAIN
        colors = np.random.uniform(0, 255, size=(len(boundary_boxes), 3))
        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boundary_boxes[i]
                label = str(classes[predicted_classes[i]])
                probability = str(round(probabilities[i], 2))
                color = colors[i]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label + " " + probability, (x, y + 20), font, 2, (255, 255, 255), 2)
        cv2.imshow('Image', img)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    vid.release()
    cv2.destroyAllWindows()


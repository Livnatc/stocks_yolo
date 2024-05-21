from ultralyticsplus import YOLO, render_result
import cv2


if __name__ == '__main__':
    # load model
    model = YOLO('foduucom/stockmarket-pattern-detection-yolov8')

    # set model parameters
    model.overrides['conf'] = 0.25  # NMS confidence threshold
    model.overrides['iou'] = 0.45  # NMS IoU threshold
    model.overrides['agnostic_nms'] = False  # NMS class-agnostic
    model.overrides['max_det'] = 1000  # maximum number of detections per image

    # initialize video capture
    # Open the video file
    img_path = "nvda_3.png"
    # cap = cv2.VideoCapture(video_path)

    # Loop through the video frames
    # while cap.isOpened():
        # Read a frame from the video
    frame = cv2.imread(img_path)

    # Run YOLOv8 inference on the frame
    results = model(frame)

    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    # Display the annotated frame
    cv2.imshow("YOLOv8 Inference", annotated_frame)
    cv2.waitKey(100)
    cv2.destroyAllWindows()


print('Done')

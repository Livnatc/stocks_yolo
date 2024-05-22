from ultralyticsplus import YOLO, render_result
import cv2
import os


if __name__ == '__main__':

    results_path = 'results_stocks-22-05-24'
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    src_data_path = 'stocks-22-05-24'
    src_data = os.listdir(src_data_path)

    # load model
    model = YOLO('foduucom/stockmarket-pattern-detection-yolov8')

    # set model parameters
    model.overrides['conf'] = 0.25  # NMS confidence threshold
    model.overrides['iou'] = 0.45  # NMS IoU threshold
    model.overrides['agnostic_nms'] = False  # NMS class-agnostic
    model.overrides['max_det'] = 1000  # maximum number of detections per image


    for img in src_data:
        if img.endswith('.png'):

            # Open the img file
            img_path = os.path.join(src_data_path, img)
            frame = cv2.imread(img_path)

            # Run YOLOv8 inference on the frame
            results = model(frame)

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Display the annotated frame
            # cv2.imshow("YOLOv8 Inference", annotated_frame)
            # cv2.waitKey(50)
            # cv2.destroyAllWindows()

            # Save the annotated frame
            cv2.imwrite(os.path.join(results_path, img), annotated_frame)


print('Done')

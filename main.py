from ultralytics import YOLO
import os

def grading_v2(model, image_directory, output_directory):
    for filename in os.listdir(image_directory):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            results = model.predict(os.path.join(image_directory, filename), save=True, stream=True)
            txt_filename = os.path.join(output_directory, os.path.splitext(filename)[0] + '.txt')
            # Check if results have any detections
            detections = False
            with open(txt_filename, 'w') as f:
                for result in results:
                    if len(result.boxes) > 0:
                        detections = True
                        boxes = result.boxes  
                        conf_scores = boxes.conf
                        classes = boxes.cls
                        w_orig, h_orig = result.orig_shape
                        print("width img:", w_orig)
                        print("h img:", h_orig)
                        for i in range(len(boxes)):
                            x1, y1, x2, y2 = boxes[i].xywh[0].cpu().numpy().astype(int)
                            print("Original res not scaled:", x1, y1, x2, y2)
                            cls = int(classes[i])
                            x1_norm = x1 / h_orig
                            x2_norm = x2 / h_orig
                            y1_norm = y1 / w_orig
                            y2_norm = y2 / w_orig
                            f.write(f"{cls} {x1_norm:.4f} {y1_norm:.4f} {x2_norm:.4f} {y2_norm:.4f} {conf_scores[i]:.4f}\n")
                # Empty file if no detections (consistency with previous behavior)
                if not detections:
                    f.write('')

if __name__ == "__main__":
    model = YOLO("model_path/Yolov5m_Colab_18_05_2024_889_v3_Sone/train10/weights/best.pt")
    image_directory = 'FINAL_test/images'
    output_directory = 'FINAL_test/labels/predict'

    os.makedirs(output_directory, exist_ok=True)
    grading_v2(model, image_directory, output_directory)
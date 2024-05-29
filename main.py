from ultralytics import YOLO
import os

# this is the function to generate output from trained model under the format: x_center normalized - y_center normalized - w - h - confidence score. 
# The model is loaded by YOLO class of Ultralytics library. The image_directory is the folder directory of images you want to inference. The output_directory is the directory of outputs.
def grading_v2(model, image_directory, output_directory):
    for filename in os.listdir(image_directory):
      results = model.predict(os.path.join(image_directory, filename), save=True, stream=True)
      txt_filename = os.path.join(output_directory, os.path.splitext(filename)[0] + '.txt')

      lines = []
      detections = False

      for result in results:
        if len(result.boxes) > 0:
            detections = True
            boxes = result.boxes
            conf_scores = boxes.conf
            classes = boxes.cls
            w_orig, h_orig = result.orig_shape
            for i in range(len(boxes)):
              x, y, w, h = boxes[i].xywhn[0].cpu().numpy().astype(float)
              cls = int(classes[i])
              lines.append(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f} {conf_scores[i]:.6f}\n")

      with open(txt_filename, 'w') as f:
        f.writelines(lines)
        if not detections:
          f.write('')
        f.close()

if __name__ == "__main__":
    # loading model
    model = YOLO("model_path/Yolov5m_Colab_18_05_2024_889_v3_Sone/train10/weights/best.pt")
    # image directory
    image_directory = 'FINAL_test/images'
    #output_directory with the format mentioned above.
    output_directory = 'FINAL_test/labels/predict/CCWH_conf'

    os.makedirs(output_directory, exist_ok=True)
    grading_v2(image_directory, output_directory)

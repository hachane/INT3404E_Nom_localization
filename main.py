from ultralytics import YOLO
import os

def grading(model, image_directory, output_directory):
    for filename in os.listdir(image_directory):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            results = model.predict(os.path.join(image_directory, filename), save=True, stream=True)
            txt_filename = os.path.join(output_directory, os.path.splitext(filename)[0] + '.txt')
            
            # Check if results have any detections
            detections = False
            for result in results:
                if len(result.boxes) > 0:
                    detections = True
                    result.save_txt(txt_filename)
                    break
            
            # If no detections, create an empty txt file
            if not detections:
                with open(txt_filename, 'w') as f:
                    f.write('')

if __name__ == "__main__":
    model = YOLO("model_path/YOLOv8_Colab_12_05_2024_v1_Ha/weights/best.pt")
    image_directory = 'wb_localization_dataset_NomNa'
    output_directory = 'FINAL_test/labels'

    os.makedirs(output_directory, exist_ok=True)
    grading(image_directory, output_directory)
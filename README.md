# Sino-Nom Character Localization - Group 2 
This is a mid-term project in Image Processing Course INT3404E. The goal of the project is to localize Sino-nom character by leveraging data augmentation techniques on the given dataset and using YOLO models. 
## Installation
1. Clone the repository
```
git clone https://github.com/hachane/NomOCR.git
```
2. Install neccessary libraries
```
pip install -r requirements.txt
```
3. Put source image files in [images](FINAL_test/images)
4. Run [main.py](main.py)

## Structure
├── config #store config files for training model\
├── data-analysis\
│   ├── list_imgs_with_rate #results of data analysis\
│   └── data_analysis_hung.ipynb #data analysis script\
├── data-augmentation\
│   ├── augment_data #generated data using augmentation techniques\
│   └── data_augmentation_anizme_and_hang #data augmentation script\
│   └── utils #basic functions
├── dataset #datasets used in the project\
├── model_path #weights and biases of models\
└── report #reports of the project\
## Contributors
Tang Vinh Ha - 22028129\
Vu Nguyet Hang - 22028079\
Le Xuan Hung - 22028172\
Le Thi Hai Anh - 22028162
## Result
With a strategy mainly focusing on data and taking advantage of famous existing architectures, our team had a model accuracy of 86.7% on the original val set and 83.67% on the val set provided on evaluation day.
## Reports
For more information on the methods and models used in this project, please refer to [report](report)


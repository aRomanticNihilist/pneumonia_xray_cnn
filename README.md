Starting with the dataset on Kaggle: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

DISCLAIMER:

- Because of time constraints and the complexities of this project for a beginner, dynamic path handling wasn't implemented, so the path for the dataset should be fixed at "D:\\chest_xray\\chest_xray_original"
- After running organiseImages.py in step 1, inside the above directory, there should be 3 folders, namely "test", "train" and "val", which contain the images.

---

DEPENDENCIES

Package Version

absl-py 2.1.0
astunparse 1.6.3
certifi 2024.7.4
charset-normalizer 3.3.2
contourpy 1.2.1
cycler 0.12.1
flatbuffers 24.3.25
fonttools 4.53.1
gast 0.6.0
google-pasta 0.2.0
graphviz 0.20.3
grpcio 1.64.1
h5py 3.11.0
idna 3.7
imageio 2.34.2
joblib 1.4.2
keras 3.4.1
kiwisolver 1.4.5
libclang 18.1.1
Markdown 3.6
markdown-it-py 3.0.0
MarkupSafe 2.1.5
matplotlib 3.9.1
mdurl 0.1.2
ml-dtypes 0.4.0
namex 0.0.8
numpy 1.26.4
opencv-python 4.10.0.84
opt-einsum 3.3.0
optree 0.12.1
packaging 24.1
pillow 10.4.0
pip 24.1.2
protobuf 4.25.3
pydot 2.0.0
Pygments 2.18.0
pyparsing 3.1.2
python-dateutil 2.9.0.post0
requests 2.32.3
rich 13.7.1
scikit-learn 1.5.1
scipy 1.14.0
setuptools 70.3.0
six 1.16.0
tensorboard 2.17.0
tensorboard-data-server 0.7.2
tensorflow 2.17.0
tensorflow-intel 2.17.0
termcolor 2.4.0
threadpoolctl 3.5.0
typing_extensions 4.12.2
urllib3 2.2.2
Werkzeug 3.0.3
wheel 0.43.0
wrapt 1.16.0

---

HOW TO USE

1. Preprocess the data:
   1.1. Run organiseImages.py
   1.2. Run balance_test_and_val.py

2. Run the model training script: train_pneumonia_model.py
   (This will also creates 2 graphs to show training/validation accuracy and loss during each epoch
   IF you uncomment the plotting part at the end.)

3. Run see_model.py to see a visual representation of the network

4. To test a single image, change the path at line 19 of test.py to the path of the to be tested image, then run test.py

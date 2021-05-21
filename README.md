# EvolvFit
EvolvFit Submission
I have Written the code on Jupyter Notebook. 
There are two files Training.py/Training.ipynb and Camera.py/Camera.ipynb.
Training.py is for training the model on the dataset given which solves the Level1 of the challenge.
Camera.py is for reading the faces through webcam and make a bounding box around the face, then it displays the label name same as folder name on the screen.
Camera.py satisfies the requirement of Level2 of the challenge.


Only requirement TensorFlow version 2.4.0 and Keras 2.4.3 for compiling the code and training the model.


Model Used- 1.VggFace which runs on top of resnet50 is used for training and making the machine learn to detect the faces.
            2.Used MTCNN to extract the face features during training and also in level 2.
            
Training and Validation accuracy generated during the training, I have uploaded the screenshot of log report of training.
Model achieved the training accuracy of 91.42% after 20 epochs.

I am uploading a screenshot of prediction made by running the Camera.py script for evaluation.

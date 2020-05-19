Gymnastic Activity Recognition 

In this repository:

- procesoVideoEjercicios.py: Receives as input a video and carries out the detection of exercises and its evaluation. It produces an output video.
This program computes the OpenPose, the neural network to classify postures, the HMM to filter the postures and the modified Levenshtein distance.
- testCompleto_parte1.py: This part only carries out the detection of postures (MLP and HMM).
- testCompleto_parte2.py: This part only carries out the computation of Levenshtein distance.
- modeloPosturas.h5: Trained model of MLP neural network.
- training.zip: This zip contains the programs used to train our MLP starting from OpenPose joints.
- DATASET: Because of the size of the dataset is 4Gb, it is not uploaded to the repository. In case you need it, please email us.
- GYMNASTIC_ACTIVITY_RECOGNITION_ROBOT.mp4. This video shows different exercises that are evaluated and a Pepper robot that repeats them.
Take into account that the robot has repeated the exercises after they have been recorded. The video has been mounted after carrying out the 
exercises putting together the videos of the robot and the person. 



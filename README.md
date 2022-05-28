## Saimese Neural Network Implementation for Face Recognition

This project is an implementation of the paper “Siamese Neural Networks for One-shot Image Recognition” which is the base paper that is well known to introduce siamese networks. A shared weights CNN based Siamese network is used for the task of face verification. The model is trained on a single identity’s faces and it learns to verify weather any new input is the same person or not.

### Code structure:
- results
  - images
- src
   - __init__.py
   - collect_data.py
   - extract_images.py
   - validate_app.py

### Features:

- A keras-tensorflow implementation of the classical siamese networks introduced  for the task of face verification.
- Includes a shared weights CNN network that is trained to recognise and verify a single identity.
- The code base contains data collection and preprocessing scripts that helps capture data from webcam for a single identity, detect faces in them, segregate and store them as a dataset.
- The main script contains multiple sections of code:
- The initial section loads the data from the folders, performs preprocessing necessary and prepapres the data iterators in batches containing anchor, positive and negative images.
- The next section contains the model definition which consists of 2 CNN networks with shared weights in a siamese template. There is also an efficient implementation of the Contrastive Loss.
- Further section consists of the training, validation and inference modules for the model. There are also modules that help in visualizing the loss curves, plotting the final results etc.
- This is and end to end module with a CLI interface that helps capture input data from web cam, perform all evaluation and return the verification results.

### Dataset
The dataset used is a self prepared dataset consisting of 300 images for the person to be identified in different poses, lightning conditions, variations etc and the negative class containing 300 negative images.

### Running the script:
Setup the environment 
- Run `collect_data.py` to initiate data collection.
- Run `extract_images.py` to arrange the data in suitable format
- The main script `validate_app.py` performs all the rest steps.


### Results

 <img align="left" width="200" height="200" src="https://github.com/isrishtisingh/saimese-implementation-for-face-recognition/blob/ecf614c74cb4cf2fc11f5980a82f90875d2d3307/matched1.png">
 <img align="center" width="200" height="200" src="https://github.com/isrishtisingh/saimese-implementation-for-face-recognition/blob/ecf614c74cb4cf2fc11f5980a82f90875d2d3307/notmatched1.png">
 
 <br><br>
 <img align="left" width="200" height="200" src="https://github.com/isrishtisingh/saimese-implementation-for-face-recognition/blob/ecf614c74cb4cf2fc11f5980a82f90875d2d3307/matched2.png">
 <img align="center" width="200" height="200" src="https://github.com/isrishtisingh/saimese-implementation-for-face-recognition/blob/ecf614c74cb4cf2fc11f5980a82f90875d2d3307/notmatched2.png">

<br>

### References
- [Paper](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)
- [Code](https://youtu.be/bK_k7eebGgc)

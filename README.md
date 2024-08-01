# Lung-Cancer-CNN-Classifier
**Steps to Train the Model:**

1. Clone the [Lung-Cancer-CNN-Classifier](https://github.com/MatteoPassalent/Lung-Cancer-CNN-Classifier) repo to a local directory  
1. Unzip the dataset  
1. Copy the .ipynb files to your notebook environment   
1. Inside each notebook update the train\_path and test\_path variables to point to the dataset in your local directory.   
1. You should now be able to run all cells, training, testing and, evaluating the model.

**Additional Steps for Google Colab:**  
If you are using Google Colab you will need to add the following import:   
from google.colab import drive

Additionally, you will need to upload the zipped dataset to your drive then unzip it in the Colab environment. Example:  
drive.mount('/content/drive')  
\!unzip /content/drive/MyDrive/CP468/LungCancerCT.zip \-d Data   
\# Adjust the above path to point to your dataset^

The train\_path and test\_path variables should now start with /content/Data  
Eg: train\_path \= "/content/Data/LungCancerCT/CTscans"

**Testing the GUI:**  
The [GUI](https://huggingface.co/spaces/matteopassalent/Custom-CNN) can be tested with any test image from the dataset. We provided two test images to try it out in the DropBox submission. The images can be uploaded to the GUI to see the results.

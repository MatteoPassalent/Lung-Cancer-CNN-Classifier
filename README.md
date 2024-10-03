### Using a CNN to Identify Lung Cancer from CT Images

Link to Demo: [CNN-Demo](https://matteopassalent.ca/)

**Abstract**  
This report compares the performance of an ensemble model built from three adapted pre-trained models to an image classification CNN made from scratch. All models were trained and evaluated on a small dataset of approximately 1000 CT lung scans. They were assessed on their ability to perform binary classification by identifying each image as positive or negative for lung cancer. The three pre-trained models were developed to build an understanding of the problem and find the optimal configuration for the custom CNN.

**Targets for Performance:**  
	Two target metrics were set to evaluate the final CNN’s performance. Firstly, it needed to identify positive cases of lung cancer with a test accuracy of at least 95%. Additionally, given the limited amount of data, the model needed to be configured to prevent overfitting. To ensure sufficient generalization, a limit of 5% was set for the difference between the average training accuracy and average validation accuracy (generalization gap).

**Pre-trained CNNs**  
I chose three models of varying filter and parameter sizes that would provide good insight into the best configuration for the final model. For each model, the convolutional base was frozen and used for feature extraction while two new fully connected layers were trained on top of it. After the initial training, I unfroze the top block of the pre-trained model to fine-tune it on the dataset. The three models chosen were InceptionV3, VGG16, and MobileNetV2.   
InceptionV3 is the largest and most complex of the three with approximately 48 layers and 24 million parameters. Given its size and complexity, it was initially expected that this model would have the best accuracy. However, due to the small data set and simple binary classification, it was found that the larger models were prone to overfitting.  
VGG16 is a medium-sized model, it has only 16 layers however each layer has high complexity with around 138 million parameters in total. This model performed similarly to InceptionV3. However, the large amount of parameters appeared to be overly complex for the limited data and resulted in slightly increased overfitting and less overall accuracy. (See Table 1\)  
MobileNetV2 is the smallest of the three. Although it has around 53 layers, it is a relatively simple model with only 3.4 million parameters. Not only was this model around 3 seconds faster per epoch but it also reached peak accuracy in significantly fewer epochs. This made it much more efficient to train. Additionally, it also had the highest accuracy and the least amount of overfitting. From this, it was concluded that a lighter and simpler model was best for the small data set.   
While adapting the pre-trained CNNs for optimal performance, certain configurations were found to improve performance universally across all three models. Firstly, given the limited data set, data augmentation was very effective in improving generalization. Similarly, regularization techniques including L2 regularization and dropout further reduced overfitting to below the 5% target. It was found that applying these techniques to the fully connected layer with standard regularization parameters of 0.001 for L2 and 0.5 for dropout was optimal for all three models. Lastly, early stopping was not necessary but helped identify the optimal amount of epochs to run for each model. 

**Ensemble Model**  
	The ensemble model takes an average of each model's prediction and determines a final value. The three pre-trained models use a dense output layer with a sigmoid activation function to produce a probability of lung cancer between 0 and 1\. Below 0.5 is classified as positive and above is classified as negative. The ensemble model takes an average of these predictions and returns a final output from all three models. In this process, each model has an equal voting share, which significantly reduces variance and increases reliability.   
Due to the model’s high accuracy and limited testing data, the final test accuracy for the ensemble model was 100%. Therefore, the confusion matrix and ROC curve were not very informative but are included in the appendix for reference (See Figure 3). However, the improved performance of the ensemble model is evident in its confidence levels when predicting test images. This is because the averaging smooths out predictions and reduces uncertainty, leading to higher confidence levels. When testing the ensemble model through the Gradio GUI a confidence level of 100% is almost always observed.

**Final Custom CNN**  
Using the information gained from pre-trained models a convolutional model was built from scratch and fine-tuned for optimal performance on the dataset. The model architecture was intentionally kept small and simple, consisting of three convolutional layers followed by one fully connected layer. These layers had 16, 32, 64, and 256 nodes, respectively. This configuration proved sufficient for extracting the necessary features to accurately classify images, without overlearning specific features from the training data.  
There was some speculation about whether regularization was necessary given the simplified model. The concern was that since the dense layer had only 256 nodes, adding L2 regularization and dropout might contribute to underfitting and prevent the model from learning the necessary patterns in the data. To address this, some testing was done to evaluate the impact of regularization on the model's performance.   
The results reaffirmed the findings from the pre-trained models. Not only did the regularization techniques reduce overfitting, but they also improved the overall accuracy of the model on validation and testing data. The graphs below illustrate the enhanced generalization ability of the model with L2 regularization and dropout (right) compared to the same model without regularization techniques (left). The differences in the loss functions are particularly notable, highlighting the significant improvement achieved with regularization.  
![image1](https://github.com/MatteoPassalent/Lung-Cancer-CNN-Classifier/blob/main/report_images/graph1.jpg)![image2](https://github.com/MatteoPassalent/Lung-Cancer-CNN-Classifier/blob/main/report_images/graph2.jpg)  
*Figure 1: Loss and accuracy for CNN with no regularization (left) and with L2 and dropout (right)*

After implementing regularization techniques, the generalization gap decreased from 6.6% to 3.9%, comfortably below the target of 5%. Furthermore, the final testing accuracy increased from 81% to 98%, surpassing the target of 95% and highlighting the significant improvement in the model's overall performance.   
	A deeper analysis of the final model's performance revealed a tendency to favour false positives when inaccuracies occurred. This bias towards false positives is preferable in the practical context of the problem at hand. The consequences of the model incorrectly classifying a healthy person as positive for cancer are significantly less severe than missing a true positive case. Therefore, the model's bias towards false positives aligns well with the priority of minimizing false negatives at all costs.   
The graph below illustrates the confusion matrix and ROC curve of the final CNN evaluated on unseen test data. The confusion matrix shows near-perfect performance with a FN, and FP rate of 0% and 2% respectively. Additionally, the ROC curve, which reflects an AUC of 1.0, also highlights the model's performance. However, the ROC curve is less informative due to the 0% FN rate, resulting in a perfect TP rate and a straight-line representation.  
![image3](https://github.com/MatteoPassalent/Lung-Cancer-CNN-Classifier/blob/main/report_images/graph3.png)![image4](https://github.com/MatteoPassalent/Lung-Cancer-CNN-Classifier/blob/main/report_images/graph4.png)  
*Figure 2: Confusion matrix and ROC curve of final CNN on test data*

### Additional Discussions:  
**Optimizers:**  
	RMSprop optimizer was used for the three pre-trained models. This optimizer was used in textbook examples and proved to be best for the pre-trained models. According to both online sources and textbook information, RMSprop is particularly suitable when the learning rate needs to be explicitly adjusted over multiple training epochs. For the pre-trained models, I started with a lower learning rate of 2e-5 for initial training on the dense layers. I then further lowered the learning rate to 1e-5 to fine-tune the top layers of the convolutional base. This lower learning rate helped ensure that the pre-trained weights were not significantly disrupted by large gradient updates.  
	For the custom CNN, I chose the Adam optimizer. Adam is a modern optimization algorithm that combines the benefits of both RMSprop and the momentum optimizer. It is known for general-purpose effectiveness and performs well without explicitly setting and changing the learning rate.

**Non-deterministic Nature of Training:**  
	During training and evaluation, I observed slight variations in the outcomes each time due to non-deterministic factors such as the random initialization of weights for new layers and the randomly applied data augmentations. In pre-trained models, this variability was minimal because the convolutional base used previously determined weights. However, for the custom CNN, every weight was randomly initialized and the differences in outcomes were more pronounced. Most of the time, these discrepancies averaged out, and the final model achieved the target of 95% testing accuracy with a generalization gap below 5%. Occasionally, the model would get stuck in a local maximum early in training, resulting in the validation accuracy remaining at 0.5769 from the first epoch to the 80th. I believe this occurred because if the model did not escape the local maximum early on, the optimizer's adjustments would gradually weaken, causing it to remain stuck indefinitely. This problem occurred rarely and was not a significant blocker, I simply restarted the training process when I noticed it was stuck.

### Appendix:

|  | Model |  |  |  |  |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Metric** | InceptionV3 | VGG16 | MobileNetV2 | Custom CNN | Ensemble Model |
| Training Acc | 0.97 | 0.95 | 0.97 | 0.88\* | \- |
| Validation Acc | 0.93 | 0.90 | 0.95 | 0.84\* | \- |
| Generalization Gap | 0.04 | 0.05 | 0.02 | 0.04 | \- |
| Testing Acc | 0.99 | 1.00 | 1.00 | 0.98 | 1.00 |

*Table 1: Comparison of training, validation, and testing accuracy for all five models*



\* Note that the training and validation accuracy for the Custom CNN are significantly lower because it was built without leveraging pre-trained weights, resulting in a randomly initialized starting point and low accuracy for early epochs.

![image5](https://github.com/MatteoPassalent/Lung-Cancer-CNN-Classifier/blob/main/report_images/graph5.png)![image6](https://github.com/MatteoPassalent/Lung-Cancer-CNN-Classifier/blob/main/report_images/graph6.png)  
*Figure 3: Confusion matrix and ROC curve of Ensemble Model on test data*


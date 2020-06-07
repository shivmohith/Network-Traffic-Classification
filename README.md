# Network-Traffic-Classification
 This project was done during months of April and March in 2019 as a mini project part of my machine learning course taken in my university.

## Description

Traffic classification is the first step for network anomaly detection or network based intrusion detection system and plays an important role in network security domain. The traffic observed in a network system can be classified mainly into two types – Normal (benign) traffic and Malware traffic. Unnecessary traffic from malicious sources can cause a lot of problems like flood the network bandwidth, DOS attacks and jam the receiver and thus detection of such unwanted traffic and classifying the required from the “malware” traffic is extremely important.

This project involves converting the raw network traffic, collected using the Wireshark tool, into images. These images are used to classify the network traffic.
There are 2 types of classification,
- Classifying as benign or malware - Binary classification
- Classifying into each class under benign and malware - Multi-class classification

#### Dataset

The dataset used for the training is USTC-TFC2016. The dataset consists of benign traffic and malware traffic. 
The following table shows the 10 different classes under beign and malware.

|Benign Classes |Malware Classes |
|---------------|----------------|
|BitTorrent     |Cridex          |
|Facetime       |Geodo           |
|FTP            |Htbot           |
|GMail          |Miuref          |
|MySQL          |Neris           |
|Outlook        |Nsis-ay         |
|Skype          |Shifu           |
|SMB            |Tinba           |
|Weibo          |Virut           |
|World of Craft |Zeus            |

There are 2 types of data for each class. One is the traffic information of only the 7th layer (Application layer) of the OSI model and the other is
traffic information of all the layers in the OSI model. This project classifies the network traffic using both the data individually.

## Methedology

1. Convert the raw Wireshark data in .pcap format into images. This was done using the code available in https://github.com/echowei/DeepTraffic/tree/master/1.malware_traffic_classification/2.PreprocessedTools(USTC-TK2016)
The following are the example images obtained,
![Example Image from each class](/Images/output_images.png)
The above images are grey scale images where each pixel value can range from 0 to 255

2. Extract the histogram of the images which are considered as the image features. These images features are stored in an excel file with the class label as the last column.
The histogram is obtained for different bin sizes (1, 2, 4, 8)

3. Use Support vector machines to for classification

## Running the application

1. Install the required libraries specified in *requirements.txt*
2. Extract histogram features of images and save them using the python files in Pre-processing directory
3. Perform classifcation by running the files in Classification directory
    - *svm_binary.py*: to perform binary classification
    - *svm_multi.py*: to perform multi-class classification
**Replace the file paths in the code with the appropriate file paths**

## References
*Wei Wang, Ming Zhu, ‘Malware Traffic Classification Using Convolutional Neural Network for Representation Learning’* 

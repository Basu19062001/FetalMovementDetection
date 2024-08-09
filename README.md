Introduction

Maternal health, encompassing pregnancy, child- birth, and the immediate postpartum period, is critical to global healthcare. In economically disad- vantaged countries, inadequate treatment and ed- ucation contribute significantly to maternal mor- tality. Hypertension during pregnancy, such as preeclampsia, poses substantial risks. Routine pre- natal appointments, crucial for monitoring and managing maternal health, involve nutritional reg- ular exercise planning, alongside medical assess- ments like fetal heart rate monitoring, blood pres- sure checks, weight measurements, fundal height assessments, and urine testing, is essential. Rou- tine medical evaluations and tracking fetal move- ments help in early detection of potential health is- sues.In rural areas, lack of awareness and financial constraints often limit access to necessary medical care, with the high cost of ultrasonic scanning de- vices being a significant barrier. IoT technology offers a promising solution by transforming conven- tional healthcare tools into intelligent, connected systems through embedded devices, communication protocols, sensor networks, and applications. This integration facilitates seamless collection, manage- ment, and sharing of healthcare information, in- cluding diagnosis, treatment, recovery, inventory, and medication.The proposed system uses various sensors, including an accelerometer for monitoring fetal movements. Data from these sensors is com- municated through an Arduino Uno and IoT tech- nology to a software program accessible via mo- bile devices or PCs. Graphical data is displayed on an LCD, with alerts for abnormal values, en- abling timely intervention and improving maternal health outcomes. This innovative approach lever- ages technology to revolutionize traditional health- care practices, enhancing accessibility and effective- ness. Fetal monitoring, a cornerstone of prenatal care, predominantly uses ultrasound methods like fetal echocardiography (fECHO) and cardiotocog- raphy (CTG).
However, CTG’s sensitivity to noise from maternal movements necessitates frequent repositioning of ultrasound transducers. Alternative methods such as fetal electrocardiography (fECG), phonocardio- graphy (fPCG), and magnetocardiography (fMCG) provide unique insights into fetal heart activity, each with distinct advantages and challenges.

Project Objectives

The objective of this project is to develop a compre- hensive system for tracking fetal movement in the womb, crucial for monitoring the health and devel- opment of the baby. By employing ADXL335 and MPU6050 [3] sensors, detailed data on the move- ments of both the mother and the fetus can be col- lected. These sensors measure acceleration along the x, y, and z axes, providing comprehensive infor- mation about the dynamic environment inside the womb. The primary aim is to distinguish between the mother’s movements and those of the fetus us- ing machine learning algorithms, thereby enabling precise monitoring of fetal activity.

1.	Sensors and Data Collection: Utilizing the ADXL335 [4] and MPU6050 sensors, which offer analog and digital output signals respectively, in conjunction with an Arduino UNO, real-time data on intrauterine movements can be captured. This data is then transmitted to a cloud-based platform such as ThingSpeak for storage and analysis.
2.	Data Transmission and Preprocessing: The collected sensor data is processed using the Ar- duino UNO to ensure accuracy and reliability. Sub- sequently, the raw data is transmitted to ThingS- peak [5] for real-time collection and visualization. Preprocessing steps involve noise filtering, data nor- malization, and segmentation into meaningful time windows to facilitate further analysis.
3.	Machine Learning for Movement Classifi- cation:
•	Feature Extraction: Relevant features are extracted from the preprocessed sensor data, potentially including statistical measures.
•	Data Labeling: Initially, a portion of the data is manually labeled to differentiate be- tween maternal and fetal movements, forming the ground truth dataset for training the ma- chine learning models. [6]
Component Selection
a.	Accelerometer Sensors for Detecting Fe- tal Movement
We selected the ADXL335 and MPU6050 ac- celerometers for precise fetal movement detection. The ADXL335 is an analog sensor known for its low power consumption and high resolution, making it suitable for continuous monitoring. The MPU6050, combining a 3-axis accelerometer and a 3-axis gy- roscope, offers comprehensive motion detection ca- pabilities. These sensors provide raw data essential for accurate and reliable fetal activity monitoring.

Figure 1: Real-time
b.	Microcontroller  and  Sensor  Connection The Arduino Uno is utilized as the microcontroller due to its simplicity and compatibility with var- ious sensors.	We connected the ADXL335 and MPU6050 to the Arduino Uno, allowing efficient data collection and processing. The Arduino Uno’s versatility and ease of use make it ideal for integrat- ing multiple sensors and managing their raw data output effectively, ensuring precise detection of fe- tal movements. [12]
c.	Wi-Fi Module for Data Transmission
For data transmission, we employed the ESP8266 Wi-Fi module. This module retrieves data from the Arduino Uno and sends it wirelessly to the cloud, facilitating real-time monitoring and analysis of fe- tal movements. By connecting to a Wi-Fi [13]
 
network, the ESP8266 ensures seamless and effi- cient data transfer, enhancing the system’s flexibil- ity and user-friendliness. [14]
d.	Portable Power Supply
A lithium-ion battery powers the Arduino Uno and ESP8266 Wi-Fi module, ensuring continuous and mobile operation. Chosen for its high energy den- sity and reliability, the lithium-ion battery effec- tively powers both the microcontroller and Wi-Fi module, maintaining the system’s functionality for extended periods. This portable power source is crucial for uninterrupted fetal movement detection and data transmission.

Methodology

Overview
This project aims to develop an IoT device to detect fetal movement using two accelerometer sensors, an Arduino Uno, a Wi-Fi module, and a lithium-ion battery. The accelerometers capture precise fetal movements, which are processed in real-time by the Arduino Uno, the core microcontroller. The pro- cessed data is transmitted via the Wi-Fi module to a server for further analysis and monitoring. Pow- ered by a rechargeable lithium-ion battery, the de- vice ensures portability and continuous operation. This methodology enables real-time tracking of fe- tal movements, providing crucial insights into fetal health and development. The integration of these components creates a robust and efficient system for reliable and accurate fetal movement data.
1.	System Design
Our IoT device architecture connects two sensors, the ADXL335 and MPU6050, to an Arduino Uno microcontroller. The Arduino processes raw data from these sensors and transmits it wirelessly to the cloud via an ESP8266 Wi-Fi module. The sys- tem is powered by a lithium-ion battery connected to both the Arduino and the Wi-Fi module, en- suring continuous, portable operation for real-time fetal movement analysis.
1.1.	IoT Device Architecture : Two sensors connect to an Arduino Uno, which links to an
 
ESP8266 Wi-Fi module.  The Arduino and ESP8266 are powered by separate lithium-ion bat- teries, enabling data collection and wireless trans- mission to the cloud.

Figure 2: Architecture
1.2.	Circuit Diagrams for Component Con- nections : Our circuit diagram shows two sensors connected to an Arduino Uno, which receives 9V power stepped down to 5V using a voltage regula- tor. The ESP8266 Wi-Fi module, also connected to the Arduino, has its voltage regulated similarly. This setup ensures safe operation and minimizes radiation exposure, making the connections and power management in our IoT device easy to un- derstand.
2.	Arduino Setup
We use an Arduino Uno operating on 5V, powered by 9V with a voltage regulator stepping it down to 5V. The sensors, ADXL335 and MPU6050, connect to the Arduino to detect fetal movement. The raw data is processed by the Arduino and transmitted wirelessly to the cloud via the ESP8266 Wi-Fi mod- ule. Safety is prioritized by managing the power supply to minimize radiation risks.

Figure 3: Ardunio SetUp
 
3.	Sensor Placement
3.1.	Optimal Sensor Placement : We con- ducted preliminary tests using a non-maternal model to determine optimal sensor placement on the maternal abdomen. The device was placed on the model’s abdomen during activities like run- ning, walking, sleeping, and sitting. Data from these activities helped us refine the sensor place- ment to maximize accuracy, with the lower ab- domen, slightly off-center, providing the most con- sistent and reliable readings.

Figure 4: Actual Device
3.2.	Comfortable and Non-Intrusive Sensor Placement : The device is designed to be comfort- able and non-intrusive, allowing natural movement and daily activities without disruption. Our careful testing confirmed that the device does not hinder walking, sitting, sleeping, or other common activi- ties, providing a reliable and comfortable monitor- ing solution for expectant mothers.

 
Figure 5: Moddel
 
4.	Wi-Fi Module Configuration
4.1.	Configuring the Wi-Fi Module for Lo- cal Network Connection : We connected a mo- bile Wi-Fi hotspot to the ESP8266 Wi-Fi module by writing specific connection code, including the network’s SSID and password. This step involved several iterations to ensure a reliable connection, crucial for wireless data transmission to the cloud.
4.2.	Implementing Code for Stable Connec- tion and Reconnections : We wrote code for the ESP8266 to connect to the local Wi-Fi network, with logic for continuous connection checks and re- connections if lost. This ensures minimal downtime and reliable data transmission for real-time moni- toring and analysis.
4.3.	Ensuring Secure Data Transmission : We implemented encryption protocols for secure data transmission. The ESP8266 uses WPA2 encryp- tion for network access, and data sent to the cloud is encrypted using TLS, ensuring privacy and secu- rity of fetal movement data.
5.	Power Management
We selected components with low power consump- tion and optimized their usage to ensure prolonged battery life and reduced environmental impact. Ef- ficient power management maximizes the device’s usability and sustainability.
6.	Data Transmission
6.1.	Setting Up Arduino for Data Transmis- sion to a Remote Server : We programmed the sensors to gather data and transmit it to the Arduino Uno. The Arduino, connected to the ESP8266, establishes a Wi-Fi connection and trans- mits data to the cloud. Error handling mechanisms ensure robust data transmission despite potential interruptions.
6.2.	Ensuring Reliable and Timely Data Transmission with Error-Handling Mecha- nisms : We implemented retry logic and data re- send strategies to ensure reliable and timely data transmission. These error-handling mechanisms minimize disruptions, maintaining data integrity and ensuring prompt delivery of vital fetal move-
ment information.
 
7.	Server Setup
7.1.	Setting Up a Server to Receive and Store Transmitted Data :We utilized ThingS- peak, a cloud-based IoT platform, to receive and store transmitted data. ThingSpeak provides real- time data analysis and visualization, simplifying data management and enhancing monitoring capa- bilities.

Figure 6: Transmission
7.2.	Utilizing ThingSpeak’s CSV Storage for Data Management : ThingSpeak automatically stores data in CSV format, eliminating the need for a separate database. This streamlined approach fa- cilitates efficient data management and compatibil- ity with various analysis tools.
7.3.	Implementing APIs for Data Retrieval and Real-Time Updates :

Figure 7: API Working
We used ThingSpeak’s Channel API for data re- trieval and real-time updates.
 
The API enables seamless communication between the Wi-Fi module and ThingSpeak, ensuring con- tinuous monitoring and access to the most up-to- date sensor data.

Figure 8: Line Graph
8.	Embarking on the Journey: An Introduc- tion to Key Challenges
Before our analysis and results, we outline the project and challenges in data collection, program- ming, machine learning (ML), deep learning (DL), and integration. We managed these challenges by dividing dataset understanding into three steps: Step 1 Creating Target Features :
a.	Initial Dataset Preparation : We created target features to understand sensor data values.
b.	Data Collection and Labeling : Sensors contin- uously gathered data, labeled with target columns corresponding to activities.
c.	Applying Machine Learning Algorithms : ML algorithms analyzed the labeled data to determine model accuracy.


Figure 9: Activities
 
Step 2 Unsupervised Learning :
a.	Dataset without Target Features : We used raw sensor data without predefined labels.
b.	Applying Unsupervised ML Techniques : Various algorithms analyzed the data to identify natural patterns.
c.	Evaluating Accuracy: Accuracy from unsu- pervised methods provided insights into data structures.

textbfStep 3 Deep Learning on Raw Data :
a.	Applying Deep Learning Techniques : DL models processed raw sensor data, achieving high accuracy.
b.	Developing a DL Model for Real-Time De- tection: A robust DL model was integrated for real-time activity detection.
8.1.	Integration and Real-Time Implemen- tation
Integration Challenges :	Integrating the DL model into a real-time application required efficient and accurate live data processing without delays. Real-Time Activity Detection :	The inte- grated DL model detected and classified activities in real-time, providing immediate feedback crucial for health monitoring systems.


Analysis and Result

Analysis:
Initial Dataset Preparation:
We collected raw sensor data from sensors placed on a model abdomen. The data included activ- ities like running, walking, sleeping, and sitting, recorded by sensors to avoid risks to a real person. This data was crucial for understanding how sensors work under different conditions. Using a model ensured safety while still providing accurate and reliable data.
 
Example Raw Data:
We collected raw sensor data from a model ab- domen during activities like running, walking, sleeping, and sitting. The controlled environment ensured safety and accurate data, despite difficul- ties in categorizing activities due to noise and arti- facts.

Figure 10: Simple
We manually created target features to label ac- tivities like running, walking, and sleeping. Con- sistently wearing the device while running, we col- lected data to form a distinct dataset, essential for applying machine learning algorithms.

Figure 11: Running
We manually created target features to label ac- tivities like running, walking, and sleeping. Con- sistently wearing the device, we collected data for distinct running and walking datasets, essential for applying machine learning algorithms, as shown in Tables 2 and 3.

Figure 12: Walking
We consistently wore the device while sleeping and sitting, collecting data to form our dataset. We manually created features for these activities, en- hancing activity recognition. The dataset for other activities, shown in Table 4, was similarly formed.
 
Next, we merged all activities, including running, walking, and others, into a single dataset, creating an overall dataset for analysis.

Figure 13: Other
We formed the dataset for merging activities by combining running, walking, and other activities consistently engaged in, as indicated in Table 5.

 
Figure 14: Marge
Data Collection and Labeling :

Continuous real-time data collection was conducted using sensors, while simultaneously labeling the data with target columns representing ongoing ac- tivities. This combined approach ensured the dataset’s comprehensiveness and accurate annota- tions. We imported essential libraries, read the CSV file using Pandas, checked for null values, and split the dataset into training and testing sets. We applied labels using LabelEncoder and standard- ized the data with MINMAX scaling. These prepro- cessing steps ensured well-structured data, proper model evaluation, and efficient learning. We then employed Wrapper-based methods, specifically Ex- haustive feature selection via SequentialFeatureSe- lector from the mlxtend library, to identify the op- timal feature subset, enhancing model performance and predictive accuracy.
Before implementing deep learning, we employed three clustering techniques—K-means, DBSCAN, and Agglomerative Clustering—to analyze and un- derstand our dataset. By visualizing the results with pair plots, we gained significant insights into the structure and distribution of the data. Here’s a summary of our findings from each method.
 
K-means Clustering:

K-means clustering partitions the dataset into a specified number of clusters (k). Each data point is assigned to the nearest cluster center, which is the mean of the points in that cluster. This method is effective for data with a clear, spherical cluster structure.
The pair plots for K-means clustering revealed that the clusters were well-separated when the data had a distinct structure. However, K-means struggled with non-spherical clusters, leading to less distinct boundaries in some cases.
Pairplot Analysis:
Pair plots were created to visualize the relationships between pairs of features in the dataset.

Figure 15: Pair Plot
DBSCAN (Density-Based Spatial Clustering of Applications with Noise):

DBSCAN groups data points based on their den- sity, identifying clusters of various shapes and sizes. It also detects outliers as noise. This method is par- ticularly useful for datasets with clusters that are not necessarily spherical and have different densi- ties.
The pair plots for DBSCAN showed that it ef- fectively identified clusters with varied shapes and densities. This made it suitable for more complex data structures. Additionally, DBSCAN success- fully marked outliers as noise, which helped in iden- tifying anomalous data points.
 
Pairplot Analysis:
Pair plots were created to visualize the relation- ships between pairs of features in the dataset.

Figure 16: Pair Plot

Agglomerative Clustering:

Agglomerative Clustering is a hierarchical method that starts with each data point as its own cluster and merges the closest clusters iteratively until a single cluster or a specified number of clusters is formed. Pair plots provided a detailed hierarchical view, with dendrograms offering insights into cluster similarities and nested structures, aiding in data interpretation.
Pairplot Analysis:
Pair plots were created to visualize the relation- ships between pairs of features in the dataset.


Figure 17: Pair Plot
 
Leveraging Deep Learning Techniques and Developing Real-Time Detection Models Analysis :

1.	Analysis :
In this phase, we leveraged various libraries and techniques to analyze and prepare our dataset for deep learning model development. We used pandas and numpy for data manipulation and numerical operations. These libraries enabled us to efficiently handle and process the large dataset required for our deep learning models.
2.	Visualization and Scaling :
We utilized matplotlib.pyplot and seaborn for data visualization. These tools helped us understand the distribution and relationships within the data. For instance, plotting heatmaps and scatter plots al- lowed us to identify patterns and correlations be- tween features. To standardize the features, we em- ployed StandardScaler, which normalized the data, ensuring that each feature contributed equally to the model’s training process.
3.	Data Splitting and Clustering :
To prepare the data for training and testing, we used train test split from sklearn.model selection. This function split the dataset into training and testing sets, ensuring that our model was evaluated on unseen data. Additionally, we applied KMeans clustering to explore potential groupings within the data. The clustering results were evaluated using metrics like silhouette score, which provided in- sights into the quality of the clusters formed.

Figure 18: K-Means
 
We used K-means clustering and pair plots to un- derstand the data for each column. This approach helped visualize the data distribution and relation- ships, aiding in identifying patterns and structures within the dataset.

Figure 19: Pair Plot
4.	Model Evaluation :
The performance of our models was assessed using accuracy score, confusion matrix, and Confusion- MatrixDisplay from sklearn.metrics. These metrics allowed us to quantitatively measure the accuracy and visualize the performance of our models, re- spectively. By analyzing the confusion matrices, we gained a deeper understanding of the model’s strengths and weaknesses.
5.	Deep Learning Model Development :
We developed our deep learning models using Se- quential, LSTM, Dense, and Dropout layers from keras. The Sequential model provided a straight- forward way to build a neural network by stacking layers. We used LSTM layers to capture temporal dependencies in the data, which is crucial for time- series analysis. Dense layers were used for fully con- nected layers, while Dropout layers helped prevent overfitting by randomly dropping a fraction of the neurons during training.
•	LSTM Equations : Forget Gate:
ft = σ(Wf · [ht−1, xt] + bf )	(1)


Input Gate:

it = σ(Wi · [ht−1, xt] + bi)	(2)
 
Candidate Cell State:

C˜t = tanh(WC · [ht−1, xt] + bC)	(3)



Cell State Update:

Ct = ft ⊙ Ct−1 + it ⊙ C˜t	(4)



Output Gate:

ot = σ(Wo · [ht−1, xt] + bo)	(5)



Hidden State Update:

ht = ot ⊙ tanh(Ct)	(6)

Dimensionality Reduction and Model Sav- ing:
To reduce the dimensionality of the data and im- prove computational efficiency, we employed PCA (Principal Component Analysis). This technique helped us retain the most important features while reducing the overall number of features. Finally, we used joblib to save our trained models, enabling us to easily load and deploy them in real-time detec- tion systems.
** Entries Algo———————————–
Algorithm: Machine Learning Pipeline for Classification and Deep Learning
Step 1: Import Libraries and Load Data
1.1.	Import necessary libraries: ‘pandas‘, ‘numpy‘, ‘matplotlib.pyplot‘, ‘seaborn‘, ‘StandardScaler‘, ‘train test split‘, ‘KMeans‘, ‘accuracy score‘, ‘confusion matrix‘,	‘ConfusionMatrixDisplay‘, ‘silhouette score‘, ‘Sequential‘, ‘LSTM‘, ‘Dense‘, ‘Dropout‘, ‘PCA‘, ‘joblib‘.
1.2.	Load  data  set  using ‘pd.read csv(’mainMargeDataset.csv’)‘.
 
Step 2: Data Preprocessing
2.1.	Select relevant columns for analysis.
2.2.	Convert ’created at’ to datetime format.
2.3.	Normalize the sensor values using ‘Standard- Scaler‘.
2.4.	Save the scaler for future use.
Step 3: Time Window Segmentation
3.1.	Set ’created at’ as the index
. 3.2. Resample data into 1-second windows and calculate the mean.
3.3. Drop any rows with missing values.
Step 4: Feature Extraction
4.1.	Define a function to extract features (mean, standard deviation, max, min) from each window.
4.2.	Apply the feature extraction function to each window.
4.3.	Convert the extracted features into a numpy array.
Step	5:	Principal	Component	Analysis (PCA)
5.1.	Perform PCA to reduce dimensionality to 2 components for visualization.
Step 6: Clustering
6.1.	Apply KMeans clustering to the feature set.
6.2.	Visualize clusters using PCA components.
6.3.	Calculate and print the silhouette score for the clustering.
Step 7: Sequence Creation
7.1.	Define a function to create sequences of data for LSTM input.
7.2.	Create sequences of a specified length (e.g., 10 seconds).
Step 8: Label Assignment
8.1.	Cluster the data using KMeans to assign ac- tivity labels.
8.2.	Map cluster labels to meaningful activity names.
8.3.	Encode activity labels using ‘LabelEncoder‘.
Step 9: Data Splitting
9.1.	Split sequences and labels into training and testing sets using ‘train test split‘.
 
Step 10: LSTM Model Building
10.1.	Define an LSTM model architecture with necessary layers (LSTM, Dropout, Dense).
10.2.	Compile the model with optimizer and loss function.
10.3.	Print the model summary.
Step 11: Model Training
11.1.	Train the LSTM model on the training data.
11.2.	Use a validation split during training to monitor performance.
Step 12: Model Evaluation
12.1.	Evaluate the model on the testing set.
12.2.	Print the test accuracy.
12.3.	Predict labels for the test set.
Step 13: Confusion Matrix Visualization
13.1.	Compute the confusion matrix for the test predictions.
13.2.	Visualize the confusion matrix using ‘Con- fusionMatrixDisplay‘.
Step 14: Model Saving
14.1.	Save the trained LSTM model using ‘model.save‘.
End

6.	Integration and Implementation Chal- lenges Analysis:
Data Handling and Preprocessing:
1.	Used pandas for data manipulation and numpy for numerical operations.
2.	Challenges:	Handling missing values and ensuring data consistency.
3.	Solutions: Normalization and feature extraction.

Model Persistence:
1.	Used joblib for saving/loading models and scalers.
2.	Challenges: Managing file paths and compati- bility across environments.
Deep Learning Implementation:
1.	Leveraged	tensorflow	and	tensor- flow.keras.models.
 
2.	Challenges: Model overfitting, mitigated by dropout and regularization.
Real-time Data Processing:
1.	Used requests, time, and threading for con- tinuous data streams. 2. Challenges: Data synchronization and latency, optimized data pipelines.
Feature Scaling:
1.	Used StandardScaler for normalization.
2.	Challenges: Maintaining consistent scaler parameters, handled by saving the scaler with joblib.
Concurrency and Multithreading:
1.	Used threading for concurrent tasks.
2.	Challenges: Thread safety and data integrity, ensured through careful synchronization mecha- nisms.
** Entries Algo———————————–
Algorithm:Real-Time Data Collection and Prediction for Fetal Movement Detection
Step 1: Import Necessary Libraries
1.1.	Import pandas,	numpy,	joblib,	tensorflow, requests, time, threading
1.2.	Import	load model	from	tensor- flow.keras.models
1.3.	Import	StandardScaler	from sklearn.preprocessing
Step 2: Load the Scaler and Model
2.1.	Load the scaler from ’scaler.pkl’ using joblib
2.2.	Load the pre-trained model from ’fetal movement detection model.h5’	using load model
Step 3: Define Data Preprocessing Function
3.1. Define a function ’preprocess data’ to nor- malize data and create sequences - Normalize data using the scaler
- Create sequences of a specified length
Step 4:	Define Function to Collect Real- Time Data from ThingSpeak
4.1.	Define a function ’collect real time data’ to fetch the latest data from ThingSpeak API
-	Build the URL with API key and channel ID
-	Send a GET request to the API
-	Parse and return the latest data
Step 5: Configure ThingSpeak API
5.1.	Set the API key and channel ID
5.2.	Specify the fields to be collected (e.g., ’field1’, ’field2’, ’field3’)
Step 6: Initialize Data Buffer and Control Variables
6.1.	Create a buffer to store incoming data
6.2.	Set the sequence length required by the model
6.3.	Initialize a flag to control the data collection loop
Step 7: Define Real-Time Prediction Func- tion
7.1.	Define a function ’real time prediction’ to handle real-time data collection and prediction
-	Continuously collect data and append to the buffer
-	Maintain a fixed buffer size
-	Process data if enough for one sequence
-	Make predictions using the model
-	Interpret and print the predicted activity
Step 8: Define Function to Stop Prediction
8.1.	Define a function ’stop prediction’ to set the stop flag and terminate the loop
Step 9: Start Real-Time Prediction in a Sep- arate Thread
9.1.	Create	and	start	a	thread	to	run ’real time prediction’
9.2.	Run ’stop prediction’ in the main thread
9.3.	Wait for the prediction thread to finish
End: Print a message indicating that real-time pre- diction has stopped.

Conclusion

This research proposes integrating wearable AI and IoT technology for maternal healthcare using MPU6050 and ADXL335 sensors to monitor fetal movements. By employing accelerometers and an Arduino Uno, our system provides real-time in- sights and alerts for timely interventions.
This approach aims to enhance prenatal care, par- ticularly in underserved areas, by improving mater- nal and fetal health monitoring, reducing mortality rates, and fostering a positive pregnancy experience through personalized, proactive care. Embracing these advancements redefines obstetric care stan- dards globally.

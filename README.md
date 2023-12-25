Our goal was to create a clear cut process of identifying and measuring bias that arises in various machine learning models. This includes distinguishing bias which arises due to the process of training the model, an pre-existing bias within the dataset. To read the full analysis, process, and conclusion of the project, please refer to Project Report.docx or view here: https://docs.google.com/document/d/1CY4VrrwWS0iTM9ocWlA1oLV9RBF5ckchKi25X3tzC60/edit?usp=sharing

The following is information relating to running the data for yourself, as well as information on how this data can be used for further research. 

To run any model and obtain a classification report, install the following libraries via command prompt:
	1. pip install pandas
 	2. pip install sklearn
  	3. pip install matplotlib
   
Both the KNN(K nearest neighbors) and SVM(support vector machine) model files contain code to train the model on a .csv file, and run the model on a .csv file. Refer to the project report to obtain a detailed explaination of the purpose of the additional .csv files provided in the 'clones' folder. 

The remaining python files 'make_clones.py' and 'make_models.py' can be referenced to modify the original data for testing purposes, such as blinding the model to certain features or changing modifying certain features. 

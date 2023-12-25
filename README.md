# Project Overview

Our goal was to create a clear-cut process for identifying and measuring bias that arises across various machine learning models. This was inspired by Amazon's scrapped [sexist AI](https://www.bbc.com/news/technology-45809919), as distinguishing correlational and causal features is difficult for an AI model, but necessary to building a reliable classification model. We attempt to isolate machine learning bias by distinguishing bias which arises due to the process of training the model, and pre-existing bias within the dataset.

**Full Analysis and Report:** For the complete analysis, process, and conclusion of the project, please refer to the [Project Report](https://docs.google.com/document/d/1CY4VrrwWS0iTM9ocWlA1oLV9RBF5ckchKi25X3tzC60/edit?usp=sharing).

## Running the Models

Below is the information relating to running the data for yourself, as well as how this data can be used for further research.

### Prerequisites

To run any model and obtain a classification report, install the following libraries via the command prompt:

```bash
pip install pandas
pip install sklearn
pip install matplotlib

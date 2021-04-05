# Multi-step Prediction Models for Forecasting Supply Chain Demands

This repository contains the code base for my undergraduate thesis on exploring multivariate,
multi-step forecasting models in an effort to predict supply chain demands using Unilever's shipment
data. 

## Explored Models

Completed models:

1. Vector Autoregression Model (VAR) 
2. Convolutional + Recurrent Neural Network (LSTNet)
3. Convolutional + Recurrent Neural Network for Multi-step Forecasting (MultiLSTNet)
4. Temporal Convolutional Network (TCN)

Models under consideration:

1. Attention is All You Need (Univariate Model)
2. Self Attention LSTM 
3. Temporal Pattern Attention LSTM (TPA-LSTM)

## Data
The data was cleaned and loaded into map-style Datasets as defined in `torch.utils.data`
to utilize PyTorch's data loading utilities to ease training/testing processes.

Unilever's dataset can be grouped into different levels of granularity. The models 
will be trained and tested on its ability to forecast on the Case UPC level, and on 
the category level. Thus, I've defined two Datasets: one for datasets aggregated on
the Case UPC level, `CaseUpc(Dataset)`, and one for datasets aggregated on the 
categories level, `Catergy(Dataset)`. Here, aggregated means that the time series
for all cases or categories have been concatenated together under the assumption 
that time series across datasets share similarities in temporal patterns. This 
was done to increase the dataset as each time series consist of roughly 200 to 300 
data series. Concatenation is possible since each data point is independent
of other data points temporally. 

Both `Dataset`s consist of input and labels that have been processed for 
training and testing of models. The input is of shape (batch_size, num_input, num_features),
where num_input is the number of previous weeks that are considered by the model
to forecast, and the label is of shape (batch_size, num_output, num_targets), where
num_output is the forecasting horizon (ex. if we want a 12 week forecast, num_output
= 12), and the num_targets is the number of features we are predicting.

Each `Dataset` also contains a dictionary, which maps a specific case UPC or category
to its time series data. The mapping is to another `Dataset`, which contains case
or category specific input/label data points. This was set up to easily test and
evaluate models' performances on specific cases or categories, and to perhaps train
a model on.

## Data Exploration
For feature selection and to help understand the assumption of time series sharing
similaries in temporal patterns across time series of case UPCs/categories

### Feature Visualization
![plot](figures/feature_visualizations.png)

### Feature Importance 
#### Case UPC Dataset
**Mutual Information Scores of 25 random Case UPCs**
![plot](figures/feature_importance_Case_UPC.png)

**Averaged Mutual Information Scores across all Case UPCs**
![plot](figures/feature_importance_Case_UPC_average.png)

#### Category Dataset
**Mutual Information Scores of 16 random Categories**
![plot](figures/feature_importance_Categories.png)

**Averaged Mutual Information Scores across all Categories**
![plot](figures/feature_importance_Categories_average.png)
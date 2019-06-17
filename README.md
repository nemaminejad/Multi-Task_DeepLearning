# Multi-Task_DeepLearning

This Repository contains the development of deep neural network models for binary classification using a set of tabular data. 
The deep learning framework used here is `TensorFlow`.
Two models are developed:
Two multi-task neural networks [multitask_dnnclassifier_alternate.py](https://github.com/nemaminejad/DL_Tabular_Classification/blob/master/multitask_dnnclassifier_alternate.py) and [multitask_dnnclassifier_joined.py](https://github.com/nemaminejad/DL_Tabular_Classification/blob/master/multitask_dnnclassifier_joined.py).
  - For binary classification of two target variables simultaneously using one neural network.
  - The deep learning models will extract information from both target variables in making prediction for each of the variables
  - Models are being trained in two different fashion: 1) joined, 2) alternative.


## Getting Started
### Requirements
Requirements for 
`TensorFlow`
`Pandas`
`Numpy`
`Scikit-Learn`

### Installation
`git clone  https://github.com/nemaminejad/Multi-Task_DeepLearningn`

### Data
1. Collect your data, preprocess (Normalize, handle missing data,etc.)
2. Divide data into a training set, a validation set to use for finding best architecture and best performing model, a test set for final testing of model performance.
  
3. To use multi-task models: 
  - Place data in the form of `multi_train.csv` and `multi_valid.csv`
  - Name your binary targets as `var_1`, `var_2`
  
5. for classification of two target variables simultaneously use the example script [example_run_dnn_multitask.py](https://github.com/nemaminejad/DL_Tabular_Classification/blob/master/example_run_dnn_multitask.py)
 
### Under construction
Details of the model architectures, development and performance will be added soon.

### Author
Nastaran Emaminejad
Find me in LinkedIn: https://www.linkedin.com/in/nastaran-emaminejad-791726137/
or Twitter: https://twitter.com/N_Emaminejad

### Citation
If you found my work useful for your publications, please kindly cite this repository


"Nastaran Emaminejad, Multi-Task_DeepLearning, (2019), GitHub repository, https://github.com/nemaminejad/Multi-Task_DeepLearning "

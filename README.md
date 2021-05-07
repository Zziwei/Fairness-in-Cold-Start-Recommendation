# Fairness-in-Cold-Start-Recommendation
Code for the SIGIR21 paper -- Fairness among New Items in Cold Start Recommender Systems

## Data
We put the pre-processed MovieLens 1M dataset in the 'Data' folder, where 10% (for validation) and 30% (for testing) items are randomly selected as cold-start items. Besides, we put the user embeddiongs and item embeddings from four pre-trained cold-start recommendation algorithms.

## Requirements
python 3, tensorflow 1.14.0, numpy, pandas

## Excution
The code for Gen model is in the 'Gen' folder, and the code for Scale model is in the 'Scale' folder. To run the models, excute the 'main.py' file in corresponding folders. After running the model code, you can run the jupyter notebook '/Data/ml1m/cold_bias_analysis_Debias.ipynb' to analyze the bias in the results of the models. And you can also run the jupyter notebook '/Data/ml1m/cold_bias_analysis_ColdRec.ipynb' to analyze bias in the pre-trained cold-start recommendation models.

## Thanks
Our code is based on the implementation of DropoutNet (https://github.com/layer6ai-labs/DropoutNet).

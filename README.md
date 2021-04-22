# Overview 

## Installation
```code 
- Using miniconda for better packages management. 

+ For windows user: 
    + First download: https://docs.conda.io/en/latest/miniconda.html
    + Then: Open miniconda terminal and type:
    
    conda create -n myenv python==3.8.5
    conda activate myenv
    
    conda install -c anaconda numpy
    conda install -c conda-forge matplotlib
    conda install -c conda-forge pandas
    conda install -c conda-forge seaborn
    conda install -c conda-forge scipy
    conda install -c conda-forge statsmodels
    conda install -c conda-forge scikit-learn
    conda install -c conda-forge tensorflow
    conda install -c conda-forge keras
    
    pip install git+ssh://git@github.com/thieu1995/permetrics.git
    pip install git+ssh://git@github.com/thieu1995/mealpy.git
    pip install git+ssh://git@github.com/thieu1995/opfunu.git
    
+ For unix/linux user:
    + Open terminal and type exactly the same above.
```

## Complete models
```code 
1. Traditional models
- ELM : Extreme Learning Machine
- FLNN : Functional Link Neural Network
- SSNN (Self-Structure NNs - SONIA, Immune,...)
- MLP : Multi-Layer Perceptron
- RNN : Recurrent Neural Network
- LSTM: Long Short Term Memory
- GRU: Gate Recurrent Unit
- CNN: Convolutional Neural Network - Run with sliding window > 3 only.

2. Compared models in my thesis
- SONIA
- GA-SSNN
- OCRO-SSNN
- PSO-SSNN
- WOA-SSNN
- OTWO-SSNN
- EO-SSNN
- TLO-SSNN
- SMA-SSNN
- AEO-SSNN
- IAEO-SSNN
```

## Experiment with auto-scaling techniques
- First run the script: get_real_VMs_usage.py to get the real VMs used in the system
- Second run the script: scaling_scripts.py to get ADI and SLAVTP from decision module.


### Time series and keras tutorials 
```code
https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/

https://machinelearningmastery.com/5-step-life-cycle-neural-network-models-keras/

https://machinelearningmastery.com/how-to-develop-lstm-models-for-multi-step-time-series-forecasting-of-household-power-consumption/

https://www.google.com/search?ei=aLEgXOTvB8ub_QbHq6TgDQ&q=multi-step+ahead+prediction+neural+network+machinelearningmistery&oq=multi-step+ahead+prediction+neural+network+machinelearningmistery&gs_l=psy-ab.3...20170.24731..24969...0.0..0.323.5911.2-18j3......0....1..gws-wiz.......0i22i30j33i22i29i30j33i21j33i160.3UXL8QJldlw

https://machinelearningmastery.com/how-to-make-classification-and-regression-predictions-for-deep-learning-models-in-keras/

https://www.kaggle.com/thebrownviking20/intro-to-recurrent-neural-networks-lstm-gru/notebook

https://www.kaggle.com/zoupet/neural-network-model-for-house-prices-tensorflow


https://machinelearningmastery.com/handle-missing-timesteps-sequence-prediction-problems-python/


https://machinelearningmastery.com/how-to-make-classification-and-regression-predictions-for-deep-learning-models-in-keras/
https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/

```


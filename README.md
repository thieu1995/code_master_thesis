# Overview 

## Installation
```code 
    pip install opfunu
    pip install mealpy
    pip install permetrics
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

2. Hybrid models
- All models above, except SVR and ELM, can you to hybrid with meta-heuristic algorithm. Meaning that instead of
 training with back-propagation algorithm, we can train with meta-heuristic algorithms.

- My current mealpy library contains 60 meta-heuristic algorithms. 
    https://github.com/thieunguyen5991/mealpy
- So the total number of hybrid models can be: 60 * 9 = 540 different models.
```

### Model comparison 
```code 
1. ELM
2. FLNN
3. MLP
4. SONIA
5. CNN
6. RNN
7. LSTM
8. GRU
9. GA-SONIA
10. WOA-SONIA
11. EO-SONIA
12. IEO-SONIA
```

```code 
running:

102 - traditional_rnn (RNN, LSTM, GRU), hybrid_sonia
104 : MLP, ELM, FLNN, SONIA, CNN
```


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


### Transform time series to Stationary by different (Lay dao ham)
```code
https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/
```

### Recent research
```code
https://machinelearningmastery.com/use-different-batch-sizes-training-predicting-python-keras/
https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
https://keras.io/optimizers/
https://keras.io/getting-started/sequential-model-guide/
https://www.kaggle.com/raoulma/ny-stock-price-prediction-rnn-lstm-gru
https://www.kaggle.com/thebrownviking20/intro-to-recurrent-neural-networks-lstm-gru/notebook
https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/


https://fairyonice.github.io/Stateful-LSTM-model-training-in-Keras.html
https://fairyonice.github.io/Extract-weights-from-Keras's-LSTM-and-calcualte-hidden-and-cell-states.html
https://fairyonice.github.io/Understand-Keras's-RNN-behind-the-scenes-with-a-sin-wave-example.html
```


### Self-Organizing Neural Network Characteristics
```code
0. Tên gọi chính gốc của nó là: Self-Organized Network inspired by Immune Algorithm (SONIA)
1. Nó được lấy cảm hứng từ giải thuật Immune system nhưng cách train mạng lại giống của Self-Organizing Map 
2. Thật ra nó là phiên bản gần giống với Radial Basic function neural network (Chỉ khác chỗ là RBFNN không sử dụng bias 
và activation ở output layer, còn thằng SONN thì có dùng)
3. Mạng SONN có 3 tầng, quá trình train gồm 2 giai đoạn hoàn toàn riêng biệt, gọi là : two-step algorithm
4. Weights và Biases ở tầng input và hidden được train bằng giải thuật phân cụm (Unsupervised Learning) của nó đề xuất
(giải thuật này lấy cảm hứng từ giải thuật Immune Algorithm). Weights và Biases ở tầng hidden và output được train bằng 
giải thuật lan truyền ngược 
5. Có 2 hướng cải tiến đối với mô hình mạng này:
    + Cải tiến giải thuật phân cụm (Thật ra thì giải thuật phân cụm của nó đã quá tốt, nhưng cho với problem it chiều)
    + Cải tiến giải thuật lan truyền ngược.
6. Giải thuật phân cụm thì có thể cải tiến bằng:
    + kmean, kmean++, Expectation–Maximization (Nhược điểm phải biết số lượng cụm) 
    + Mean Shift (good), DBSCAN (giải thuật này không xác định được tâm của cluster nên chưa biết cách áp dụng vào kiểu gì)
    + Kết hợp phân cụm của paper sau đó biết số cụm train lại kmean, kmean++ hoặc EM (Time lâu hơn, kết quả khả năng kém hơn)
7. Lan truyền ngược thì có rât nhiều hướng cải tiến 
    + Sử dụng GA, PSO, ...
    + Trong paper này sẽ sử dụng: Advanced Whale Optimization 
```


# Helper link
1. https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/

### Server Errors: Check lỗi multi-threading giữa numpy và openBlas
```code
Ta phải check xem core-backend của numpy nó đang dùng thư viện hỗ trợ nào : blas hay mkl
    python
    import numpy
    numpy.__config__.show()
    
https://stackoverflow.com/questions/17053671/python-how-do-you-stop-numpy-from-multithreading
https://stackoverflow.com/questions/19257070/unintented-multithreading-in-python-scikit-learn

---> Để chặn numpy không chạy multi-thread sẽ tốn thời gian trao đổi:
Thêm vào file ~/.bashrc hoặc ~/.bash_profile dòng sau:
    export OPENBLAS_NUM_THREADS=1   (Nếu dùng OpenBlas)
    export MKL_NUM_THREADS=1        (Nếu dùng MKL)

    export OPENBLAS_NUM_THREADS=1  
    export MKL_NUM_THREADS=1      
```

### Neu bi loi: Cannot share object lien quan den matplotlib tren server thi sua nhu sau:
```code
    sudo apt update
    sudo apt install libgl1-mesa-glx
```


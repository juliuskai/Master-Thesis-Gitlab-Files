## Contents  
  
    1. Intro  
        1.1 Problem & Goals  
        1.2 Structure of Thesis  
    2. Theoretical Basis & State of the research  
        2.1 Definitions of Terms used --> hyperparameters, features  
        2.2 Literature Review  
            2.2.1 Dimensionality Reduction  
            2.2.2 Models for Dimensionality Reduction  
        2.3 Autoencoders & what are they used for  
            2.3.1 Autoencoder A  
            2.3.2 Autoencoder B  
        2.4 Model Selection  
            2.4.1 Features & Hyperparameter Tuning  
            2.4.2 Validation  
    3. Methods  
        3.1 Technological Requirements/Methods  
        3.2 Data  
            3.2.1 Dataset & Machine Setup & Basic Stats
            3.2.2 Classification & Similarity Observation
        3.3 Autoencoder
            3.3.1 Architectures of Models used  
            3.3.2 Hyperparameter Tuning and Model Training --> loss plots
            3.3.3 Feaure Selection using SHAP values
            3.3.4 Comparison Methods used for different Models
        3.4 Input Optimization  
            3.4.1 BFGS
            3.4.2 Model Setup 
    4. Results  --> comparison of autoencoders and their shap values
    5. Discussion  
    6. Summary  

 



1. Introduction 
    1.1 Background                                                              1.5
    1.2 Problem Definition and Goals of the Thesis                              1
    1.3 Structure of the Thesis                                                 0.5
2. State of the Research and Literature Review                                  
    2.1 Definitions of Terms used 
    2.2 Models used in Combination with SHAP Values for Feature Selection       1.5
    2.3 Autoencoders 
        2.3.1 Simple and Deep Autoencoder                                       1.5
        2.3.2 Denoising Autoencoder                                             1.5
        2.3.3 Convolutional Autoencoder                                         1.5   
        2.3.4 Variational Autoencoder                                           1.5       
    2.4 Non-Linear Optimization Algorithms              
        2.4.1 Broyden–Fletcher–Goldfarb–Shanno Algorithm                        1.5      
        2.4.3 Nelder-Mead Method                                                1.5
        2.4.4 Conjugate Gradient Method/Gauss-Newton Method                     1.5  
    2.5 Feature Selection                   
        2.5.1 Filter Methods                                                    1.5
        2.5.2 Wrapper Methods                                                   1.5            
        2.5.3 Embedded Methods                                                  2           
    2.6 Model Selection                  
        2.6.1 Hyperparameter Tuning Methods                                     1.5        
        2.6.2 Model Evaluation Methods                                          1  
3. Methodology 
    3.1 Technological Requirements and Methods                                  1  
    3.2 Data                    
        3.2.1 Dataset and Machine Setup                                         1  
        3.2.2 Classification and Similarity Observation                         1-1.5      
    3.3 Autoencoder Models                  
        3.3.1 Architectures of Models used                                      1.5       
        3.3.2 Hyperparameter Tuning and Model Selection and Training            1.5
        3.3.3 Feature Selection using SHAP Values                               1
        3.3.4 Comparison Methods used for different Models                      1.5    
    3.4 Input Optimization 
        3.4.1 Model Setup and Architecture                                      1.5
        3.4.2 Broyden–Fletcher–Goldfarb–Shanno Setup                            1.5    
4. Results 
    4.1 Performance of Autoencoder Models                                       2   
    4.2 Evaluation of SHAP Values                                               2.5
    4.3 Evaluation of Input Optimization                                        1.5       
5. Discussion                                                                   1
6. Summary                                                                      1-1.5





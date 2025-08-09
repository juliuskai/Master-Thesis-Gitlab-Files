# Autoencoder-Based Feature Selection and Optimization for Machine Settings Enhancement

This repository contains the implementation of the methodology described in the master's thesis:

> *"Comparative Analysis of Autoencoder Approaches for Input Optimization in the Context of Machine Capability Enhancement by Ideal Setting Configuration"*

The goal is to improve machine capability by identifying and fine-tuning the most impactful machine settings that influence product quality, using advanced Autoencoder models, SHAP analysis, and optimization algorithms.

---

## 📑 Overview

The pipeline in this repository:

1. **Data Preprocessing**  
   - Load and clean manufacturing process data  
   - Normalize and handle missing values  
   - Focus on high-quality products for one-class training  

2. **Model Building**  
   - Implement and train different Autoencoder architectures:  
     - Simple Autoencoder  
     - Deep Autoencoder  
     - Denoising Autoencoder  
     - Convolutional Autoencoder  
     - Variational Autoencoder  
   - Train using one-class approach to model normal (high-quality) production data

3. **Feature Selection via SHAP**  
   - Calculate SHAP values on Autoencoder outputs  
   - Identify top influential machine settings

4. **Regression Modeling**  
   - Train regression models using:
     - Selected features (SHAP-selected subset)  
     - All features (baseline comparison)

5. **Input Optimization**  
   - Apply multiple optimization algorithms to find optimal machine settings for desired product quality  
   - Compare performance between SHAP-selected and all-feature models

6. **Evaluation**  
   - Model performance metrics  
   - Optimization outcome comparison  
   - Practical feasibility check for real-world machine settings

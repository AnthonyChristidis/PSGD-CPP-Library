#/*
* ===========================================================
* File Type: CPP
* File Name: PSGD.cpp
* Package Name: PSGD
*
* Created by Anthony-A. Christidis.
* Copyright © Anthony-A. Christidis. All rights reserved.
* ===========================================================
*/

// Libraries included 
#include <RcppArmadillo.h>
#include <iostream>

// Header files included
#include "PSGD.hpp"
#include "Stepwise_Split.hpp"
#include "config.h"

// Function to update subset of a model
void PSGD::Update_Subset_Matrix(arma::uword& group) {
  
  arma::colvec group_subset = arma::zeros(x.n_cols);
  group_subset(arma::find(betas.col(group) != 0)).ones();
  subset_indices.col(group) = group_subset;
}
void PSGD::Update_Subset_Matrix(arma::uword& group, arma::mat& subset_indices, arma::mat& betas) {
  
  arma::colvec group_subset = arma::zeros(x.n_cols);
  group_subset(arma::find(betas.col(group) != 0)).ones();
  subset_indices.col(group) = group_subset;
}

// Function to check subset of model
arma::uvec PSGD::Model_Subset(arma::uword& group) {
  
  arma::vec predictor_usage = arma::sum(subset_indices, 1) - subset_indices.col(group);
  return(arma::find(predictor_usage < split));
}
arma::uvec PSGD::Model_Subset(arma::uword& group, arma::mat& subset_indices) {
  
  arma::vec predictor_usage = arma::sum(subset_indices, 1) - subset_indices.col(group);
  return(arma::find(predictor_usage < split));
}

// Constructor
PSGD::PSGD(const arma::mat& x, const arma::vec& y, arma::uword& n_models,
           const arma::uword& model_type, const arma::uword& include_intercept,
           arma::uword split, arma::uword size,
           arma::uword max_iter,
           arma::uword cycling_iter) :
  x(x), y(y), n_models(n_models),
  model_type(model_type), include_intercept(include_intercept),
  split(split), size(size),
  max_iter(max_iter),
  cycling_iter(cycling_iter) {
  
  // Initializing rest of data in object
  subset_indices = arma::zeros(x.n_cols, n_models);
  intercepts = arma::zeros(n_models);
  betas = arma::zeros(x.n_cols, n_models);
  ensemble_loss = arma::zeros(n_models);
  
  // Fixed parameters for initialization of subsets (full diversity)
  const arma::uword MODEL_CRITERION = 2;
  const arma::uword STOP_CRITERION = 4;
  const double STOP_PARAMETER = 0.05;
  
  // Initial variables in the models
  variables_list = Stepwise_Split(x, y,
                                  n_models,
                                  size,
                                  MODEL_CRITERION,
                                  STOP_CRITERION,
                                  STOP_PARAMETER);
}

// Function to set the split of the model
void PSGD::Set_Split(arma::uword split) {
  this->split = split;
}

// Function to set the size of the model
void PSGD::Set_Size(arma::uword size) {
  this->size = size;
}

// Function to set the maximum number of iterations
void PSGD::Set_Max_Iter(arma::uword max_iter) {
  this->max_iter = max_iter;
}

// Function to initialize the ensemble model
void PSGD::Ensemble_Initialization() {
  
  // Updating the subset indices matrix
  for (arma::uword group = 0; group < n_models; group++) {
    
    arma::uvec step_model_indices = variables_list[group];
    arma::vec group_indices = arma::zeros(x.n_cols);
    group_indices(step_model_indices).ones();
    subset_indices.col(group) = group_indices; 
  }
}

// Function to compute the ensemble model
void PSGD::Compute_Ensemble() {
  
  // Create the memory for the models 
  std::vector<PS_Model*> models;
  // Initialize the models through the constructors
  for (arma::uword group = 0; group < n_models; group++)
  {
    arma::uvec model_subset = Model_Subset(group);
    models.push_back(new PS_Model(x, y, model_type, include_intercept, model_subset, size, max_iter));
    
    // Computing the coefficients
    models[group]->Set_Subset(Model_Subset(group));
    models[group]->Compute_PS();
    intercepts(group) = arma::as_scalar(models[group]->Get_Final_Intercept());
    betas.col(group) = models[group]->Get_Final_Betas();
    ensemble_loss(group) = models[group]->Get_Final_Loss();
    Update_Subset_Matrix(group);
  }
}

// Function to compute the ensemble model via cycling
void PSGD::Compute_Ensemble_Cycling() {
  
  // Current model subsets
  arma::mat subset_indices_initial = subset_indices;
  
  // Compute initial ensemble fit
  Compute_Ensemble();
  
  // COnditional on at least one cycling iteration
  if(cycling_iter>0){
    
    // Variables for cycling algorithm
    arma::vec ensemble_loss_candidate = arma::zeros(n_models);
    arma::mat subset_indices_candidate = arma::zeros(x.n_cols, n_models);
    arma::mat betas_candidate = arma::zeros(x.n_cols, n_models);
    arma::vec intercepts_candidate = arma::zeros(n_models);
    
    // Cycling iterations
    for (arma::uword cycling_count = 0; cycling_count < cycling_iter; cycling_count++) {
      
      // Order of updates
      Rcpp::IntegerVector group_order = Rcpp::sample(n_models, n_models, false)-1;
      arma::uword group_id;
      
      // Create the memory for the models 
      std::vector<PS_Model*> models; 
      // Initialize the models through the constructors
      for (arma::uword group = 0; group < n_models; group++) {
        
        arma::uvec model_subset = Model_Subset(group, subset_indices_initial);
        models.push_back(new PS_Model(x, y, model_type, include_intercept, model_subset, size, max_iter));
      }
      
      // Initial subsets for the models
      subset_indices_candidate = subset_indices_initial;
      
      // Cycle groups
      for (arma::uword group = 0; group < n_models; group++) {
        
        group_id = group_order[group];
        // Computing the coefficients
        models[group_id]->Set_Subset(Model_Subset(group_id, subset_indices_candidate));
        models[group_id]->Compute_PS();
        intercepts_candidate(group_id) = arma::as_scalar(models[group_id]->Get_Final_Intercept());
        betas_candidate.col(group_id) = models[group_id]->Get_Final_Betas();
        ensemble_loss_candidate(group_id) = models[group_id]->Get_Final_Loss();
        Update_Subset_Matrix(group_id, subset_indices_candidate, betas_candidate);
      }
      
      // std::cout << "ensemble_loss: " << arma::mean(ensemble_loss) << std::endl;
      // std::cout << "ensemble_loss_candidate: " << arma::mean(ensemble_loss_candidate) << std::endl << std::endl;
      
      // Updating the groups
      if (arma::as_scalar(arma::mean(ensemble_loss_candidate)) < arma::as_scalar(arma::mean(ensemble_loss))) {
        
        intercepts = intercepts_candidate;
        betas = betas_candidate;
        subset_indices = subset_indices_candidate;
        ensemble_loss = ensemble_loss_candidate;
      }
    }
  } // End of cycling iterations
}

// Function to return final intercept and betas
arma::vec PSGD::Get_Intercepts() {
  return(intercepts);
}
arma::mat PSGD::Get_Betas() {
  return(betas);
}
// Function to return RSS
arma::vec PSGD::Get_Ensemble_Loss() {
  return(ensemble_loss);
}

// Function to compute predictions
arma::vec PSGD::Predict(arma::mat new_data) {
  
  return(arma::as_scalar(arma::mean(intercepts)) + new_data * arma::mean(betas, 1)); 
}
// Function to compute prediction loss
double PSGD::Prediction_Loss(arma::mat new_data, arma::vec new_y){
  
  arma::vec linear_fit = Predict(new_data);
  if(model_type==1)
    return(arma::accu(arma::square(linear_fit - new_y))/(2*new_y.n_elem));
  else
    return(arma::accu(arma::log(1 + arma::exp(linear_fit)) - linear_fit % new_y)/new_y.n_elem);
}

// Destructor
PSGD::~PSGD() {}

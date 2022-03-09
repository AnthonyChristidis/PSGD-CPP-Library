#/*
* ===========================================================
* File Type: HPP
* File Name: PSGD.hpp
* Package Name: PSGD
*
* Created by Anthony-A. Christidis.
* Copyright © Anthony-A. Christidis. All rights reserved.
* ===========================================================
*/

#ifndef PSGD_hpp
#define PSGD_hpp

// Libraries included 
#include <RcppArmadillo.h>

// Header files included
#include "PS_Model.hpp"
#include "config.h"

class PSGD {
  
private:
  
  // Variables supplied by the user
  arma::mat x;
  arma::vec y;
  arma::uword n_models;
  arma::uword model_type;
  arma::uword include_intercept;
  arma::uword split;
  arma::uword size;
  arma::uword max_iter;
  arma::uword cycling_iter;
  
  // Variables created inside class
  Rcpp::List variables_list;
  arma::mat subset_indices;
  arma::vec intercepts;
  arma::mat betas;
  arma::vec ensemble_loss;
  
  // Function to update subset of a model
  void Update_Subset_Matrix(arma::uword& group);
  
  // Function to check subset of model
  arma::uvec Model_Subset(arma::uword& group);
  
public:
  
  // Constructor
  PSGD(const arma::mat& x, const arma::vec& y, arma::uword& n_models,
       const arma::uword& model_type, const arma::uword& include_intercept,
       arma::uword split, arma::uword size,
       arma::uword max_iter,
       arma::uword cycling_iter);
  
  // Function to set the split of the model
  void Set_Split(arma::uword split);
  
  // Function to set the size of the model
  void Set_Size(arma::uword size);
  
  // Function to set the maximum number of iterations
  void Set_Max_Iter(arma::uword max_iter);
  
  // Function to initialize the ensemble model
  void Ensemble_Initialization();
  // Function to compute the ensemble model
  void Compute_Ensemble();
  // Function to compute the ensemble model via cycling
  void Compute_Ensemble_Cycling();
  
  // Function to return final intercept and betas
  arma::vec Get_Intercepts();
  arma::mat Get_Betas();
  // Function to return ensemble loss
  arma::vec Get_Ensemble_Loss();
  
  // Function to compute predictions
  arma::vec Predict(arma::mat new_data);
  // Function to compute prediction loss
  double Prediction_Loss(arma::mat new_data, arma::vec new_y);
  
  // Destructor
  ~PSGD();
};

#endif // PS_Model_hpp

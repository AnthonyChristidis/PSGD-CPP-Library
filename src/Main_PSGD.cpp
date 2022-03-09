/*
 * ===========================================================
 * File Type: CPP
 * File Name: Main_PSGD.cpp
 * Package Name: PSGD
 *
 * Created by Anthony-A. Christidis.
 * Copyright © Anthony-A. Christidis. All rights reserved.
 * ===========================================================
 */
 
// Libraries included 
#include <RcppArmadillo.h>

// Header files included
#include "PSGD.hpp"
#include "config.h"

// [[Rcpp::export]]
Rcpp::List Main_PSGD(const arma::mat& x, const arma::vec& y, arma::uword& n_models,
                     const arma::uword& model_type, const arma::uword& include_intercept,
                     arma::uword split, arma::uword size,
                     arma::uword max_iter,
                     arma::uword cycling_iter){
  
  // Ensemble PSGD model (initialization with no sharing)
  PSGD ensemble_model = PSGD(x, y, n_models,
                             model_type, include_intercept, 
                             1, size,
                             max_iter,
                             cycling_iter);
  ensemble_model.Ensemble_Initialization();
  ensemble_model.Compute_Ensemble_Cycling();
  
  // Increasing until split parameter is achieved
  for(arma::uword split_param = 2; split_param <= split; split_param++){
    
    ensemble_model.Set_Split(split_param);
    ensemble_model.Compute_Ensemble_Cycling();
  }
  
  // Return ensemble data
  Rcpp::List final_ensemble;
  final_ensemble["intercepts"] = ensemble_model.Get_Intercepts();
  final_ensemble["betas"] = ensemble_model.Get_Betas();
  final_ensemble["ensemble_loss"] = ensemble_model.Get_Ensemble_Loss();
  return(final_ensemble);
}
/*
 * ===========================================================
 * File Type: CPP
 * File Name: Main_CV_PSGD.cpp
 * Package Name: PSGD
 *
 * Created by Anthony-A. Christidis.
 * Copyright (c) Anthony-A. Christidis. All rights reserved.
 * ===========================================================
 */

// Libraries included  
#include <RcppArmadillo.h>

// Header files included
#include "PSGD.hpp"
#include "Set_Diff.hpp"
#include "config.h"

// [[Rcpp::export]]
Rcpp::List Main_CV_PSGD(const arma::mat& x, const arma::vec& y, arma::uword& n_models,
                        const arma::uword& model_type, const arma::uword& include_intercept,
                        arma::uvec split_grid, arma::uvec size_grid,
                        arma::uword max_iter,
                        arma::uword cycling_iter,
                        arma::uword n_folds,
                        arma::uword n_threads) {
   
  // Variable to store the grid ensemble loss
  arma::cube prediction_loss_folds = arma::zeros(size_grid.n_elem, split_grid.n_elem, n_folds);
  
  // Storing the number of variables and observations
  const arma::uword n = x.n_rows;
  // Creating indices for the folds of the data
  const arma::uvec indin = arma::linspace<arma::uvec>(0, n - 1, n);
  const arma::uvec inint = arma::linspace<arma::uvec>(0, n, n_folds + 1);
  
  // Looping over the folds (with parallelization)
  # pragma omp parallel for num_threads(n_threads)
  for (arma::uword fold = 0; fold < n_folds; fold++){
    
    // Looping over the model splits
    for (arma::uword size_ind = 0; size_ind < size_grid.n_elem; size_ind++) {
      
      // Get test and training samples
      arma::uvec test = arma::linspace<arma::uvec>(inint[fold], inint[fold + 1] - 1, inint[fold + 1] - inint[fold]);
      arma::uvec train = Set_Diff(indin, test);
      arma::uword size_fold = ((size_grid[size_ind] < train.n_elem) ? size_grid[size_ind] : train.n_elem);
      
      // Ensemble PSGD model (initialization with no sharing)
      PSGD ensemble_model = PSGD(x.rows(train), y.elem(train), n_models,
                                 model_type, include_intercept,
                                 1, size_fold,
                                 max_iter,
                                 cycling_iter);
      ensemble_model.Ensemble_Initialization();
      ensemble_model.Compute_Ensemble_Cycling();
      prediction_loss_folds(size_ind, 0, fold) += ensemble_model.Prediction_Loss(x.rows(test), y.elem(test));
      
      // Looping over the model sizes
      for (arma::uword split_ind = 1; split_ind < split_grid.n_elem; split_ind++) {
        
        ensemble_model.Set_Split(split_grid[split_ind]);
        ensemble_model.Compute_Ensemble_Cycling();
        prediction_loss_folds(size_ind, split_ind, fold) += ensemble_model.Prediction_Loss(x.rows(test), y.elem(test));
      }
    }
  }
  
  // Adjusting prediction loss
  arma::mat prediction_loss = arma::zeros(size_grid.n_elem, split_grid.n_elem);
  for(arma::uword fold = 0; fold < n_folds; fold++)
    prediction_loss += prediction_loss_folds.slice(fold);
  prediction_loss /= n_folds;

  // Optimal split and size
  arma::uword optimal_split_ind = arma::index_min(arma::min(prediction_loss)); 
  arma::uword optimal_size_ind = arma::index_min(prediction_loss.col(optimal_split_ind));
  arma::uword optimal_split = split_grid[optimal_split_ind];
  arma::uword optimal_size = size_grid[optimal_size_ind];
  
  // Ensemble PSGD model (initialization with no sharing)
  PSGD ensemble_model = PSGD(x, y, n_models,
                             model_type, include_intercept, 
                             1, optimal_size,
                             max_iter,
                             cycling_iter);
  ensemble_model.Ensemble_Initialization();
  ensemble_model.Compute_Ensemble_Cycling();
  
  // Increasing until split parameter is achieved
  for(arma::uword split_param = 2; split_param <= optimal_split; split_param++){
    
    ensemble_model.Set_Split(split_param);
    ensemble_model.Compute_Ensemble_Cycling();
  }
  
  // Return ensemble data
  Rcpp::List final_ensemble;
  final_ensemble["prediction_loss"] = prediction_loss;
  final_ensemble["optimal_size"] = optimal_size;
  final_ensemble["optimal_split"] = optimal_split;
  final_ensemble["intercepts"] = ensemble_model.Get_Intercepts();
  final_ensemble["betas"] = ensemble_model.Get_Betas();
  final_ensemble["ensemble_loss"] = ensemble_model.Get_Ensemble_Loss();
  return(final_ensemble);
}
  
  
  
  
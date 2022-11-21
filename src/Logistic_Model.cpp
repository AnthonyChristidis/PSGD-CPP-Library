/*
 * ===========================================================
 * File Type: CPP
 * File Name: Logistic_Model.cpp
 * Package Name: PSGD
 * 
 * Created by Anthony-A. Christidis.
 * Copyright (c) Anthony-A. Christidis. All rights reserved.
 * ===========================================================
 */

 // Libraries included 
#include <RcppArmadillo.h>
#include <math.h>

// Header files included
#include "Logistic_Model.hpp"
#include "config.h"

// Constructor for Logistic_Model
Logistic_Model::Logistic_Model(arma::mat x, arma::vec y, 
         arma::uword & include_intercept,
         double alpha, double lambda_sparsity,
         double tolerance, arma::uword max_iter):  
  x(x), y(y), 
  include_intercept(include_intercept), 
  alpha(alpha), lambda_sparsity(lambda_sparsity),
  tolerance(tolerance), max_iter(max_iter){
  
  // Initializing the object
  Initialize();
}

void Logistic_Model::Initialize(){
  
  // Standardization of design matrix
  mu_x = arma::mean(x);
  sd_x = arma::stddev(x, 1);
  x_std = x;
  x_std.each_row() -= mu_x;
  x_std.each_row() /= sd_x;
  x_std_2 = x_std % x_std;
  mu_y = arma::mean(y);

  // Setting the parameters
  n = x.n_rows;
  p = x.n_cols;
  expected_val = arma::zeros(n);
  weights = arma::zeros(n);
  residuals = arma::zeros(n);
  betas = arma::zeros(p);
  new_betas = arma::zeros(p);
  intercept = 0; 

  // Setting initial values and function pointers for expected values and weights
  if (include_intercept == 1) {
      intercept = std::log(arma::mean(y) / (1 - arma::mean(y)));
      Compute_Expected_Weights = &Logistic_Model::Logistic_Update_Intercept;
  }
  else
      Compute_Expected_Weights = &Logistic_Model::Logistic_Update;
  
  // Convenience vector for soft-thresholding
  xj_y = x_std.t()*y;
  
  // Initializing the expected values, weights, and residuals
  Adjust_Expected_Weights();
}

// Functions to set new data
void Logistic_Model::Set_X(arma::mat & x){
  
  this->x = x;
  // Standardization of design matrix
  mu_x = arma::mean(x);
  sd_x = arma::stddev(x, 1);
  x_std = x;
  x_std.each_row() -= mu_x;
  x_std.each_row() /= sd_x;
  x_std_2 = x_std % x_std;
}
void Logistic_Model::Set_Y(arma::vec & y){
  this->y = y;
}

// Functions to set maximum number of iterations and tolerance
void Logistic_Model::Set_Max_Iter(arma::uword & max_iter){
  this->max_iter = max_iter;
}
void Logistic_Model::Set_Tolerance(double & tolerance){
  this->tolerance = tolerance;
}

void Logistic_Model::Adjust_Expected_Weights(){ 
  
  (this->*Compute_Expected_Weights)(this->x_std, this->new_intercept, this->new_betas, this->expected_val, this->weights);
}   
void Logistic_Model::Adjust_Residuals(){
  residuals = y - expected_val;
}

double Logistic_Model::Soft(double z, double gamma){
  
  return(((z<0) ? -1 : 1)*fmax(std::abs(z)-gamma,0));
}

void Logistic_Model::Cycle_Full_Set(){
  
  // Initial iteration over all the variables
  new_intercept = ((include_intercept) ? (intercept + (n*(mu_y-arma::mean(expected_val)))/arma::accu(weights)) : 0); 
  // Update expected values, weights and residuals if there is a change (intercept)
  if(std::fabs(new_intercept-intercept)>=EQUAL_TOLERANCE){
    Adjust_Expected_Weights();
  }
  // Initial cycle over all variables (and intercept)
  for(arma::uword j=0; j<p; j++){
    
    w_xj2 = arma::dot(x_std_2.col(j), weights); 
    new_betas[j] = Soft((xj_y[j]-arma::dot(x_std.col(j),expected_val))/n + betas[j]*w_xj2/n, alpha*lambda_sparsity) / 
      (w_xj2/n + (1-alpha)*lambda_sparsity); 
    // Update expected values, weights and residuals if there is a change (coefficients)
    if(std::fabs(new_betas[j]-betas[j])>=EQUAL_TOLERANCE){
      Adjust_Expected_Weights();
    }
  }
}

void Logistic_Model::Compute_Coef(){
  
  for(arma::uword iter=0; iter<max_iter; iter++){
    
    // Cycle over all variables
    Cycle_Full_Set();
    
    // End of coordinate descent if variables are already converged
    if(pow(arma::max(arma::abs(new_betas-betas)),2) < tolerance){
      intercept=new_intercept;
      betas=new_betas;
      Scale_Coefficients();
      Scale_Intercept();
      return;
    }

    // Adjusting the intercept and betas
    intercept=new_intercept;
    betas = new_betas;
  }
  
  // Scaling of coefficients and intercept
  Scale_Coefficients();
  Scale_Intercept();
}

// Comparison of active sets of variables after cycling
arma::uword Logistic_Model::Compare_Active_Set(arma::uvec & active_set){ 
  
  // Finding the candidate active set
  arma::uvec candidate_active_set = arma::find(new_betas!=0);
  
  // Updating the parameters
  intercept = new_intercept;
  betas = new_betas;
  
  if(candidate_active_set.n_elem!=active_set.n_elem){
    return(0);
  }
  else if(arma::accu(arma::abs(candidate_active_set-active_set)) < EQUAL_TOLERANCE){
    return(1);
  }
  else{
    return(0);
  }
}

// Functions for updates
void Logistic_Model::Logistic_Update(arma::mat& x, double& intercept, arma::vec& betas,
    arma::vec& expected_val, arma::vec& weights) {

    // Constants - COMPUTATION 
    const double DIVERGENCE_CONST = 1e-3;

    expected_val = arma::exp(x * betas) % (1 / (1 + arma::exp(x * betas)));
    weights = expected_val % (1 - expected_val);
    // Correction factor to avoid divergence
    weights.elem(find(expected_val < DIVERGENCE_CONST)).fill(DIVERGENCE_CONST);
    expected_val.elem(find(expected_val < DIVERGENCE_CONST)).zeros();
    weights.elem(find(expected_val > 1 - DIVERGENCE_CONST)).fill(DIVERGENCE_CONST);
    expected_val.elem(find(expected_val > 1 - DIVERGENCE_CONST)).ones();
}
void Logistic_Model::Logistic_Update_Intercept(arma::mat& x, double& intercept, arma::vec& betas,
    arma::vec& expected_val, arma::vec& weights) {

    // Constants - COMPUTATION 
    const double DIVERGENCE_CONST = 1e-3;

    expected_val = arma::exp(intercept + x * betas) % (1 / (1 + arma::exp(intercept + x * betas)));
    weights = expected_val % (1 - expected_val);
    // Correction factor to avoid divergence
    weights.elem(find(expected_val < DIVERGENCE_CONST)).fill(DIVERGENCE_CONST);
    expected_val.elem(find(expected_val < DIVERGENCE_CONST)).zeros();
    weights.elem(find(expected_val > 1 - DIVERGENCE_CONST)).fill(DIVERGENCE_CONST);
    expected_val.elem(find(expected_val > 1 - DIVERGENCE_CONST)).ones();
}

// Coordinate descent iterations over the active set
void Logistic_Model::Cycle_Active_Set(arma::uvec & active_set){
   
  // Initial iteration over all the variables
  new_intercept = ((include_intercept) ? (intercept + (n*(mu_y-arma::mean(expected_val)))/arma::accu(weights)) : 0); 
  // Update expected values, weights and residuals if there is a change (intercept)
  if(std::fabs(new_intercept-intercept)>=EQUAL_TOLERANCE){ 
    Adjust_Expected_Weights();
  }
  // Initial cycle over all variables (and intercept)
  for(arma::uword j=0; j<active_set.n_elem; j++){
    
    w_xj2 = arma::dot(x_std_2.col(active_set[j]), weights); 
    new_betas[active_set[j]] = Soft((xj_y[active_set[j]]-arma::dot(x_std.col(active_set[j]),expected_val))/n + betas[active_set[j]]*w_xj2/n, alpha*lambda_sparsity) / 
      (w_xj2/n + (1-alpha)*lambda_sparsity); 
    // Update expected values, weights and residuals if there is a change (coefficients)
    if(std::fabs(new_betas[active_set[j]]-betas[active_set[j]])>=EQUAL_TOLERANCE){
      Adjust_Expected_Weights();
    }
  }
}
 
void Logistic_Model::Compute_Coef_Active(){

    const double ACTIVE_SET_PRE_ITER = 2;
  
  // Initial cycles for all the variables
  for(arma::uword test_iter=0; test_iter<ACTIVE_SET_PRE_ITER; test_iter++){
  
    // Cycle over all variables
    Cycle_Full_Set();
    
    // End of coordinate descent if variables are already converged
    if(pow(arma::max(arma::abs(new_betas-betas)),2) < tolerance){
      intercept=new_intercept;
      betas=new_betas;
      Scale_Coefficients();
      Scale_Intercept();
      return;
    }
    
    // Adjusting the intercept and betas
    intercept = new_intercept;
    betas = new_betas;
  }
  
  // Vector for the active set
  arma::uvec active_set;
  arma::vec active_set_helper = arma::zeros(p);

  // Cycling over the active set
  do{
    
    // Active set for the variables
    active_set_helper.zeros(); 
    active_set_helper(arma::find(betas!=0)).fill(1);
    active_set.set_size(arma::sum(active_set_helper));
    active_set = arma::find(betas!=0);

    // Convergence around the active variables
    for(arma::uword iter=0; iter<max_iter; iter++){
      
      // Cycle over all variables
      Cycle_Active_Set(active_set);
      
      // End of coordinate descent if variables are already converged
      if(pow(arma::max(arma::abs(new_betas-betas)),2) < tolerance){
        intercept=new_intercept;
        betas=new_betas;
        break;
      }
      
      // Adjusting the intercept and betas
      intercept = new_intercept;
      betas = new_betas;
    }
    
    // Cycle over all variables
    Cycle_Full_Set();
    
  } while(Compare_Active_Set(active_set)!=1);
  
  // Scaling of coefficients and intercept
  Scale_Coefficients();
  Scale_Intercept();
  
}

// Functions to set and get alpha
void Logistic_Model::Set_Alpha(double alpha){
  this->alpha = alpha;
}
double Logistic_Model::Get_Alpha(){
  return(this->alpha);
}

// Functions to set and get lambda_sparsity
void Logistic_Model::Set_Lambda_Sparsity(double lambda_sparsity){
  this->lambda_sparsity = lambda_sparsity;
}
double Logistic_Model::Get_Lambda_Sparsity(){
  return(this->lambda_sparsity);
}

// Functions to return expected values and weights
arma::vec Logistic_Model::Get_Expected(){
  return(expected_val);
}
arma::vec Logistic_Model::Get_Weights(){ 
  return(weights);
}

// Functions to return coefficients and the intercept
arma::vec Logistic_Model::Get_Coef(){
  return(betas);
}
double Logistic_Model::Get_Intercept(){ 
  return(intercept);
}

arma::vec Logistic_Model::Get_Coef_Scaled(){
  return(betas_scaled);
}
double Logistic_Model::Get_Intercept_Scaled(){ 
  return(intercept_scaled);
}



// Function to return objective function value
double Logistic_Model::Get_Objective_Value(){
  Adjust_Residuals();
  return(arma::mean(arma::square(residuals))/2 + 
         lambda_sparsity*(((1-alpha)/2)*arma::accu(arma::square(new_betas)) + alpha*arma::accu(arma::abs(new_betas))) );
}

void Logistic_Model::Scale_Coefficients(){
  betas_scaled = betas % (1/sd_x.t());
} 

void Logistic_Model::Scale_Intercept(){
  intercept_scaled = ((include_intercept) ? 1 : 0)*(intercept - arma::accu(betas_scaled % mu_x.t()));
}

Logistic_Model::~Logistic_Model(){
  // Class destructor
}
 

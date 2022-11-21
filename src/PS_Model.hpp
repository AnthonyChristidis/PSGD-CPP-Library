/*
* ===========================================================
* File Type: HPP
* File Name: PS_Model.hpp
* Package Name: PSGD
*
* Created by Anthony-A. Christidis.
* Copyright (c) Anthony-A. Christidis. All rights reserved.
* ===========================================================
*/

#ifndef PS_Model_hpp
#define PS_Model_hpp

// Libraries included 
#include <RcppArmadillo.h>
 
// Header files included
#include "Logistic_Model.hpp"
#include "config.h"

class PS_Model {

private:

	// Variables supplied by the user
	arma::mat x;
	arma::vec y;
	arma::uword model_type;
	arma::uword include_intercept;
	arma::uvec variables_subset;
	arma::uword size;
	arma::uword max_iter;

	// Variables created inside class
	arma::mat x_subset;
	arma::rowvec mu_x_subset;
	arma::rowvec sd_x_subset;
	arma::vec betas, new_betas;
	arma::vec final_betas; arma::vec final_intercept;
	arma::vec grad_vector;
	double step_size;
	double loss_final;

	// Function to project vector
	void Project();

	// Function to compute likelihood
	double (PS_Model::* Compute_Likelihood)(arma::mat& x, arma::vec& y, arma::vec& betas);
	double Linear_Likelihood(arma::mat& x, arma::vec& y, arma::vec& betas);
	double Logistic_Likelihood(arma::mat& x, arma::vec& y, arma::vec& betas);

	// Function to compute gradient
	arma::vec (PS_Model::* Compute_Gradient)(arma::mat& x, arma::vec& y, arma::vec& betas);
	arma::vec Linear_Gradient(arma::mat& x, arma::vec& y, arma::vec& betas);
	arma::vec Logistic_Gradient(arma::mat& x, arma::vec& y, arma::vec& betas);

	// Function to adjust design data
	void Adjust_Design();

	// Function to compute final beta vector
	void Compute_Final_Betas();

public:

	// Constructor
	PS_Model(const arma::mat& x, const arma::vec& y,
			 const arma::uword& model_type, const arma::uword& include_intercept,
		 	 arma::uvec variables_subset, arma::uword size,
		 	 arma::uword max_iter);

	// Function to set the variables subset
	void Set_Subset(arma::uvec variables_subset);

	// Function to set the size of the model
	void Set_Size(arma::uword size);

	// Function to set the maximum number of iterations
	void Set_Max_Iter(arma::uword max_iter);

	// Function to compute the PS model (Must call Copmute_Final_Beta function)
	void Compute_PS(); 
	// Function to compute RSS
	void Compute_Loss();

	// Function to return final intercept and betas
	arma::vec Get_Final_Intercept();
	arma::vec Get_Final_Betas();
	// Function to return loss
	double Get_Final_Loss();

	// Destructor
	~PS_Model();
};

#endif // PS_Model_hpp

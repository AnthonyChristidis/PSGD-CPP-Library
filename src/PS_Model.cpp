#/*
* ===========================================================
* File Type: CPP
* File Name: PS_Model.cpp
* Package Name: PSGD
*
* Created by Anthony-A. Christidis.
* Copyright © Anthony-A. Christidis. All rights reserved.
* ===========================================================
*/

// Libraries included 
#include <RcppArmadillo.h> 

// Header files included
#include "PS_Model.hpp"
// #include "config.h" 
 
// Constructor Function
PS_Model::PS_Model(const arma::mat& x, const arma::vec& y,
	const arma::uword& model_type, const arma::uword& include_intercept,
	arma::uvec variables_subset, arma::uword size,
	arma::uword max_iter) :
	x(x), y(y),
	model_type(model_type), include_intercept(include_intercept),
	variables_subset(variables_subset), size(size),
	max_iter(max_iter) {

	// Setting the gradient function
	if (model_type == 1) {
		Compute_Likelihood = &PS_Model::Linear_Likelihood;
		Compute_Gradient = &PS_Model::Linear_Gradient;
	}
	else if (model_type == 2) {
		Compute_Likelihood = &PS_Model::Logistic_Likelihood;
		Compute_Gradient = &PS_Model::Logistic_Gradient;
	}

	// Setting the initial data
	final_betas.set_size(x.n_cols);
	final_betas.zeros();
	Adjust_Design();
}

// Function to project vector
void PS_Model::Project() {

	arma::uvec ranks = arma::sort_index(new_betas, "descend");
	ranks = arma::linspace<arma::uvec>(0, new_betas.n_elem - 1, new_betas.n_elem)(ranks);
	arma::uvec disabled_ranks = ranks(arma::linspace<arma::uvec>(size, new_betas.n_elem - 1, new_betas.n_elem - size));
	new_betas(disabled_ranks).zeros();
}

// Function to compute likelihood
double PS_Model::Linear_Likelihood(arma::mat& x, arma::vec& y, arma::vec& betas) {
	arma::vec linear_fit = x * betas;
	return(arma::accu(arma::square(linear_fit - y)) / (2 * y.n_elem));
}
double PS_Model::Logistic_Likelihood(arma::mat& x, arma::vec& y, arma::vec& betas) {
	arma::vec linear_fit = x * betas;
	return(arma::accu(arma::log(1 + arma::exp(linear_fit)) - linear_fit % y) / y.n_elem);
}

// Function to compute gradient
arma::vec PS_Model::Linear_Gradient(arma::mat& x, arma::vec& y, arma::vec& betas) {
	return(x.t() * (x * betas - y) / y.n_elem);
}
arma::vec PS_Model::Logistic_Gradient(arma::mat& x, arma::vec& y, arma::vec& betas) {
	return(x.t() * (1 / (1 + arma::exp(x * (-betas))) - y) / y.n_elem);
}

void PS_Model::Adjust_Design() {

	// Extracting the variables
	x_subset.set_size(arma::size(x.cols(variables_subset)));
	x_subset = x.cols(variables_subset);

	// Standardization of design matrix
	mu_x_subset = arma::mean(x_subset);
	sd_x_subset = arma::stddev(x_subset, 1);
	x_subset.each_row() -= mu_x_subset;
	x_subset.each_row() /= sd_x_subset;

	// Augmented matrix for GD
	if (include_intercept == 1)
		x_subset = arma::join_rows(arma::ones(x_subset.n_rows, 1), x_subset);
	else
		x_subset = arma::join_rows(arma::zeros(x_subset.n_rows, 1), x_subset);

	// Setting the step size for GD
	if (model_type == 1)
		step_size = 2 / std::pow(arma::max(arma::eig_sym(x_subset.t() * x_subset)), 2);
	else if (model_type == 2)
		step_size = 4 / std::pow(arma::max(arma::eig_sym(x_subset.t() * x_subset)), 2);

	// Setting the size of the GD vectors
	betas.set_size(x_subset.n_cols); betas.zeros();
	betas(arma::linspace<arma::uvec>(1, x_subset.n_cols - 1, x_subset.n_cols - 1)) = final_betas(variables_subset);
	new_betas.set_size(x_subset.n_cols); new_betas.zeros();
	grad_vector.set_size(x_subset.n_cols);
}

// Function to compute final beta vector
void PS_Model::Compute_Final_Betas() {

	final_betas.zeros();
	arma::uvec active_set = variables_subset(arma::find(new_betas(arma::linspace<arma::uvec>(1, x_subset.n_cols - 1, x_subset.n_cols - 1)) != 0));
	arma::mat x_active = x.cols(active_set);

	// Final linear fit
	if (model_type == 1) {

		arma::mat x_active_t_x_active = x_active.t() * x_active;
		final_betas(active_set) = arma::solve(x_active_t_x_active, arma::eye(arma::size(x_active_t_x_active)), arma::solve_opts::fast) * x_active.t() * y;
		final_intercept = arma::mean(y) - arma::mean(x, 0) * final_betas;
	}
	else if (model_type == 2) { // Final logistic fit

		double alpha = 1;
		double lambda_sparsity = 0;
		double tolerance = 0.01;
		arma::uword max_iter = 100;

		Logistic_Model logistic_model = Logistic_Model(x_active, y, include_intercept,
			alpha, lambda_sparsity,
			tolerance, max_iter);
		logistic_model.Compute_Coef();
		final_betas(active_set) = logistic_model.Get_Coef_Scaled();
		final_intercept = logistic_model.Get_Intercept_Scaled();
	}
}

// (+) Public Functions

// Function to set the variables subset
void PS_Model::Set_Subset(arma::uvec variables_subset) {

	this->variables_subset.set_size(arma::size(variables_subset));
	this->variables_subset = variables_subset;
	Adjust_Design();
}

// Function to set the size of the model
void PS_Model::Set_Size(arma::uword size) {
	this->size = size;
}

// Function to set the maximum number of iterations
void PS_Model::Set_Max_Iter(arma::uword max_iter) {
	this->max_iter = max_iter;
}

// Function to compute the PS model (Must call Copmute_Final_Beta function)
void PS_Model::Compute_PS() {

	const double TOLERANCE = 0.025;

	arma::uword iter_count = 0;
	betas.zeros(); new_betas.zeros();

	do {

		// GD update
		betas = new_betas;
		grad_vector = (this->*Compute_Gradient)(x_subset, y, betas);
		new_betas = betas - step_size * grad_vector;
		Project();

		// Testing whether the subsets are the same
		if (std::abs((this->*Compute_Likelihood)(x_subset, y, new_betas) - (this->*Compute_Likelihood)(x_subset, y, betas)) < TOLERANCE)
			break;

	} while (++iter_count < max_iter);

	// Computing final betas
	Compute_Final_Betas();
}

// Function to compute RSS
void PS_Model::Compute_Loss() {

	arma::vec predictions = x * final_betas + arma::as_scalar(final_intercept); 
	if (model_type == 1)
		loss_final = arma::mean(arma::square(y - predictions));
	else if (model_type == 2)
		loss_final = arma::accu(arma::square(predictions - y)) / (2 * y.n_elem);
}

// Function to return final intercept and betas
arma::vec PS_Model::Get_Final_Intercept() {
	return(final_intercept);
}
arma::vec PS_Model::Get_Final_Betas() {
	return(final_betas);
}

// Function to return RSS
double PS_Model::Get_Final_Loss() {
	Compute_Loss();
	return(loss_final);
}

// Class destructor
PS_Model::~PS_Model() {}
/*
 * ===========================================================
 * File Type: CPP
 * File Name: Step_Model.cpp
 * Package Name: PSGD
 *
 * Created by Anthony-A. Christidis.
 * Copyright © Anthony-A. Christidis. All rights reserved.
 * ===========================================================
 */

 // Libraries included
#include <RcppArmadillo.h>

// Header files included
#include "config.h"
#include "Step_Model.hpp"

// (+) Model Constructor
// 
// This constructor initializes the initial number of variables in model to zero
// Also, also allocating the memory for the number of variables that will be included in each model
Step_Model::Step_Model(arma::uword max_variables_per_model, const arma::vec& y,
                       const int& stop_criterion, const double& stop_parameter, const arma::uword& number_variables) :
                       variables_in_model(max_variables_per_model), current_design(y.n_elem, 0), current_H(y.n_elem, y.n_elem), current_res(y.n_elem), current_rss(accu(pow(y, 2))),
                       max_variables(max_variables_per_model), stop_criterion(stop_criterion), stop_parameter(stop_parameter), variables_counter(0),
                       decrease_rss(number_variables), model_full(false){
}

// (+) Functions that update the current state of the model

// Update the Design matrix
void Step_Model::UpdateDesign(const arma::mat& x) {

    current_design.insert_cols(variables_counter - 1, x.col(variables_in_model(variables_counter - 1)));
}

// Update the Hat matrix
void Step_Model::UpdateH(const arma::vec& y) {

    arma::mat cp_design = current_design.t() * current_design;
    current_H = current_design * arma::solve(cp_design, eye(size(cp_design)), arma::solve_opts::fast) * current_design.t();
}

// Update the current residuals
void Step_Model::UpdateRes(const arma::vec& y) {

    current_res = (eye(size(current_H)) - current_H) * y;
    current_rss = accu(pow(current_res, 2));
}

// Update the stoping criterion
void Step_Model::UpdateCriteria(const arma::vec& y) {
    
    F_val = optimal_rss_decrease / ((current_rss - optimal_rss_decrease) / (y.n_elem - (variables_counter + 1)));
    p_val = R::pf(F_val, 1, y.n_elem - (variables_counter + 1), 0, 1);

    switch (stop_criterion) {

    case 1: R2 = 1 - (current_rss - optimal_rss_decrease) / accu(pow(y, 2));
        break;
    case 2: aR2 = 1 - ((current_rss - optimal_rss_decrease) / (y.n_elem - (variables_counter + 1))) / (accu(pow(y, 2)) / y.n_elem - 1);
        break;
    case 3: pR2 = optimal_rss_decrease / (current_rss);
        break;
    }
}

// Functions to determine whether the model is full
void Step_Model::FixedFull() {
    if (variables_counter == max_variables)
        model_full = true;
}
void Step_Model::R2Full() {
    if (R2 >= stop_parameter || variables_counter == max_variables)
        model_full = true;
}
void Step_Model::aR2Full() {
    if (aR2 >= stop_parameter || variables_counter == max_variables)
        model_full = true;
}
void Step_Model::pR2Full() {
    if (pR2 <= stop_parameter || variables_counter == max_variables)
        model_full = true;
}
void Step_Model::FTestFull(const arma::vec& y) { // Here we expect soft_threshold to contain the p-value
    double F_crit = R::qf(1 - stop_parameter, 1, y.n_elem - (variables_counter + 1), 1, 0);
    if (F_val < F_crit || variables_counter == variables_in_model.n_elem)
        model_full = true;
}

// Member function to add a variable to the model
void Step_Model::Variable_Update(int variable_ind, const arma::mat& x, const arma::vec& y) {

    // Adding the variables to the object variables
    variables_counter++;
    variables_in_model(variables_counter - 1) = variable_ind;

    // Update the design matrix, the Hat matrix, and the current residuals
    UpdateDesign(x);
    UpdateH(y);
    UpdateRes(y);

    // Assess whether the model is full (based on number of variables in the model)
    FixedFull();
}

// (+) Functions that computes the optimal new variable that may be included in a model 
//     -> Case where this was the optimal model in previous iteration
void Step_Model::Update_Optimal_Variable_New(arma::uvec candidates, const arma::mat& x, const arma::vec& y,
                                        const bool& initialization) {

    // We only update the variable if it's not full (otherwise, we leave the data at the last time it added a variable)
    if (!Get_Full()) {

        // Vector to store the decreases in RSS for all the possible candidates
        if (!initialization)
            decrease_rss.shed_row(decrease_rss.n_elem - 1);

        // Computing the annihilator matrix
        if (variables_counter != 0) {// Case where we have already included at least one variable in the model
            arma::mat current_annihilator = eye(size(current_H)) - current_H;
            // Looping over the candidates to find the new optimal new variable
            for (arma::uword i = 0; i < candidates.n_elem; i++)
                decrease_rss(i) = pow(accu(y.t() * current_annihilator * x.col(candidates(i))), 2) / accu(x.col(candidates(i)).t() * current_annihilator * x.col(candidates(i)));
        }
        else { // Case where we haven't included a variable in the model yet
            arma::mat current_annihilator = arma::eye(x.n_rows, x.n_rows);
            // Looping over the candidates to find the new optimal new variable
            for (arma::uword i = 0; i < candidates.n_elem; i++) {
                decrease_rss(i) = pow(accu(y.t() * current_annihilator * x.col(candidates(i))), 2) / accu(x.col(candidates(i)).t() * current_annihilator * x.col(candidates(i)));
            }
        }

        // Storing the index of the biggest decrease in RSS
        optimal_variable = candidates(decrease_rss.index_max());
        optimal_rss_decrease = decrease_rss.max();
        
        // Update the variables for the stoping criterion (except for the Fixed case)
        UpdateCriteria(y);
        // Check whether the model is declared full 
        switch (stop_criterion) {

        case 1: R2Full();
            break;
        case 2: aR2Full();
            break;
        case 3: pR2Full();
            break;
        case 4: FTestFull(y);
            break;
        }
    }
}

// (+) Functions that computes the optimal new variable that may be included in a model 
//     -> Case where this was NOT the optimal model in previous iteration
void Step_Model::Update_Optimal_Variable_Check(arma::uvec candidates, const arma::mat& x, const arma::vec& y, 
                                          arma::uword previous_optimal_index, arma::uword previous_optimal) {

    // We only update the variable if it's not full (otherwise, we leave the data at the last time it added a variable)
    if (!Get_Full()) {

        // Removing the element from decrease_rss for the previous optimal variable (in another model)
        if (optimal_variable != previous_optimal) // Case where the previous optimal was not the optimal in this model
            decrease_rss.shed_row(previous_optimal_index);
        else { // Case where the previous optimal was also the optimal in this model

            // Vector to store the decreases in RSS for all the possible candidates
            decrease_rss.shed_row(decrease_rss.index_max());

            // Storing the index of the biggest decrease in RSS
            optimal_variable = candidates(decrease_rss.index_max());
            optimal_rss_decrease = decrease_rss.max();
        }

        // Update the variables for the stoping criterion (except for the Fixed case)
        UpdateCriteria(y);
        // Check whether the model is declared full 
        switch (stop_criterion) {

        case 1: R2Full();
            break;
        case 2: aR2Full();
            break;
        case 3: pR2Full();
            break;
        case 4: FTestFull(y);
            break;
        }
    }
}



// (+) Functions that return variables for the current state of the model
arma::uword Step_Model::Get_Counter() {
    return(variables_counter);
}

void Step_Model::Shed_Variables() { // (!) Only use this function at the end, it deletes the extra spaces
    if (variables_counter < max_variables)
        variables_in_model.shed_rows(variables_counter, variables_in_model.n_elem - 1);
}

arma::uvec Step_Model::Get_Variables() { // (!) Only use this function at the end, it deletes the extra spaces
    return(variables_in_model);
}

arma::mat Step_Model::Get_Design() {
    return(current_design);
}

arma::mat Step_Model::Get_H() {
    return(current_H);
}
arma::vec Step_Model::Get_Res() {
    return(current_res);
}

double Step_Model::Get_RSS() {
    return(current_rss);
}

// (+) Functions that return variables for the optimal candidate variable of the model
int Step_Model::Get_Optimal_Variable() {
    return(optimal_variable);
}

double Step_Model::Get_Optimal_RSS_Decrease() {
    return(optimal_rss_decrease);
}

double Step_Model::Get_F_val() {
    return(F_val);
}

double Step_Model::Get_p_val() {
    return(p_val);
}

// (+) Function that returns a bool value - full model
bool Step_Model::Get_Full() {
    return(model_full);
}

// Function to compute the beta regression vector
void Step_Model::Set_Final_Design(const arma::mat& x) {

    Shed_Variables();
    final_design = x.cols(variables_in_model);
}

// (+) Function to adjust the variable numbers to R convention (start at 1 [C++] rather than 0 [R])
void Step_Model::Adapt_Variables() {

    variables_in_model.for_each([](arma::uvec::elem_type& val) {val++; });
}

// (+) Model destructor
Step_Model::~Step_Model(){}






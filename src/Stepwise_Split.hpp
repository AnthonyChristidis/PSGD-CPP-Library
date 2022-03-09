/*
 * ===========================================================
 * File Type: HPP
 * File Name: Stepwise_Split.hpp
 * Package Name: PSGD
 *
 * Created by Anthony-A. Christidis.
 * Copyright © Anthony-A. Christidis. All rights reserved. 
 * ===========================================================
 */

// Libraries included
#include <RcppArmadillo.h>
#include <vector>

// Header files included
#include "Step_Model.hpp"
#include "Model_Functions.hpp"
#include "config.h"

// [[Rcpp::export]]
Rcpp::List Stepwise_Split(arma::mat x,
                          arma::vec y,
                          arma::uword n_models,
                          arma::uword max_variables_per_model,
                          const arma::uword& model_criterion,
                          const arma::uword& stop_criterion,
                          const double& stop_parameter) {

    // Standardizing the covariates
    arma::rowvec mu_x = mean(x);
    arma::rowvec sd_x = stddev(x, 1);
    // New scaled design matrix
    arma::mat x_std = x;
    x_std.each_row() -= mu_x;
    x_std.each_row() /= sd_x;
    // New centered response vector
    arma::vec y_c = y;
    double mu_y = mean(y);
    y_c = y - mu_y;

    // Storing the number of variables
    arma::uword n_var = x_std.n_cols;

    // Create the memory for the models (through dynamic allocation)
    std::vector<Step_Model*> models;
    // Initialize the models through the constructors
    for (arma::uword m = 0; m < n_models; m++)
    {
        models.push_back(new Step_Model(max_variables_per_model, y_c,
            stop_criterion, stop_parameter, x_std.n_cols));
    }

    // Let's initialize the candidates
    arma::uvec candidates = arma::linspace<arma::uvec>(0, x_std.n_cols - 1, x_std.n_cols);

    // Variables used to store the optimal model
    arma::vec models_rss_decrease(n_models);
    arma::vec models_p_val(n_models);
    arma::uword optimal_model;

    // Variable to store index of optimal variable (for the optimal model) from the candidates
    arma::uvec index_optimal(1);

    // Variable to store the number of full models
    arma::uword full_models;

    // Initializing the optimal variable for the models
    for (arma::uword m = 0; m < n_models; m++) {
        models[m]->Update_Optimal_Variable_New(candidates, x_std, y_c, true);
    }

    // Variable to store number of variables used
    arma::uword n_var_iter = 0;

    // Let's see which of the variables is the best candidate
    do {

        if (model_criterion == 1) {

            // Find the model with the optimal decrease in RSS
            for (arma::uword m = 0; m < n_models; m++) {
                if (!(models[m]->Get_Full()))
                    models_rss_decrease(m) = models[m]->Get_Optimal_RSS_Decrease();
                else
                    models_rss_decrease(m) = -1;
            }
            // Optimal model 
            optimal_model = models_rss_decrease.index_max();
        }
        else if (model_criterion == 2) {
            // Find the model with the optimal decrease in RSS
            for (arma::uword m = 0; m < n_models; m++) {
                if (!(models[m]->Get_Full()))
                    models_p_val(m) = models[m]->Get_p_val();
                else
                    models_p_val(m) = -1;
            }
            // Optimal model 
            optimal_model = models_p_val.index_min();
        }

        // Add the best variable for the optimal model
        models[optimal_model]->Variable_Update(models[optimal_model]->Get_Optimal_Variable(), x_std, y_c);
        // Remove the optimal variable from the candidates
        index_optimal = arma::find(candidates == models[optimal_model]->Get_Optimal_Variable(), 1);
        candidates.shed_row(index_optimal(0));

        // Updating the optimal variable for the non-optimal model(s)
        // Parallelization over the models? Not recommended to multi-thread in and out of R.
        for (arma::uword m = 0; m < n_models; m++)
            if (m != optimal_model && !(models[m]->Get_Full()))
                models[m]->Update_Optimal_Variable_Check(candidates, x_std, y_c, index_optimal(0), models[optimal_model]->Get_Optimal_Variable());
        // Updating the optimal variable for the optimal model
        if (!models[optimal_model]->Get_Full())
            models[optimal_model]->Update_Optimal_Variable_New(candidates, x_std, y_c, false);

        // Computing the number of full models
        full_models = 0;
        for (arma::uword m = 0; m < n_models; m++) {
            full_models += models[m]->Get_Full();
        }

        // Iterating the numbers of variables used
        n_var_iter++;

    } while (!(full_models == n_models) && (n_var_iter + 1) < n_var);
    
    // Shed extra space for variables
    for (arma::uword m = 0; m < n_models; m++) {
        models[m]->Shed_Variables();
    }

    // We return the variables in each model
    // arma::umat final_variables = Generate_Variables(models, n_models, max_variables_per_model);
    Rcpp::List final_variables = Generate_Variables_List(models, n_models);

    // Destroy the models
    for (arma::uword m = 0; m < n_models; m++) {
        delete(models[m]);
    }
    return(final_variables);
}

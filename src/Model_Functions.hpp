/*
 * ===========================================================
 * File Type: HPP
 * File Name: Model_Functions.hpp
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
// #include "config.h"
#include "Step_Model.hpp"

// Return a list of vectors with the variables in each model
Rcpp::List Generate_Variables_List(std::vector<Step_Model*> final_models,
    const arma::uword& n_models) {

    Rcpp::List final_variables(n_models);
    for (arma::uword m = 0; m < n_models; m++)
        final_variables[m] = final_models[m]->Get_Variables();

    return(final_variables);
}

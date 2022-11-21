/*
 * ===========================================================
 * File Type: HPP
 * File Name: Set_Diff.hpp
 * Package Name: StepSplitReg
 *
 * Created by Anthony-A. Christidis.
 * Copyright (c) Anthony-A. Christidis. All rights reserved.
 * ===========================================================
 */

// Libraries included
#include <RcppArmadillo.h>

// Header files included
#include "config.h"

arma::uvec Set_Diff(const arma::uvec & big, const arma::uvec & small) {

    // Find set difference between a big and a small set of variables.
    // Note: small is a subset of big (both are sorted).
    int m = small.n_elem;
    int n = big.n_elem;
    arma::uvec test = arma::uvec(n, arma::fill::zeros);
    arma::uvec zeros = arma::uvec(n - m, arma::fill::zeros);

    for (int j = 0; j < m; j++) {
        test[small[j]] = small[j];
    }

    test = big - test;
    if (small[0] != 0)
        test[0] = 1;
    
    zeros = arma::find(test != 0);
    return(zeros);
}
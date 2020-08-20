/*
 *  PIXON
 *  A Pixon-based method for reconstructing velocity-delay map in reverberation mapping.
 * 
 *  Yan-Rong Li, liyanrong@mail.ihep.ac.cn
 * 
 */

#ifndef _CONT_MODEL_HPP
#define _CONT_MODEL_HPP

#include <iostream>
#include <fstream> 
#include <vector>
#include <iomanip>
#include <cstring>
#include <cmath>
#include <random>

/* dnest header file */
#include <dnestvars.h>

#include "utilities.hpp"
#include "mathfun.h"

using namespace std;

double prob_cont(const void *model);
void from_prior_cont(void *model);
void print_particle_cont(FILE *fp, const void *model);
double perturb_cont(void *model);

class ContModel
{
  public:
    ContModel();
    ContModel(Data& cont_in);
    ~ContModel();
    void compute_mean_error();
    void mcmc();
    void recon();
    void recon(const void *model);
    void get_best_params();
    void set_covar_Umat(double sigma, double tau, double alpha);

    Data cont;   /* continuum data */
    Data cont_recon; /* continuum reconstruction */
    
    int size_max;
    double mean_error;
    double *workspace;
    double *workspace_uv;
    double *Larr_data;
    double *USmat;
    double *PEmat1, *PEmat2;
    
    int nq; 
    int num_params;
    int num_params_var;
    int num_params_drw;
    double **par_range_model;
    int *par_fix;
    double *par_fix_val;
    int *par_prior_model;
    double **par_prior_gaussian;
    double *best_params, *best_params_std;
    default_random_engine uniform_generator;
    default_random_engine normal_generator;
    uniform_real_distribution<double> *uniform_dist;
    normal_distribution<double> *normal_dist;

    DNestFptrSet *fptrset;
};

extern ContModel* cont_model;
#endif
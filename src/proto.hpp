/*
 *  PIXON
 *  A Pixon-based method for reconstructing velocity-delay map in reverberation mapping.
 * 
 *  Yan-Rong Li, liyanrong@mail.ihep.ac.cn
 * 
 */
#include "utilities.hpp"

int run(Config &cfg);

void run_pixon_uniform(Data&, Data&, double*, int, int&, int, Config&);
void run_pixon(Data&, Data&, double*, int, int&, int, Config&);
void run_cont_pixon_uniform(Data&, Data&, Data&, double *, int, int&, int, Config&);
void run_cont_pixon(Data&, Data&, Data&, double *, int, int&, int, Config&);
void run_cont_drw_uniform(Data&, Data&, Data&, double *, int, int&, int, double, double, double, Config&);
void run_cont_drw(Data&, Data&, Data&, double *, int, int&, int, double, double, double, Config&);

void test();
void test_nlopt();
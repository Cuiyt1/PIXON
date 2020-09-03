#include "utilities.hpp"

void run_pixon_uniform(Data&, Data&, double *, int, int& , int, int, double);
void run_pixon(Data&, Data&, double *, int, int&, int, int, double);
void run_cont_pixon_uniform(Data&, Data&, Data&, double *, int, int&, int, int, double);
void run_cont_pixon(Data&, Data&, Data&, double *, int, int&, int, int, double);
void run_cont_drw_uniform(Data&, Data&, Data&, double *, int, int&, int, int, double, double, double, double);
void run_cont_drw(Data&, Data&, Data&, double *, int, int&, int, int, double, double, double, double);

void test();
void test_nlopt();
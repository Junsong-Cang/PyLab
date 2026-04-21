/*
gcc gsl_integration_example.c -o tmp.out -lgsl -lgslcblas -lm;./tmp.out
*/
#include <stdio.h>
#include <math.h>
#include <gsl/gsl_integration.h>

typedef struct
{
    double u;
    double sigma;
} input_params;

double my_function(double x, void *params) 
{
    input_params *p = (input_params *) params;
    double u = p->u;
    double sigma = p->sigma;
    return 2.0*exp(-0.5 * pow((x - u)/sigma, 2.0))/(sigma*sqrt(2*M_PI));
}

int main(void) {
    gsl_integration_workspace *w = gsl_integration_workspace_alloc(1000);

    double result, error;

    gsl_function F;

    // Set parameters
    input_params p;
    p.u = 0.0;
    p.sigma = 1.0;

    F.function = &my_function;
    F.params = &p;

    double a = 0.0;
    double b = 10.0;

    gsl_integration_qag(&F, a, b,
                        0,
                        1e-7,
                        1000,
                        GSL_INTEG_GAUSS15,
                        w, &result, &error);

    printf("Result = %.10f\n", result);
    printf("Estimated error = %.10f\n", error);

    gsl_integration_workspace_free(w);
    return 0;
}

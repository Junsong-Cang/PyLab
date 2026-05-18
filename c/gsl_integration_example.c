/*
gcc gsl_integration_example.c -o tmp.out -lgsl -lgslcblas -lm;./tmp.out
*/
#include <stdio.h>
#include <math.h>
#include <gsl/gsl_integration.h>
#define x_len 100000

typedef struct
{
    double p1;
    double p2;
    long n_calls;
} input_params;

double my_function(double x, void *params)
{
    input_params *p = (input_params *)params;
    double p1 = p->p1;
    double p2 = p->p2;
    p->n_calls++;
    // return 2.0 * exp(-0.5 * pow((x - p1) / p2, 2.0)) / (p2 * sqrt(2 * M_PI)); // Gaussian
    return sin(p1*x + p2); // Easier to integrate analytically, can be used to check results
}

// Now define some useful functions for numerical integration
void linspace(double xmin, double xmax, double *x, int nx)
{
    /*
    Create a linspace array
    -- inputs --
    xmin: minimum of x
    xmax: maximum of x
    x: pointer of pre-created x array
    nx: array size
    */
    int idx;
    double dx;
    dx = (xmax - xmin) / ((double)nx - 1.0);
    for (idx = 0; idx < nx; idx++)
    {
        x[idx] = xmin + ((double)idx * dx);
    }
}

double Integrate(double *x, double *fx, int nx)
{
    /*
    Integrate fx over x: \int dx f(x)
    -- inputs --
    method: integration method. For equally spaced x methods 0 and 1 are pretty much the same, but trapz generally outperforms for non-equally spaced x
        0 - summation, simplest method
            \int dx f(x) = \sum_{i=0}^{n-1}f_idx_i
        1 - trapz
    */
    double dx, result, f;
    int idx;
    result = 0.0;
    for (idx = 0; idx < nx - 1; idx++)
    {
        dx = x[idx + 1] - x[idx];
        f = (fx[idx] + fx[idx + 1]) / 2.0;
        result += f * dx;
    }
    return result;
}

int main(void)
{
    // ---- GSL ----
    gsl_integration_workspace *w = gsl_integration_workspace_alloc(1000);
    double Result_GSL, error;
    gsl_function F;
    // Set parameters
    input_params params;
    params.p1 = 1.0;
    params.p2 = 0.0;
    params.n_calls = 0;
    F.function = &my_function;
    F.params = &params;

    double xmin = 0.0;
    double xmax = M_PI;

    gsl_integration_qag(&F, xmin, xmax,
                        0,
                        1e-7,
                        1000,
                        GSL_INTEG_GAUSS15,
                        w, &Result_GSL, &error);
    gsl_integration_workspace_free(w);
    printf("GSL Function calls = %ld\n", params.n_calls);

    // ---- Numerical ----
    double xax[x_len], Integrand[x_len], Result_Numerical;
    int idx;
    linspace(xmin, xmax, xax, x_len);
    for (idx=0; idx<x_len;idx++)
    {
        Integrand[idx] = my_function(xax[idx], &params);
    }
    Result_Numerical = Integrate(xax, Integrand, x_len);
    
    // ---- Analytic ----
    // Do the integration yourself
    double Result_analytic = 2.0;

    // Print summaries
    printf("GSL Result = %.10f\n", Result_GSL);
    printf("Estimated error = %.10f\n", error);
    
    printf("Numerical: %.10f\n", Result_Numerical);
    printf("Analytic: %.10f\n", Result_analytic);
    printf("\n");
    printf("Numerical error: %10E\n", fabs(Result_Numerical-Result_analytic)/Result_analytic);
    printf("GSL error: %10E\n", fabs(Result_GSL-Result_analytic)/Result_analytic);
    
    return 0;
}

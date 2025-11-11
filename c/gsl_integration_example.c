#include <stdio.h>
#include <math.h>
#include <gsl/gsl_integration.h>

// Define the integrand function
double my_function(double x, void *params) {
    return exp(x);
}

int main(void) {
    gsl_integration_workspace *w = gsl_integration_workspace_alloc(1000);

    double result, error;

    // Define the gsl_function structure
    gsl_function F;
    F.function = &my_function;
    F.params = NULL;

    // Integration limits
    double a = 0.0;
    double b = M_PI;

    // Perform the integration
    gsl_integration_qag(&F, a, b,
                        0,              // absolute error
                        1e-7,           // relative error
                        1000,           // workspace limit
                        GSL_INTEG_GAUSS15, // integration rule (15-point Gauss-Kronrod)
                        w, &result, &error);

    printf("Result = %.10f\n", result);
    printf("Estimated error = %.10f\n", error);

    gsl_integration_workspace_free(w);
    printf("%10E\n", M_PI);
    return 0;
}

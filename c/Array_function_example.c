/* 
C-Python function examples that take array as input and return array
*/
#include<math.h>
#include<stdio.h>
#define test_len 10

void Example_1(double *results, int nx, double arg)
{
    int idx;
    for (idx=0; idx < nx; idx++)
    {
        results[idx] = arg;
    }
}

void Example_2(double *results, double *x, int nx, double arg)
{
    int idx;
    for (idx=0; idx < nx; idx++)
    {
        results[idx] = sin(arg*x[idx]);
    }
}

/*
int main()
{
    int n, idx;
    n = test_len;
    double x[test_len], y[test_len];

    for (idx=0; idx<n; idx++)
    {
        printf("x = %15E\n", x[idx]);
    }

    // Check that the result has been changed
    Example_1(x, n, 3.14);
    for (idx=0; idx<n; idx++)
    {
        printf("x = %7E\n", x[idx]);
    }
    
    // Function array
    printf("==== Break Point ====\n");
    for (idx=0; idx<n; idx++)
    {
        x[idx] = 0.1 * idx;
    }
    //for (idx=0; idx<n; idx++)
    //{
    //    printf("x = %7E\n", x[idx]);
    //}
    Example_2(y, x, n, 2.0);
    for (idx=0; idx<n; idx++)
    {
        printf("x = %7E, y = %7E\n", x[idx], y[idx]);
    }

    return 0;
}
*/
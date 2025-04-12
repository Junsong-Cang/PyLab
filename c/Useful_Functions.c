/* Some useful c functions
here is an example of python interface

1. compile library by one of the following:
    gcc -shared -o Useful_Functions.so Useful_Functions.c
    icc -shared -o Useful_Functions.so -fPIC Useful_Functions.c

2. if you want HaloProfile_Kernel, at your python script write:
import ctypes
def Halo_Profile_c(z, m, r, type):
    c_lib = ctypes.CDLL('/Users/cangtao/cloud/Library/PyLab/c/Useful_Functions.so')
    # Declare the function signature
    Double = ctypes.c_double
    Int = ctypes.c_int
    # specify the name of function you want
    c_function = c_lib.HaloProfile_Kernel
    # set input type
    c_function.argtypes = (Double, Double, Double, Int)
    # set result type
    c_function.restype = Double
    # Call the C function
    result = c_function(z, m, r, type)
    return result
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int Find_Index(double *x_axis, double x, int nx)
{
    /*
    Find the index of closest left element, need this for interpolation
    range handle:
        if x is on the LEFT of x_axis[0] : return -1
        if x is on the RIGHT of x_axis[nx-1] : return nx
    */
    double x1, x2, x3;
    int id1, id2, id3, Stop, s1, s2, s3, idx, count, reversed;
    id1 = 0;
    id3 = nx - 1;
    Stop = 0;
    x1 = x_axis[id1];
    x3 = x_axis[id3];
    reversed = x1 < x3 ? 0 : 1;
    if (!reversed)
    {
        if (x < x1)
        {
            Stop = 1;
            idx = -1;
        }
        if (x > x3)
        {
            Stop = 1;
            idx = nx;
        }
    }
    else
    {
        // printf("x1 = %f, x3 = %f\n", x1, x3);
        if (x > x1)
        {
            Stop = 1;
            idx = -1;
        }
        if (x < x3)
        {
            Stop = 1;
            idx = nx;
        }
    }

    count = 0;
    while (Stop == 0)
    {
        count = count + 1;
        id2 = (int)round((((double)(id1 + id3))) / 2.0);
        if (id3 == id1 + 1)
        {
            idx = id1;
            Stop = 1;
        }

        x1 = x_axis[id1];
        x2 = x_axis[id2];
        x3 = x_axis[id3];

        if (!reversed)
        {
            if (x < x2)
            {
                id3 = id2;
            }
            else
            {
                id1 = id2;
            }
        }
        else
        {
            if (x < x2)
            {
                id1 = id2;
            }
            else
            {
                id3 = id2;
            }
        }
        if (count > 100)
        {
            fprintf(stderr, "Error: solution not found after 100 iterations.\n");
            exit(1);
        }
    }

    // printf("Stopping, id1 = %d, id3 = %d, x1 = %f, x = %f, x3 = %f, idx = %d\n", id1, id3, x1, x, x3, idx);

    return idx;
}

double Interp_1D(double x, double *x_axis, double *y_axis, int nx, int Use_LogX, int Use_LogY, int Overflow_Handle)
{
    /* Find value of y at x
    Use_LogX : whether to use log axis for x
    Use_LogY : whether to use log axis for y
    Overflow_Handle : what to do if x is not in x_axis
                      0 : raise error and exit
                      1 : give nearest value
    */
    int id1, id2;
    double x1, x2, y1, y2, x_, r, Small;
    id1 = Find_Index(x_axis, x, nx);
    Small = 1e-280;

    if (id1 == -1)
    {
        if (Overflow_Handle == 1)
        {
            r = y_axis[0];
        }
        else
        {
            fprintf(stderr, "Error from Interp_1D: x is not in range, axis range: [%E   %E], x = %E\n", x_axis[0], x_axis[nx - 1], x);
            exit(1);
        }
    }
    else if (id1 == nx)
    {
        if (Overflow_Handle == 1)
        {
            r = y_axis[nx - 1];
        }
        else
        {
            fprintf(stderr, "Error from Interp_1D: x is not in range, axis range: [%E   %E], x = %E\n", x_axis[0], x_axis[nx - 1], x);
            exit(1);
        }
    }
    else
    {
        id2 = id1 + 1;
        if (!Use_LogX)
        {
            x1 = x_axis[id1];
            x2 = x_axis[id2];
            x_ = x;
        }
        else
        {
            // Detect negative element
            x1 = x_axis[id1];
            x2 = x_axis[id2];
            if (((x1 < 0) || (x2 < 0)) || (x < 0))
            {
                fprintf(stderr, "cannot use LogX for axis or x with negative element\n");
                exit(1);
            }

            x1 = log(x1);
            x2 = log(x2);
            x_ = log(x);
        }
        y1 = y_axis[id1];
        y2 = y_axis[id2];

        if (Use_LogY)
        {
            // This is to avoid nan at log
            if ((y1 < 0) || (y2 < 0))
            {
                fprintf(stderr, "cannot use LogY for axis with negative element\n");
                exit(1);
            }

            y1 = y1 > Small ? y1 : Small;
            y2 = y2 > Small ? y2 : Small;
            y1 = log(y1);
            y2 = log(y2);
        }

        r = (y2 - y1) * (x_ - x1) / (x2 - x1) + y1;

        if (Use_LogY)
        {
            r = exp(r);
        }
        // printf("x_ = %f, x1 = %f, x2 = %f, y1 = %f, y2 = %f\n", x_, x1, x2, y1, y2);
    }

    return r;
}

int main()
{
    // testing precision of Interp_1D, don't forget to add these includes when running outside 21cmFAST env:
    // #include <stdio.h>
    // #include <stdlib.h>
    // #include <math.h>
    // conclusion: works well with:
    //     vec size = filled size
    //     vec size != filled size
    //     reversed vec
    //     LogY
    //
    int nx, idx, nx2;
    double x_vec[1000], y_vec[1000], x1, x2, dx, x, y;
    FILE *OutputFile;
    nx = 100;
    x1 = 10;
    x2 = -4;
    dx = (x2 - x1) / ((double)nx - 1);

    OutputFile = fopen("tmp_1.txt", "w");

    for (idx = 0; idx < nx; idx++)
    {
        x_vec[idx] = x1 + ((double)idx) * dx;
        y_vec[idx] = cos(x_vec[idx]);
        fprintf(OutputFile, "%E  %E\n", x_vec[idx], y_vec[idx]);
        if ((idx == 0) || (idx == nx - 1))
        {
            printf("%f\n", x_vec[idx]);
        }
    }
    fclose(OutputFile);

    nx2 = 90;
    x1 = 3.2;
    x2 = 6.32;
    dx = (x2 - x1) / ((double)nx2 - 1);
    OutputFile = fopen("tmp_2.txt", "w");

    for (idx = 0; idx < nx2; idx++)
    {
        x = x1 + ((double)idx) * dx;
        // printf("%d  %f\n", idx, x);
        y = Interp_1D(x, x_vec, y_vec, nx, 0, 1, 0);
        fprintf(OutputFile, "%E  %E\n", x, y);
    }
    fclose(OutputFile);

    return 0;
}

double HaloProfile_Kernel(double z, double mh, double r, int ProfileType)
{
    /* Halo density profile
    ---- inputs ----
    z : redshift
    mh : halo mass in msun, dm+baryon
    r : distance to center in pc
    ProfileType : result type
                  0 - Rho_DM in msun/pc^3
                  1 - viral radius in pc
    */
   
   // printf("Check that the integrated mass converges to m\n");
   double OmM, OmC, m, OmR, OmL, h, pi, rho_cr0, zp, OmMz, d, Delta_C, log10_c, c, delta_c, rv1, rv2, rv3, r_vir, x, cx, rho_cr, RhoDM;
   
   // Some settings
    OmM = 0.30964168161;
    OmR = 9.1e-5;
    OmC = 0.260667;
    OmL = 0.69026731839;
    h = 0.6766;
    pi = 3.141592653589793;
    rho_cr0 = 2.775e-7 * pow(h, 2.);// critical density in msun/pc^3
    
    // Pre-requisites
    m = mh * OmC/OmM; // DM mass
    zp = 1. + z;
    OmMz = OmM * pow(zp, 3.) /(OmM * pow(zp, 3.) + OmL);
    d = OmMz - 1.;
    Delta_C = 18. * pow(pi, 2.) + 82. * d - 39. * pow(d, 2.);
    log10_c = 1.071 - 0.098 * (log10(m) - 12.);
    c = pow(10., log10_c)/zp ; // concentration, see appdx.A of Zip.et for the additional (1+z) factor
    delta_c = Delta_C * pow(c, 3.) / (3. * (log(1. + c) - c/(1. + c)));

    rv1 = 0.784 * pow(m * h/1.0e8, 1./3.);
    rv2 = pow(OmM * Delta_C / (OmMz * 18. * pow(pi, 2.)), -1./3.);
    rv3 = (10./zp)/h * 1000.;
    r_vir = rv1 * rv2 * rv3;

    x = r/r_vir;
    if (x > 1.)
    {
        RhoDM = 0.;
        // printf("%E  %E\n", r, r_vir);
    }
    else
    {
        cx = c * x;
        // rho_cr = rho_cr0 * (OmL + OmM * zp**3 + OmR * zp**4);
        rho_cr = rho_cr0 * (OmL + OmM * pow(zp, 3.) + OmR * pow(zp, 4.));
        RhoDM = rho_cr * delta_c /(cx * pow(1. + cx, 2.));
    }
    if (ProfileType == 0)
    {
        return r_vir;
    }
    else
    {
        return RhoDM;
    }
}

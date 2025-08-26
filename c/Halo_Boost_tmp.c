// Halo Boost Factor module, can be run and testes outside 21cmFAST

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define HaloProfile_Length 2000
#define Boost_nmh 1000
#define PBH_Table_Size 100

double HaloProfile_Kernel(double z, double mh, double r, int type)
{
    double h, OmM, OmL, OmB, Grav, pc, zp, pi, c, Omega_m_z, d, Delta_c, F, r_vir, p1, p2, p3, RhoM, H, rho_c, delta_c, x;
    h = 0.6766;
    OmM = 0.30964168161;
    OmB = 0.04897468161;
    OmL = 1.0 - OmM;
    Grav = 6.67259e-8;                     //  cm^3/g/s^2
    pc = 3.26 * 3e10 * 365 * 24 * 60 * 60; // cm
    zp = 1.0 + z;
    pi = 3.141592653589793;

    c = pow(10, (1.071 - 0.098 * (log10(mh) - 12))) / zp; // Concentration

    Omega_m_z = OmM * pow(zp, 3.0) / (OmM * pow(zp, 3.0) + OmL);

    d = Omega_m_z - 1;
    Delta_c = 18.0 * pow(pi, 2.0) + 82.0 * d - 39.0 * pow(d, 2.0);

    F = log(1 + c) - c / (1 + c);

    // Get r_vir
    p1 = 0.784 * pow((mh / 1e8 * h), 1.0 / 3.0);
    p2 = pow((OmM / Omega_m_z * Delta_c / 18 / (pi * pi)), -1.0 / 3.0);
    p3 = (10.0 / zp / h) * 1.e3;
    r_vir = p1 * p2 * p3;

    if (r > r_vir)
    {
        RhoM = 0;
    }
    else
    {
        H = h * 100.0 * sqrt(OmL + OmM * pow(zp, 3.0));
        rho_c = 3.0 * pow(H * 1.0e5 / 1.0e6 / pc, 2.0) / (8 * pi * Grav);
        rho_c = rho_c / 1.989e33 * pow(pc * 1.0e6, 3.0);
        delta_c = Delta_c / 3 * pow(c, 3.0) / F;
        x = r / r_vir;
        RhoM = rho_c * delta_c / (c * x * pow(1 + c * x, 2.0));
        RhoM = RhoM / 1.0e18;
    }
    if (type == 0)
    {
        return RhoM;
    }
    else
    {
        return r_vir;
    }
}

int main()
{
    int nr = 1000, idx;
    double z, mh, r, rho, r1, r2, dr;

    z = 6;
    mh = 1.0e9;
    r1 = 40.;
    r2 = 5000.;
    dr = (r2 - r1) / ((double)(nr - 1));
    r = r1;

    for (idx = 0; idx < nr; idx++)
    {
        rho = HaloProfile_Kernel(z, mh, r, 0);
        // printf("%E  %E\n", r, rho);
        r = r + dr;
    }

    r = fmax(1.0, 3.1);
    printf("%E\n", r);

    return 0;
}

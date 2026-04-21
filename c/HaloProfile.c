#include <math.h>
#include <stdio.h>
#define Use_Conde_Concentration 0

struct Constants
{
    double kB;
    double OmM;
    double OmB;
    double OmL;
    double G;
    double kpc;
    double mp;
    double h;
    double msun;
    double rho_cr;
    double pc;
    double Mpc;
};

struct Constants CST = {
    .OmB = 0.04897468161,
    .OmM = 0.30964168161,
    .OmL = 1-0.30964168161,
    .h = 0.6766,
    .kB = 1.38064852E-23,
    .G = 6.6740831313131E-11,
    .kpc = 3.086E19,
    .mp = 1.67262158e-27,
    .msun = 1.98847E30,
    .rho_cr = 8.6018282524e-27,
    .pc = 3.086E16,
    .Mpc = 3.086E22,
};

double OMZ_fun(double z)
{
    double OmM, OmL, r;
    // OmM = 0.30964168161;
    // OmL = 1.0 - OmM;
    OmM = CST.OmM;
    OmL = CST.OmL;
    r = OmM*pow(1.0 + z, 3.0) / (OmM*pow(1.0 + z, 3.0) + OmL);
    return r;
}

double DeltaC_fun(double z, double OMZ)
{
    double d, r;
    d = OMZ - 1.0;
    r = 18.0 * pow(M_PI, 2.0) + 82.0*d - 39.0*pow(d, 2.0);
    return r;
}

double Tvir_fun(double z, double m, double h)
{
    double OMZ, r, DeltaC, u, OmM;
    OMZ = OMZ_fun(z);
    // OmM = 0.30964168161;
    OmM = CST.OmM;
    DeltaC = DeltaC_fun(z, OMZ);
    u = 1.22;
    r = 1.98E4 * (u/0.6) * pow(m*h/1.0E8, 2.0/3.0);
    r *= pow(OmM * DeltaC/(OMZ * 18.0 * pow(M_PI, 2.0)), 1.0/3.0) * (1.0+z)/10.0;
    return r;
}

double Rvir_fun(double z, double m, double h)
{
    double OMZ, OmM, DeltaC, result, kpc;
    OMZ = OMZ_fun(z);
    DeltaC = DeltaC_fun(z, OMZ);
    // OmM = 0.30964168161;
    // kpc = 3.086E19;
    OmM = CST.OmM;
    kpc = CST.kpc;
    result = 0.784 * pow(m*h/1.0E8, 1.0/3.0) * pow(OmM * DeltaC/(OMZ*18.0*M_PI*M_PI), -1.0/3.0) * (10.0/(1.0+z)) /h * kpc;
    return result;
}

double F_fun(double x)
{
    return log(1.0+x) - x/(1.0+x);
}

double Halo_Concentration_fun(double m, double z, double h)
{
    double c0, c1, c2, c3, c4, c5, x, r;
    double LgC, C_zip;
    
    c0 = 37.5153;
    c1 = -1.5093;
    c2 = 1.636E-2;
    c3 = 3.66E-4;
    c4 = -2.89237E-5;
    c5 = 5.32E-7;
    x = log(m*h);
    r = c0 + c1*x + c2*pow(x, 2.0)+ c3*pow(x, 3.0)+ c4*pow(x, 4.0)+ c5*pow(x, 5.0);
    r /= 1.0+z;
    if (! Use_Conde_Concentration)
    {// Zip's model
        LgC = 1.071 - 0.098 * (log10(m) - 12.0);
        r = pow(10.0, LgC)/(1.0+z);
    }
    return r;
}

double VE2_fun(double x, double m, double z, double h)
{
    double C, G, msun, vcv2, R_vir, vc2, result, cx, small;
    // G = 6.6740831313131E-11;
    // msun = 1.98847E30;
    G = CST.G;
    msun = CST.msun;
    small = 1.0E-10;
    C = Halo_Concentration_fun(m, z, h);
    R_vir = Rvir_fun(z, m, h);
    vcv2 = G*m*msun/R_vir;
    cx = C * x;
    vc2 = vcv2 * F_fun(cx)/F_fun(C);
    result = 2.0 * vc2 * (F_fun(cx) + cx/(1 + cx))/(x * F_fun(cx));
    if (x < small)
    {
        result = 2 * vcv2 * C/F_fun(C);
    }
    return result;
}

double Halo_Baryon_Profile_kernel(double z, double m, double x)
{
    double h, u, mp, kB, ve20, ve2, result, Tvir;
    u = 1.22;
    // h = 0.6766;
    // mp = 1.67262158e-27;
    // kB = 1.38064852E-23;
    h = CST.h;
    mp = CST.mp;
    kB = CST.kB;
    ve20 = VE2_fun(0.0, m, z, h);
    ve2 = VE2_fun(x, m, z, h);
    Tvir = Tvir_fun(z, m, h);
    result = - u * mp * (ve20 - ve2)/(2.0 * kB * Tvir);
    result = exp(result);
    return result;
}

void Halo_Baryon_Profile(double z, double m, double *xax, double *rax, double *result, int nx)
{
    int idx;
    double x, dx, Integ, OmB, OmM, h, Rvir, rho0, msun;
    Integ = 0.0;
    for (idx=0; idx<nx; idx++)
    {
        x = xax[idx];
        result[idx] = Halo_Baryon_Profile_kernel(z, m, x);
        dx = idx==nx-1? 0.0 : xax[idx+1] - xax[idx];
        Integ += result[idx] * pow(x, 2.0) * dx;
    }

    // OmB = 0.04897468161;
    // OmM = 0.30964168161;
    // h = 0.6766;
    // msun = 1.98847E30;

    OmB = CST.OmB;
    OmM = CST.OmM;
    h = CST.h;
    msun = CST.msun;
    
    Rvir = Rvir_fun(z, m, h);
    rho0 = m * OmB / (4 * M_PI * Integ * OmM * pow(Rvir, 3.0));
    for (idx=0; idx<nx; idx++)
    {
        rax[idx] = xax[idx] * Rvir;
        result[idx] = rho0 * result[idx] * msun;
    }
}

double Halo_DM_Profile_Kernel(double z, double mh, double r, int ProfileType)
{
    /* Halo density profile
    ---- inputs ----
    z : redshift
    mh : halo mass in msun, dm+baryon
    r : distance to center in pc
    ProfileType : result type
                  0 - viral radius in pc
                  1 - Rho_DM in msun/pc^3
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
    rho_cr0 = 2.775e-7 * pow(h, 2.); // critical density in msun/pc^3

    // Pre-requisites
    m = mh * OmC / OmM; // DM mass
    zp = 1. + z;
    OmMz = OmM * pow(zp, 3.) / (OmM * pow(zp, 3.) + OmL);
    d = OmMz - 1.;
    Delta_C = 18. * pow(pi, 2.) + 82. * d - 39. * pow(d, 2.);
    // log10_c = 1.071 - 0.098 * (log10(m) - 12.);
    // c = pow(10., log10_c) / zp; // concentration, see appdx.A of Zip.et for the additional (1+z) factor
    c = Halo_Concentration_fun(mh, z, h);
    delta_c = Delta_C * pow(c, 3.) / (3. * (log(1. + c) - c / (1. + c)));

    rv1 = 0.784 * pow(mh * h / 1.0e8, 1. / 3.);
    rv2 = pow(OmM * Delta_C / (OmMz * 18. * pow(pi, 2.)), -1. / 3.);
    rv3 = (10. / zp) / h * 1000.;
    r_vir = rv1 * rv2 * rv3;

    x = r / r_vir;
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
        RhoDM = rho_cr * delta_c / (cx * pow(1. + cx, 2.));
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

void Halo_DM_Profile(double z, double m, double *xax, double *rax, double *result, int nx)
{
    /* Halo density profile
    */
    double Rvir, pc, Mpc, msun, SI_unit;
    int idx;
    pc = CST.kpc/1000.0;
    Mpc = pc * 1.0E6;
    msun = CST.msun;
    
    SI_unit = msun / pow(pc, 3.0);
    Rvir = Halo_DM_Profile_Kernel(z, m, 0.0, 0) * pc;

    for (idx = 0; idx<=nx; idx++)
    {
        rax[idx] = Rvir * xax[idx];
        result[idx] = Halo_DM_Profile_Kernel(z, m, rax[idx]/pc, 1) * SI_unit;
    }
}

double HaloProfile_Integrator_DM(double z, double mh)
{
    /* Do the following HaloProfile integration analytically:
    \int dr r^2 \rho^2_{dm}
    ---- inputs ----
    z : redshift
    mh : halo mass in msun, dm+baryon
    ---- output unit ----
    msun^2/Mpc^3
    */

    // printf("Check that the integrated mass converges to m\n");
    double OmM, OmC, OmR, OmL, h, pi, rho_cr0, zp, OmMz, d, Delta_C, log10_c, c, delta_c, rv1, rv2, rv3, r_vir, rho_cr, res;

    // Some settings
    OmM = 0.30964168161;
    OmR = 9.1e-5;
    OmC = 0.260667;
    OmL = 0.69026731839;
    h = 0.6766;
    pi = 3.141592653589793;
    rho_cr0 = 2.775e-7 * pow(h, 2.); // critical density in msun/pc^3

    // Pre-requisites
    zp = 1. + z;
    OmMz = OmM * pow(zp, 3.) / (OmM * pow(zp, 3.) + OmL);
    d = OmMz - 1.;
    Delta_C = 18. * pow(pi, 2.) + 82. * d - 39. * pow(d, 2.);
    c = Halo_Concentration_fun(mh, z, h);
    delta_c = Delta_C * pow(c, 3.) / (3. * (log(1. + c) - c / (1. + c)));

    rv1 = 0.784 * pow(mh * h / 1.0e8, 1. / 3.);
    rv2 = pow(OmM * Delta_C / (OmMz * 18. * pow(pi, 2.)), -1. / 3.);
    rv3 = (10. / zp) / h * 1000.;
    r_vir = rv1 * rv2 * rv3;                                          // pc
    rho_cr = rho_cr0 * (OmL + OmM * pow(zp, 3.) + OmR * pow(zp, 4.)); // msun/pc^3

    res = pow(rho_cr * delta_c, 2.0);
    res = res * pow(r_vir / c / (1. + c), 3.0);
    res = res / 3.0 * (pow(1. + c, 3.) - 1.0);
    res = res * 1.0e18; // convert to msun^2/Mpc^3

    return res;
}

int main()
{
    printf("Hello World!\n");
    // printf("%10E\n", M_PI_2);
    return 0;
}

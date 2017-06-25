#define real double

#include <vector_functions.hpp>
#include "vectorOps.cu"
#include "twobodyForcePolynomial.cu"

typedef struct {
    double x, y, z;
    double fx, fy, fz;
} AtomData;

#define CAL2JOULE 4.184
#define Oa  0
#define Ha1 1
#define Ha2 2
#define Ob  3
#define Hb1 4
#define Hb2 5
#define Xa1 6
#define Xa2 7
#define Xb1 8
#define Xb2 9

#define k_HH_intra -6.480884773303821e-01 // A^(-1)
#define k_OH_intra  1.674518993682975e+00 // A^(-1)
#define k_HH_coul 1.148231864355956e+00 // A^(-1)
#define k_OH_coul 1.205989761123099e+00 // A^(-1)
#define k_OO_coul 1.395357065790959e+00 // A^(-1)
#define k_XH_main 7.347036852042255e-01 // A^(-1)
#define k_XO_main 7.998249864422826e-01 // A^(-1)
#define k_XX_main 7.960663960630585e-01 // A^(-1)
#define in_plane_gamma  -9.721486914088159e-02
#define out_of_plane_gamma  9.859272078406150e-02
#define r2i 4.500000000000000e+00 // A
#define r2f 6.500000000000000e+00 // A
#define d_intra 1.0
#define d_inter 4.0

extern "C" __device__ void computeExtraPoint(double3 * O, double3 * H1, double3 * H2, double3 * X1, double3 * X2) {
    // TODO save oh1 and oh2 to be used later?
    double3 oh1 = *H1 - *O;
    double3 oh2 = *H2 - *O;

    double3 v = cross(oh1, oh2);
    double3 in_plane = (*O) + (oh1 + oh2) * 0.5 * in_plane_gamma;
    double3 out_of_plane = v * out_of_plane_gamma;

    *X1 = in_plane + out_of_plane;
    *X2 = in_plane - out_of_plane;
}

extern "C" __device__ void computeExp(double r0, double k, double3 * O1, double3 * O2, double * exp1, double3 * g) {
    *g = *O1 - *O2;

    double r = sqrt(dot(*g, *g));
    *exp1 = exp(k*(r0 - r));
    *g *= -k * (*exp1) / r;
}

extern "C" __device__ void computeCoul(double r0, double k, double3 * O1, double3 * O2, double * val, double3 * g) {
    *g = *O1 - *O2;

    double r = sqrt(dot(*g, *g));
    double exp1 = exp(k * (r0 - r));
    double rinv = 1.0/r;
    *val = exp1*rinv;
    *g *=  - (k + rinv) * (*val) * rinv;
}

extern "C" __device__ void computeGrads(double * g, double3 * gOO, double3 * force1, double3 * force2, double sw) {

    double3 d = *g * (*gOO);
    *force1 += sw * d;
    *force2 -= sw * d;
}

extern "C" __device__ void distributeXpointGrad(double3 * O, double3 * H1, double3 * H2, double3 * forceX1, double3 * forceX2, double3 * forceO, double3 * forceH1, double3 * forceH2, double sw) {

    // TODO save oh1 and oh2 to be used later?
    double3 oh1 = *H1 - *O;
    double3 oh2 = *H2 - *O;

    double3 gm = *forceX1-*forceX2;

    double3 t1 = cross(oh2, gm);

    double3 t2 = cross(oh1, gm);

    double3 gsum = *forceX1 + *forceX2;
    double3 in_plane = gsum*0.5*in_plane_gamma;

    double3 gh1 = in_plane + t1*out_of_plane_gamma;
    double3 gh2 = in_plane - t2*out_of_plane_gamma;

    *forceO +=  sw * (gsum - (gh1 + gh2)); // O
    *forceH1 += sw * gh1; // H1
    *forceH2 += sw * gh2; // H2

}

extern "C" __device__ void evaluateSwitchFunc(double r, double * sw, double * gsw) {

    if (r > r2f) {
        *gsw = 0.0;
        *sw  = 0.0;
    } else if (r > r2i) {
        double t1 = M_PI/(r2f - r2i);
        double x = (r - r2i)*t1;
        *gsw = - sin(x)*t1/2.0;
        *sw  = (1.0 + cos(x))/2.0;
    } else {
        *gsw = 0.0;
        *sw = 1.0;
    }
}

extern "C" __device__ double computeInteraction(
        const unsigned int atom1,
        const unsigned int atom2,
        const double4* __restrict__ posq,
        double3 * forces) {
                    double tempEnergy = 0.0f;
                    // 2 water molecules and extra positions
                    double3 positions[10];
                    // first water
                    for (int i = 0; i < 3; i++) {
                        positions[Oa + i] = make_double3( posq[atom1+i].x,
                                                        posq[atom1+i].y,
                                                        posq[atom1+i].z);
                        positions[Ob + i] = make_double3( posq[atom2+i].x,
                                                        posq[atom2+i].y,
                                                        posq[atom2+i].z);
                    }

                    double3 delta = make_double3(positions[Ob].x-positions[Oa].x, positions[Ob].y-positions[Oa].y, positions[Ob].z-positions[Oa].z);
                    double r2 = delta.x*delta.x + delta.y*delta.y + delta.z*delta.z;
                    double invR = rsqrt(r2);
                    double rOO = r2*invR;
                    double sw = 1.;
                    double gsw = 1.;

                    if ((rOO > r2f) || (rOO < 2.)) {
                        tempEnergy = 0.;
                    } else {

                        evaluateSwitchFunc(rOO, &sw, &gsw);

                        computeExtraPoint(positions + Oa, positions + Ha1, positions + Ha2,
                               positions + Xa1, positions + Xa2);
                        computeExtraPoint(positions + Ob, positions + Hb1, positions + Hb2,
                                positions + Xb1, positions + Xb2);

                        double exp[31];
                        double3 gOO[31];
                        int i = 0;
                        computeExp(d_intra, k_HH_intra, positions +Ha1, positions +Ha2, exp+i, gOO+i); i++;
                        computeExp(d_intra, k_HH_intra, positions +Hb1, positions +Hb2, exp+i, gOO+i); i++;
                        computeExp(d_intra, k_OH_intra, positions +Oa,  positions +Ha1, exp+i, gOO+i); i++;
                        computeExp(d_intra, k_OH_intra, positions +Oa,  positions +Ha2, exp+i, gOO+i); i++;
                        computeExp(d_intra, k_OH_intra, positions +Ob,  positions +Hb1, exp+i, gOO+i); i++;
                        computeExp(d_intra, k_OH_intra, positions +Ob,  positions +Hb2, exp+i, gOO+i); i++;
                        computeCoul(d_inter, k_HH_coul, positions +Ha1, positions +Hb1, exp+i, gOO+i); i++;
                        computeCoul(d_inter, k_HH_coul, positions +Ha1, positions +Hb2, exp+i, gOO+i); i++;
                        computeCoul(d_inter, k_HH_coul, positions +Ha2, positions +Hb1, exp+i, gOO+i); i++;
                        computeCoul(d_inter, k_HH_coul, positions +Ha2, positions +Hb2, exp+i, gOO+i); i++;
                        computeCoul(d_inter, k_OH_coul, positions +Oa,  positions +Hb1, exp+i, gOO+i); i++;
                        computeCoul(d_inter, k_OH_coul, positions +Oa,  positions +Hb2, exp+i, gOO+i); i++;
                        computeCoul(d_inter, k_OH_coul, positions +Ob,  positions +Ha1, exp+i, gOO+i); i++;
                        computeCoul(d_inter, k_OH_coul, positions +Ob,  positions +Ha2, exp+i, gOO+i); i++;
                        computeCoul(d_inter, k_OO_coul, positions +Oa,  positions +Ob , exp+i, gOO+i); i++;
                        computeExp(d_inter, k_XH_main,  positions +Xa1, positions +Hb1, exp+i, gOO+i); i++;
                        computeExp(d_inter, k_XH_main,  positions +Xa1, positions +Hb2, exp+i, gOO+i); i++;
                        computeExp(d_inter, k_XH_main,  positions +Xa2, positions +Hb1, exp+i, gOO+i); i++;
                        computeExp(d_inter, k_XH_main,  positions +Xa2, positions +Hb2, exp+i, gOO+i); i++;
                        computeExp(d_inter, k_XH_main,  positions +Xb1, positions +Ha1, exp+i, gOO+i); i++;
                        computeExp(d_inter, k_XH_main,  positions +Xb1, positions +Ha2, exp+i, gOO+i); i++;
                        computeExp(d_inter, k_XH_main,  positions +Xb2, positions +Ha1, exp+i, gOO+i); i++;
                        computeExp(d_inter, k_XH_main,  positions +Xb2, positions +Ha2, exp+i, gOO+i); i++;
                        computeExp(d_inter, k_XO_main,  positions +Oa , positions +Xb1, exp+i, gOO+i); i++;
                        computeExp(d_inter, k_XO_main,  positions +Oa , positions +Xb2, exp+i, gOO+i); i++;
                        computeExp(d_inter, k_XO_main,  positions +Ob , positions +Xa1, exp+i, gOO+i); i++;
                        computeExp(d_inter, k_XO_main,  positions +Ob , positions +Xa2, exp+i, gOO+i); i++;
                        computeExp(d_inter, k_XX_main,  positions +Xa1, positions +Xb1, exp+i, gOO+i); i++;
                        computeExp(d_inter, k_XX_main,  positions +Xa1, positions +Xb2, exp+i, gOO+i); i++;
                        computeExp(d_inter, k_XX_main,  positions +Xa2, positions +Xb1, exp+i, gOO+i); i++;
                        computeExp(d_inter, k_XX_main,  positions +Xa2, positions +Xb2, exp+i, gOO+i); i++;

                        double g[31];
                        tempEnergy = poly_2b_v6x_eval(exp, g);

                        computeGrads(g+0,  gOO+0,  forces + Ha1, forces + Ha2, sw);
                        computeGrads(g+1,  gOO+1,  forces + Hb1, forces + Hb2, sw);
                        computeGrads(g+2,  gOO+2,  forces + Oa , forces + Ha1, sw);
                        computeGrads(g+3,  gOO+3,  forces + Oa , forces + Ha2, sw);
                        computeGrads(g+4,  gOO+4,  forces + Ob , forces + Hb1, sw);
                        computeGrads(g+5,  gOO+5,  forces + Ob , forces + Hb2, sw);
                        computeGrads(g+6,  gOO+6,  forces + Ha1, forces + Hb1, sw);
                        computeGrads(g+7,  gOO+7,  forces + Ha1, forces + Hb2, sw);
                        computeGrads(g+8,  gOO+8,  forces + Ha2, forces + Hb1, sw);
                        computeGrads(g+9,  gOO+9,  forces + Ha2, forces + Hb2, sw);
                        computeGrads(g+10, gOO+10, forces + Oa , forces + Hb1, sw);
                        computeGrads(g+11, gOO+11, forces + Oa , forces + Hb2, sw);
                        computeGrads(g+12, gOO+12, forces + Ob , forces + Ha1, sw);
                        computeGrads(g+13, gOO+13, forces + Ob , forces + Ha2, sw);
                        computeGrads(g+14, gOO+14, forces + Oa , forces + Ob , sw);
                        computeGrads(g+15, gOO+15, forces + Xa1, forces + Hb1, sw);
                        computeGrads(g+16, gOO+16, forces + Xa1, forces + Hb2, sw);
                        computeGrads(g+17, gOO+17, forces + Xa2, forces + Hb1, sw);
                        computeGrads(g+18, gOO+18, forces + Xa2, forces + Hb2, sw);
                        computeGrads(g+19, gOO+19, forces + Xb1, forces + Ha1, sw);
                        computeGrads(g+20, gOO+20, forces + Xb1, forces + Ha2, sw);
                        computeGrads(g+21, gOO+21, forces + Xb2, forces + Ha1, sw);
                        computeGrads(g+22, gOO+22, forces + Xb2, forces + Ha2, sw);
                        computeGrads(g+23, gOO+23, forces + Oa , forces + Xb1, sw);
                        computeGrads(g+24, gOO+24, forces + Oa , forces + Xb2, sw);
                        computeGrads(g+25, gOO+25, forces + Ob , forces + Xa1, sw);
                        computeGrads(g+26, gOO+26, forces + Ob , forces + Xa2, sw);
                        computeGrads(g+27, gOO+27, forces + Xa1, forces + Xb1, sw);
                        computeGrads(g+28, gOO+28, forces + Xa1, forces + Xb2, sw);
                        computeGrads(g+29, gOO+29, forces + Xa2, forces + Xb1, sw);
                        computeGrads(g+30, gOO+30, forces + Xa2, forces + Xb2, sw);


                    distributeXpointGrad(positions + Oa, positions + Ha1, positions + Ha2,
                            forces + Xa1, forces + Xa2,
                            forces + Oa, forces + Ha1, forces + Ha2, sw);

                    distributeXpointGrad(positions + Ob, positions + Hb1, positions + Hb2,
                            forces + Xb1, forces + Xb2,
                            forces + Ob, forces + Hb1, forces + Hb2, sw);

                    }

                    // gradient of the switch
                    gsw *= tempEnergy/rOO;
                    double3 d = gsw * delta;
                    forces[Oa] += d;
                    forces[Ob] -= d;

                    return sw * tempEnergy;
}

__global__ void evaluate_2b(
        const double4* __restrict__ posq,
        double3 * forces,
        double * energy) {
        // This function will parallelize computeInteraction to run in parallel with all the pairs of molecules,
        // for now just computing the interaction between the 2 molecules (identified by their Oxygen atom)
        energy[0] = computeInteraction(0, 3, posq, forces);
}

void launch_evaluate_2b(
        const double4* __restrict__ posq,
        double3 * forces,
        double * energy) {
        // This function is useful to configure the number of device threads, just one for now:
        evaluate_2b<<<1,1>>>(posq, forces, energy);
        cudaDeviceSynchronize();
}

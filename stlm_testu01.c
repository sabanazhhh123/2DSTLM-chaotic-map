/*
 * 2D-STLM  ->  TestU01 (SmallCrush / Crush / BigCrush)
 *   x_{n+1} = [ r1*sin(pi*x_n)*cos(pi*y_n) + mu*y_n*(1-y_n) ] mod 1
 *   y_{n+1} = [ r2*sin(pi*y_n)*cos(pi*x_n) + mu*x_n*(1-x_n) ] mod 1
 * Bits: digits 17..32 of x_n (or y_n); two iterates -> one 32-bit word.
 * Usage:  ./stlm_testu01  x|y  smallcrush|crush|bigcrush
 */
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "unif01.h"
#include "bbattery.h"

static const double R1 = 12.0, R2 = 12.0, MU = 3.9;
static const double X0 = 0.32665264, Y0 = 0.21961687;
static const int    WARMUP = 1000;

static double x, y;
static int use_y = 0;

static inline void stlm_step(void)
{
    double sx = sin(M_PI * x), cx = cos(M_PI * x);
    double sy = sin(M_PI * y), cy = cos(M_PI * y);
    double xn = fmod(R1 * sx * cy + MU * y * (1.0 - y), 1.0);
    double yn = fmod(R2 * sy * cx + MU * x * (1.0 - x), 1.0);
    if (xn < 0.0) xn += 1.0;
    if (yn < 0.0) yn += 1.0;
    x = xn; y = yn;
}

static inline uint32_t word16(void)
{
    double s = use_y ? y : x;
    return (uint32_t)((uint64_t)(s * 4294967296.0) & 0xFFFFu);
}

static unsigned int stlm_bits(void)
{
    stlm_step(); uint32_t a = word16();
    stlm_step(); uint32_t c = word16();
    return (a << 16) | c;
}

int main(int argc, char *argv[])
{
    if (argc < 3) {
        printf("Usage: %s x|y smallcrush|crush|bigcrush\n", argv[0]);
        return 1;
    }
    use_y = (strcmp(argv[1], "y") == 0);

    x = X0; y = Y0;
    for (int i = 0; i < WARMUP; i++) stlm_step();

    char name[64];
    snprintf(name, sizeof name, "2D-STLM %s-sequence", use_y ? "Y" : "X");
    unif01_Gen *gen = unif01_CreateExternGenBits(name, stlm_bits);

    if      (strcmp(argv[2], "smallcrush") == 0) bbattery_SmallCrush(gen);
    else if (strcmp(argv[2], "crush")      == 0) bbattery_Crush(gen);
    else if (strcmp(argv[2], "bigcrush")   == 0) bbattery_BigCrush(gen);
    else { printf("Unknown battery: %s\n", argv[2]); return 1; }

    unif01_DeleteExternGenBits(gen);
    return 0;
}

#include <lapacke.h>

#ifdef __cplusplus
extern "C" {
#endif

// layout: 101 = LAPACK_ROW_MAJOR, 102 = LAPACK_COL_MAJOR (LAPACKE constants)
// jobz:   'N' (S only), 'S' (thin), 'A' (full)
//
// The caller must allocate A, S, U, VT and set lda/ldu/ldvt correctly.
int svd_dgesdd(int layout, char jobz, int m, int n,
               double* a, int lda,
               double* s,
               double* u, int ldu,
               double* vt, int ldvt)
{
    return (int)LAPACKE_dgesdd((int)layout, jobz,
                              (lapack_int)m, (lapack_int)n,
                              a, (lapack_int)lda,
                              s,
                              u, (lapack_int)ldu,
                              vt, (lapack_int)ldvt);
}

int svd_zgesdd(int layout, char jobz, int m, int n,
               lapack_complex_double* a, int lda,
               double* s,
               lapack_complex_double* u, int ldu,
               lapack_complex_double* vt, int ldvt)
{
    return (int)LAPACKE_zgesdd((int)layout, jobz,
                              (lapack_int)m, (lapack_int)n,
                              a, (lapack_int)lda,
                              s,
                              u, (lapack_int)ldu,
                              vt, (lapack_int)ldvt);
}

#ifdef __cplusplus
}
#endif


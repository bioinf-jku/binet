

cdef extern from "./_mini_cblas.h":
    enum CBLAS_ORDER:     CblasRowMajor, CblasColMajor
    enum CBLAS_TRANSPOSE: CblasNoTrans, CblasTrans, CblasConjTrans

    void cblas_sgemm(CBLAS_ORDER Order, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB,
                     int M, int N, int K,
                     float alpha, float *A, int lda,
                     float *B, int ldb, float beta, float *C, int ldc) nogil
    void cblas_dgemm(CBLAS_ORDER Order, CBLAS_TRANSPOSE TransA, CBLAS_TRANSPOSE TransB,
                     int M, int N, int K,
                     double alpha, double *A, int lda,
                     double *B, int ldb, double beta, double *C, int ldc) nogil
    void cblas_saxpby(int N, float alpha,
                      float *X, int incX,
                      float beta, float *Y, int incY) nogil
    void cblas_daxpby(int N, double alpha,
                      double *X, int incX,
                      double beta, double *Y, int incY) nogil

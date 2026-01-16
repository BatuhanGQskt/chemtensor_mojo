from memory import UnsafePointer, alloc
from .types import i32, f64, min_i32, Layout, CComplexF64
from .matrix import MatrixF64, MatrixC64
from .backend import SVDBackend

@fieldwise_init
struct SVDResultF64:
    var jobz: Int8
    var m: i32
    var n: i32
    var k: i32
    var layout: i32
    var S: UnsafePointer[mut=True, type=f64, origin=MutAnyOrigin]
    var U: UnsafePointer[mut=True, type=f64, origin=MutAnyOrigin]
    var VT: UnsafePointer[mut=True, type=f64, origin=MutAnyOrigin]
    var u_rows: i32
    var u_cols: i32
    var vt_rows: i32
    var vt_cols: i32
    var owns: Bool

    fn deinit(self):
        if self.owns:
            self.S.free()
            self.U.free()
            self.VT.free()

fn svd_f64(backend: SVDBackend, A: MatrixF64, jobz: Int8) raises -> SVDResultF64:
    var m = A.m
    var n = A.n
    var k = min_i32(m, n)

    # Allocate S always (length k)
    var S = alloc[f64](Int(k))
    for i in range(Int(k)):
        (S + i).init_pointee_copy(0.0)

    # Determine U and VT shapes
    var u_rows: i32
    var u_cols: i32
    var vt_rows: i32
    var vt_cols: i32

    if jobz == Int8(ord('S')):
        u_rows = m
        u_cols = k
        vt_rows = k
        vt_cols = n
    elif jobz == Int8(ord('A')):
        u_rows = m
        u_cols = m
        vt_rows = n
        vt_cols = n
    else:
        # 'N': allocate minimal placeholders
        u_rows = 1
        u_cols = 1
        vt_rows = 1
        vt_cols = 1

    var U = alloc[f64](Int(u_rows * u_cols))
    var VT = alloc[f64](Int(vt_rows * vt_cols))
    for i in range(Int(u_rows * u_cols)):
        (U + i).init_pointee_copy(0.0)
    for i in range(Int(vt_rows * vt_cols)):
        (VT + i).init_pointee_copy(0.0)

    # Leading dimensions depend on layout
    var lda = A.lda()
    var ldu = u_rows if A.layout == Layout.COL_MAJOR() else u_cols
    var ldvt = vt_rows if A.layout == Layout.COL_MAJOR() else vt_cols

    # Call LAPACKE wrapper
    var info = backend.dgesdd(A.layout, jobz, m, n, A.data, lda, S, U, ldu, VT, ldvt)
    if info != 0:
        raise Error("dgesdd failed, info=" + String(info))

    return SVDResultF64(jobz, m, n, k, A.layout, S, U, VT, u_rows, u_cols, vt_rows, vt_cols, True)


# Complex result
@fieldwise_init
struct SVDResultC64:
    var jobz: Int8
    var m: i32
    var n: i32
    var k: i32
    var layout: i32
    var S: UnsafePointer[mut=True, type=f64, origin=MutAnyOrigin]
    var U: UnsafePointer[mut=True, type=CComplexF64, origin=MutAnyOrigin]
    var VT: UnsafePointer[mut=True, type=CComplexF64, origin=MutAnyOrigin]
    var u_rows: i32
    var u_cols: i32
    var vt_rows: i32
    var vt_cols: i32
    var owns: Bool

    fn deinit(self):
        if self.owns:
            self.S.free()
            self.U.free()
            self.VT.free()

fn svd_c64(backend: SVDBackend, A: MatrixC64, jobz: Int8) raises -> SVDResultC64:
    var m = A.m
    var n = A.n
    var k = min_i32(m, n)

    var S = alloc[f64](Int(k))
    for i in range(Int(k)):
        (S + i).init_pointee_copy(0.0)

    var u_rows: i32
    var u_cols: i32
    var vt_rows: i32
    var vt_cols: i32

    if jobz == Int8(ord('S')):
        u_rows = m
        u_cols = k
        vt_rows = k
        vt_cols = n
    elif jobz == Int8(ord('A')):
        u_rows = m
        u_cols = m
        vt_rows = n
        vt_cols = n
    else:
        u_rows = 1
        u_cols = 1
        vt_rows = 1
        vt_cols = 1

    var U = alloc[CComplexF64](Int(u_rows * u_cols))
    var VT = alloc[CComplexF64](Int(vt_rows * vt_cols))
    for i in range(Int(u_rows * u_cols)):
        (U + i).init_pointee_copy(CComplexF64(0.0, 0.0))
    for i in range(Int(vt_rows * vt_cols)):
        (VT + i).init_pointee_copy(CComplexF64(0.0, 0.0))

    var lda = A.lda()
    var ldu = u_rows if A.layout == Layout.COL_MAJOR() else u_cols
    var ldvt = vt_rows if A.layout == Layout.COL_MAJOR() else vt_cols

    var info = backend.zgesdd(A.layout, jobz, m, n, A.data, lda, S, U, ldu, VT, ldvt)
    if info != 0:
        raise Error("zgesdd failed, info=" + String(info))

    return SVDResultC64(jobz, m, n, k, A.layout, S, U, VT, u_rows, u_cols, vt_rows, vt_cols, True)


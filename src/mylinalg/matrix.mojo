from memory import UnsafePointer, alloc
from src.mylinalg.types import i32, f64, Layout, CComplexF64

# Minimal dense matrix (real)
struct MatrixF64:
    var m: i32
    var n: i32
    var layout: i32
    var data: UnsafePointer[mut=True, type=f64, origin=MutAnyOrigin]
    var owns: Bool

    fn __init__(out self, m: i32, n: i32, layout: i32):
        self.m = m
        self.n = n
        self.layout = layout
        self.data = alloc[f64](Int(m * n))
        self.owns = True

        # Initialize to 0.0 (allocated memory is uninitialized).
        for idx in range(Int(m * n)):
            (self.data + idx).init_pointee_copy(0.0)

    fn deinit(self):
        if self.owns:
            self.data.free()

    fn lda(self) -> i32:
        # For contiguous dense storage:
        #  - COL_MAJOR: leading dimension is number of rows
        #  - ROW_MAJOR: leading dimension is number of cols
        if self.layout == Layout.COL_MAJOR():
            return self.m
        else:
            return self.n

    fn index(self, i: i32, j: i32) -> i32:
        # 0-based indexing
        if self.layout == Layout.COL_MAJOR():
            # column-major: A[i + j*lda]
            return i + j * self.m
        else:
            # row-major: A[i*lda + j]
            return i * self.n + j

    fn set(self, i: i32, j: i32, v: f64):
        self.data[Int(self.index(i, j))] = v

    fn get(self, i: i32, j: i32) -> f64:
        return self.data[Int(self.index(i, j))]

# Minimal dense matrix (complex)
struct MatrixC64:
    var m: i32
    var n: i32
    var layout: i32
    var data: UnsafePointer[mut=True, type=CComplexF64, origin=MutAnyOrigin]
    var owns: Bool

    fn __init__(out self, m: i32, n: i32, layout: i32):
        self.m = m
        self.n = n
        self.layout = layout
        self.data = alloc[CComplexF64](Int(m * n))
        self.owns = True

        for idx in range(Int(m * n)):
            (self.data + idx).init_pointee_copy(CComplexF64(0.0, 0.0))

    fn deinit(self):
        if self.owns:
            self.data.free()

    fn lda(self) -> i32:
        if self.layout == Layout.COL_MAJOR():
            return self.m
        else:
            return self.n

    fn index(self, i: i32, j: i32) -> i32:
        if self.layout == Layout.COL_MAJOR():
            return i + j * self.m
        else:
            return i * self.n + j

    fn set(self, i: i32, j: i32, v: CComplexF64):
        self.data[Int(self.index(i, j))] = v

    fn get(self, i: i32, j: i32) -> CComplexF64:
        return self.data[Int(self.index(i, j))]


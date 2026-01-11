from sys.ffi import OwnedDLHandle
from memory import UnsafePointer
from src.mylinalg.types import i32, f64, CComplexF64

struct SVDBackend:
    var lib: OwnedDLHandle

    var dgesdd: fn (i32, Int8, i32, i32,
                    UnsafePointer[mut=True, type=f64, origin=MutAnyOrigin], i32,
                    UnsafePointer[mut=True, type=f64, origin=MutAnyOrigin],
                    UnsafePointer[mut=True, type=f64, origin=MutAnyOrigin], i32,
                    UnsafePointer[mut=True, type=f64, origin=MutAnyOrigin], i32) -> i32

    var zgesdd: fn (i32, Int8, i32, i32,
                    UnsafePointer[mut=True, type=CComplexF64, origin=MutAnyOrigin], i32,
                    UnsafePointer[mut=True, type=f64, origin=MutAnyOrigin],
                    UnsafePointer[mut=True, type=CComplexF64, origin=MutAnyOrigin], i32,
                    UnsafePointer[mut=True, type=CComplexF64, origin=MutAnyOrigin], i32) -> i32

    fn __init__(out self, so_path: String) raises:
        self.lib = OwnedDLHandle(so_path)
        self.dgesdd = self.lib.get_function[
            fn (i32, Int8, i32, i32,
                UnsafePointer[mut=True, type=f64, origin=MutAnyOrigin], i32,
                UnsafePointer[mut=True, type=f64, origin=MutAnyOrigin],
                UnsafePointer[mut=True, type=f64, origin=MutAnyOrigin], i32,
                UnsafePointer[mut=True, type=f64, origin=MutAnyOrigin], i32) -> i32
        ]("svd_dgesdd")

        self.zgesdd = self.lib.get_function[
            fn (i32, Int8, i32, i32,
                UnsafePointer[mut=True, type=CComplexF64, origin=MutAnyOrigin], i32,
                UnsafePointer[mut=True, type=f64, origin=MutAnyOrigin],
                UnsafePointer[mut=True, type=CComplexF64, origin=MutAnyOrigin], i32,
                UnsafePointer[mut=True, type=CComplexF64, origin=MutAnyOrigin], i32) -> i32
        ]("svd_zgesdd")


comptime i32 = Int32
comptime i64 = Int64
comptime f64 = Float64

# LAPACKE layout constants:
#   LAPACK_ROW_MAJOR = 101
#   LAPACK_COL_MAJOR = 102
struct Layout:
    @staticmethod
    fn ROW_MAJOR() -> i32:
        return 101
    
    @staticmethod
    fn COL_MAJOR() -> i32:
        return 102

@fieldwise_init
struct CComplexF64:
    var re: f64
    var im: f64

fn min_i32(a: i32, b: i32) -> i32:
    if a < b:
        return a
    return b


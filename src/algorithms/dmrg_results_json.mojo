"""Save DMRG results to a JSON file in the same format as the C implementation.

Schema: see chemtensor/manual_tests/DMRG_RESULTS_JSON_SCHEMA.md
Use so that dmrg_results_c_*.json and dmrg_results_mojo_*.json can be compared (e.g. diff or tolerance script).
"""

from collections.list import List
# open() is a Mojo built-in; use open(path, "w") to write


@fieldwise_init
struct DMRGJsonParams:
    """Parameters written to the 'params' object in JSON. Match C dmrg_json_params."""
    var nsites: Int
    var d: Int
    var J: Float64
    var D: Float64
    var h: Float64
    var num_sweeps: Int
    var maxiter_lanczos: Int
    var chi_max: Int
    var tol_split: Float64  # use -1.0 to omit from JSON (e.g. single-site)


fn _format_float(x: Float64) -> String:
    """Format double for JSON (enough digits for comparison; no exponent if not needed)."""
    return String(x)


fn _format_int(i: Int) -> String:
    return String(i)


fn dmrg_results_to_json_string(
    impl: String,
    model: String,
    params: DMRGJsonParams,
    energy_final: Float64,
    en_sweeps: List[Float64],
    entropy: List[Float64],  # empty to omit "entropy" key
    bond_dims: List[Int],
    norm: Float64,
) -> String:
    """Build the JSON string (same structure as C dmrg_results_to_json)."""
    var out = String("{\n")
    out += "  \"impl\": \"" + impl + "\",\n"
    out += "  \"model\": \"" + model + "\",\n"
    out += "  \"params\": {\n"
    out += "    \"nsites\": " + _format_int(params.nsites) + ",\n"
    out += "    \"d\": " + _format_int(params.d) + ",\n"
    out += "    \"J\": " + _format_float(params.J) + ",\n"
    out += "    \"D\": " + _format_float(params.D) + ",\n"
    out += "    \"h\": " + _format_float(params.h) + ",\n"
    out += "    \"num_sweeps\": " + _format_int(params.num_sweeps) + ",\n"
    out += "    \"maxiter_lanczos\": " + _format_int(params.maxiter_lanczos) + ",\n"
    out += "    \"chi_max\": " + _format_int(params.chi_max)
    if params.tol_split >= 0.0:
        out += ",\n    \"tol_split\": " + _format_float(params.tol_split)
    out += "\n  },\n"
    out += "  \"energy_final\": " + _format_float(energy_final) + ",\n"
    out += "  \"en_sweeps\": ["
    for i in range(len(en_sweeps)):
        if i > 0:
            out += ", "
        out += _format_float(en_sweeps[i])
    out += "],\n"
    if len(entropy) > 0:
        out += "  \"entropy\": ["
        for i in range(len(entropy)):
            if i > 0:
                out += ", "
            out += _format_float(entropy[i])
        out += "],\n"
    out += "  \"bond_dims\": ["
    for i in range(len(bond_dims)):
        if i > 0:
            out += ", "
        out += _format_int(bond_dims[i])
    out += "],\n"
    out += "  \"norm\": " + _format_float(norm) + "\n"
    out += "}\n"
    return out


fn save_dmrg_results_to_json(
    path: String,
    impl: String,
    model: String,
    params: DMRGJsonParams,
    energy_final: Float64,
    en_sweeps: List[Float64],
    entropy: List[Float64],
    bond_dims: List[Int],
    norm: Float64,
) raises -> None:
    """Write DMRG results to a JSON file. Same format as C dmrg_results_to_json."""
    var json_str = dmrg_results_to_json_string(
        impl, model, params, energy_final, en_sweeps, entropy, bond_dims, norm
    )
    var f = open(path, "w")
    f.write(json_str)
    f.close()

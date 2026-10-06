#[cfg(feature = "extension")]
#[pyo3::pymodule]
fn _native(m: &pyo3::Bound<'_, pyo3::types::PyModule>) -> pyo3::PyResult<()> {
    flatppl_python_api::register_module(m)
}

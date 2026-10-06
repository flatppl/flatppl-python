use std::sync::Arc;

use pyo3::exceptions::{PyException, PyValueError};
use pyo3::prelude::*;

use crate::compiler::{Context, Diagnostic, LoadedModule};

pyo3::create_exception!(_native, Error, PyException);

fn error(diagnostic: Diagnostic) -> PyErr {
    Error::new_err(
        serde_json::to_string(&diagnostic).expect("diagnostics contain only JSON values"),
    )
}

#[pyclass(name = "Context", module = "flatppl._native")]
struct NativeContext {
    inner: Context,
}

#[pymethods]
impl NativeContext {
    #[new]
    fn new() -> Self {
        Self {
            inner: Context::default(),
        }
    }

    #[pyo3(signature = (source, name=None, source_path=None))]
    fn parse(
        &mut self,
        source: &str,
        name: Option<&str>,
        source_path: Option<&str>,
    ) -> PyResult<NativeModule> {
        self.inner
            .parse(source, name, source_path)
            .map(|inner| NativeModule { inner })
            .map_err(error)
    }

    fn load(&mut self, path: &str) -> PyResult<NativeModule> {
        self.inner
            .load(path)
            .map(|inner| NativeModule { inner })
            .map_err(error)
    }

    fn register(&mut self, name: &str, module: &NativeModule) -> PyResult<()> {
        self.inner.register(name, &module.inner).map_err(error)
    }
}

#[pyclass(name = "Module", module = "flatppl._native", frozen)]
struct NativeModule {
    inner: Arc<LoadedModule>,
}

#[pymethods]
impl NativeModule {
    #[getter]
    fn source(&self) -> &str {
        self.inner.source()
    }

    #[getter]
    fn source_name(&self) -> &str {
        self.inner.source_name()
    }

    fn bindings(&self) -> String {
        serde_json::to_string(&self.inner.bindings())
            .expect("binding metadata contains only JSON values")
    }

    fn export(&self, py: Python<'_>, dtype: &str, autodiff: bool) -> PyResult<String> {
        let dtype = match dtype {
            "float32" => flatppl_stablehlo::Dtype::F32,
            "float64" => flatppl_stablehlo::Dtype::F64,
            _ => return Err(PyValueError::new_err("dtype must be float32 or float64")),
        };
        let options = flatppl_stablehlo::EmitOptions {
            dtype,
            enzyme_compatible: autodiff,
        };
        let exported = py.detach(|| self.inner.compile(&options)).map_err(error)?;
        Ok(serde_json::to_string(&exported).expect("exports contain only JSON values"))
    }
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<NativeContext>()?;
    m.add_class::<NativeModule>()?;
    m.add("Error", m.py().get_type::<Error>())?;
    Ok(())
}

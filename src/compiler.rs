//! Host-owned source snapshots. Imported definitions never cache inferred instances.

use std::collections::{HashMap, HashSet};
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

use flatppl_core::{CallHead, Idx, Module, Node, NodeId, RefNs, Scalar};
use flatppl_fileaccess::{Cache, DenyAll, Location, OfflineFetcher, Resolver};
use flatppl_infer::{Level, ModuleBundle, Severity};
use flatppl_stablehlo::EmitOptions;
use serde::Serialize;

use crate::export::{Export, emit_query};

#[derive(Debug, Serialize)]
pub struct Diagnostic {
    pub stage: &'static str,
    pub source: String,
    pub message: String,
    pub span: Option<(u32, u32)>,
    pub import_chain: Vec<String>,
}

impl Diagnostic {
    fn new(stage: &'static str, source: &str, message: impl ToString) -> Self {
        Self {
            stage,
            source: source.into(),
            message: message.to_string(),
            span: None,
            import_chain: Vec::new(),
        }
    }

    fn at(mut self, module: &Module, node: Option<NodeId>) -> Self {
        self.span = node
            .and_then(|id| module.span_of(id))
            .map(|s| (s.start, s.end));
        self
    }
}

#[derive(Clone)]
enum Origin {
    File {
        location: Location,
        bundle_root: Option<PathBuf>,
    },
    Inline {
        name: String,
        file: Option<Location>,
    },
}

struct Definition {
    identity: String,
    label: String,
    source: String,
    parsed: Arc<Module>,
    imports: Vec<(String, Arc<Definition>)>,
}

/// An immutable root instance and the parsed definitions required to compile it.
pub struct LoadedModule {
    owner: Arc<()>,
    definition: Arc<Definition>,
    typed: Module,
    bundle: ModuleBundle,
}

#[derive(Serialize)]
pub struct BindingInfo {
    pub name: String,
    pub value_type: Option<String>,
    pub phase: Option<String>,
    pub span: Option<(u32, u32)>,
}

impl LoadedModule {
    pub fn source(&self) -> &str {
        &self.definition.source
    }

    pub fn source_name(&self) -> &str {
        &self.definition.label
    }

    pub fn bindings(&self) -> Vec<BindingInfo> {
        self.typed
            .public_bindings()
            .map(|(_, b)| BindingInfo {
                name: self.typed.resolve(b.name).into(),
                value_type: self
                    .typed
                    .type_of(b.rhs)
                    .map(|t| self.typed.display_type(t)),
                phase: self
                    .typed
                    .phase_of(b.rhs)
                    .map(|p| format!("{p:?}").to_lowercase()),
                span: self.typed.span_of(b.rhs).map(|s| (s.start, s.end)),
            })
            .collect()
    }

    pub fn compile(&self, options: &EmitOptions) -> Result<Export, Diagnostic> {
        let roots: Vec<_> = self
            .typed
            .public_bindings()
            .filter(|(_, b)| matches!(self.typed.resolve(b.name), "inputs" | "outputs"))
            .map(|(_, b)| b.name)
            .collect();
        if roots.is_empty() {
            return Err(Diagnostic::new(
                "signature",
                self.source_name(),
                "declare inputs and outputs before compiling a module",
            ));
        }
        let lowered = flatppl_determinizer::determinize_with_options(
            &self.typed,
            &self.bundle,
            Some(&roots),
            &flatppl_determinizer::LoweringOptions {
                preserve_rand_tuple: true,
                ..flatppl_stablehlo::LOWERING_OPTIONS
            },
        )
        .map_err(|e| {
            Diagnostic::new("determinize", self.source_name(), e.reason)
                .at(&self.typed, Some(e.node))
        })?;
        emit_query(lowered, &self.typed, options)
            .map_err(|e| Diagnostic::new("stablehlo", self.source_name(), e.msg))
    }
}

/// Append-only definitions and explicit names. Existing modules own their import closure.
#[derive(Default)]
pub struct Context {
    owner: Arc<()>,
    registry: HashMap<String, Arc<Definition>>,
    files: HashMap<(String, Option<PathBuf>), Arc<Definition>>,
    sources: HashMap<String, String>,
    next_inline: usize,
}

impl Context {
    pub fn parse(
        &mut self,
        source: &str,
        name: Option<&str>,
        source_path: Option<&str>,
    ) -> Result<Arc<LoadedModule>, Diagnostic> {
        let logical = match name {
            Some(name) => registry_name(name)?,
            None => format!("<inline-{}>", self.next_inline),
        };
        let identity = format!("inline:{}", self.next_inline);
        self.next_inline += 1;
        let file = source_path.map(absolute_location).transpose()?;
        let label = file
            .as_ref()
            .map_or_else(|| logical.clone(), Location::display);
        let origin = Origin::Inline {
            name: logical,
            file,
        };
        let definition =
            self.definition(identity, label, source.into(), origin, &mut Vec::new())?;
        self.instantiate(definition)
    }

    pub fn load(&mut self, path: &str) -> Result<Arc<LoadedModule>, Diagnostic> {
        let location = absolute_location(path)?;
        let definition = self.file(location, None, &mut Vec::new())?;
        self.instantiate(definition)
    }

    pub fn register(&mut self, name: &str, module: &LoadedModule) -> Result<(), Diagnostic> {
        let name = registry_name(name)?;
        if !Arc::ptr_eq(&self.owner, &module.owner) {
            return Err(Diagnostic::new(
                "context",
                &name,
                "the module belongs to a different context",
            ));
        }
        if self.registry.contains_key(&name) {
            return Err(Diagnostic::new(
                "context",
                &name,
                "the name is already registered; use a new context to change a definition",
            ));
        }
        self.registry.insert(name, module.definition.clone());
        Ok(())
    }

    fn instantiate(&self, definition: Arc<Definition>) -> Result<Arc<LoadedModule>, Diagnostic> {
        let mut bundle = ModuleBundle::new();
        bundle.set_root(&definition.identity);
        let mut pending = vec![definition.clone()];
        let mut seen = HashSet::new();
        while let Some(parent) = pending.pop() {
            if !seen.insert(parent.identity.clone()) {
                continue;
            }
            for (literal, child) in &parent.imports {
                bundle.insert_resolved(
                    &parent.identity,
                    literal,
                    &child.identity,
                    child.parsed.clone(),
                );
                pending.push(child.clone());
            }
        }
        let mut typed = (*definition.parsed).clone();
        let (diagnostics, bundle) =
            flatppl_infer::infer_module_with_inputs(&mut typed, bundle, Level::Shape);
        if let Some(d) = diagnostics
            .into_iter()
            .find(|d| d.severity == Severity::Error)
        {
            return Err(Diagnostic::new("infer", &definition.label, d.message).at(&typed, d.node));
        }
        Ok(Arc::new(LoadedModule {
            owner: self.owner.clone(),
            definition,
            typed,
            bundle,
        }))
    }

    fn definition(
        &mut self,
        identity: String,
        label: String,
        source: String,
        origin: Origin,
        stack: &mut Vec<String>,
    ) -> Result<Arc<Definition>, Diagnostic> {
        if stack.contains(&identity) {
            return Err(Diagnostic::new("resolve", &label, "cyclic module imports"));
        }
        let parsed = flatppl_syntax::parse(&source).map_err(|e| {
            let mut d = Diagnostic::new("parse", &label, e.message);
            d.span = e.span;
            d
        })?;
        let mut literals = Vec::new();
        for i in 0..parsed.node_count() {
            if let Node::Call(c) = parsed.node(NodeId::from_usize(i))
                && matches!(c.head, CallHead::Builtin(s) if parsed.resolve(s) == "load_module")
                && let Some(&arg) = c.args.first()
            {
                let id = resolve_ref(&parsed, arg);
                if let Node::Lit(Scalar::Str(s)) = parsed.node(id)
                    && !literals.contains(&s.to_string())
                {
                    literals.push(s.to_string());
                }
            }
        }
        stack.push(identity.clone());
        let imports = literals
            .into_iter()
            .map(|literal| {
                self.resolve(&origin, &literal, stack)
                    .map(|d| (literal, d))
                    .map_err(|mut e| {
                        e.import_chain.insert(0, label.clone());
                        e
                    })
            })
            .collect::<Result<Vec<_>, _>>();
        stack.pop();
        Ok(Arc::new(Definition {
            identity,
            label,
            source,
            parsed: Arc::new(parsed),
            imports: imports?,
        }))
    }

    fn resolve(
        &mut self,
        origin: &Origin,
        literal: &str,
        stack: &mut Vec<String>,
    ) -> Result<Arc<Definition>, Diagnostic> {
        let Origin::Inline { name, file } = origin else {
            let Origin::File {
                location,
                bundle_root,
            } = origin
            else {
                unreachable!()
            };
            let root = if Path::new(literal).is_absolute()
                || flatppl_core::text::uri_scheme(literal).is_some()
            {
                None
            } else {
                bundle_root.clone()
            };
            return self.file(location.join(literal), root, stack);
        };
        let explicit_location =
            flatppl_core::text::uri_scheme(literal).is_some() || Path::new(literal).is_absolute();
        let location = if explicit_location {
            Some(Location::parse(literal))
        } else {
            file.as_ref().map(|f| f.join(literal))
        };
        let registered = if explicit_location {
            None
        } else {
            let relative = Path::new(name)
                .parent()
                .unwrap_or(Path::new(""))
                .join(literal);
            registry_name(&relative.to_string_lossy())
                .ok()
                .and_then(|key| self.registry.get(&key))
                .cloned()
        };
        if let Some(definition) = registered {
            if let Some(Location::Local(path)) = &location
                && path.exists()
            {
                let root = if path.is_dir() {
                    Some(
                        std::fs::canonicalize(path)
                            .map_err(|e| Diagnostic::new("resolve", literal, e))?,
                    )
                } else {
                    None
                };
                let target = if root.is_some() {
                    path.join("main.flatppl")
                } else {
                    path.clone()
                };
                let canonical = std::fs::canonicalize(target)
                    .map_err(|e| Diagnostic::new("resolve", literal, e))?;
                let key = (canonical.to_string_lossy().into_owned(), root);
                if file_identity(&key) != definition.identity {
                    return Err(Diagnostic::new(
                        "resolve",
                        literal,
                        "ambiguous registered and file source",
                    ));
                }
            }
            return Ok(definition);
        }
        match location {
            Some(location) => self.file(location, None, stack),
            None => Err(Diagnostic::new(
                "resolve",
                literal,
                "no registered module; provide source_path for relative file imports",
            )),
        }
    }

    fn file(
        &mut self,
        mut location: Location,
        mut bundle_root: Option<PathBuf>,
        stack: &mut Vec<String>,
    ) -> Result<Arc<Definition>, Diagnostic> {
        location = location.normalized();
        let requested = location.display();
        let request_key = (requested.clone(), bundle_root.clone());
        if let Some(definition) = self.files.get(&request_key) {
            return Ok(definition.clone());
        }
        if let Location::Local(path) = &location {
            let path = if path.is_dir() {
                let directory = std::fs::canonicalize(path)
                    .map_err(|e| Diagnostic::new("resolve", &requested, e))?;
                check_bundle_path(&directory, bundle_root.as_deref(), &requested)?;
                bundle_root = Some(directory);
                path.join("main.flatppl")
            } else {
                path.clone()
            };
            location = Location::Local(
                std::fs::canonicalize(path)
                    .map_err(|e| Diagnostic::new("resolve", &requested, e))?,
            );
            if let Location::Local(path) = &location {
                check_bundle_path(path, bundle_root.as_deref(), &requested)?;
            }
        }
        let label = location.display();
        let key = (label.clone(), bundle_root.clone());
        if let Some(definition) = self.files.get(&key).cloned() {
            self.files.insert(request_key, definition.clone());
            return Ok(definition);
        }
        let identity = file_identity(&key);
        let source = if let Some(source) = self.sources.get(&label) {
            source.clone()
        } else {
            let mut cache = Cache::from_env();
            cache.set_offline(true);
            let resolver = Resolver::new(cache, &OfflineFetcher, &DenyAll);
            let bytes = resolver
                .read(&location)
                .map_err(|e| Diagnostic::new("resolve", &label, e))?;
            let source =
                String::from_utf8(bytes).map_err(|e| Diagnostic::new("parse", &label, e))?;
            self.sources.insert(label.clone(), source.clone());
            source
        };
        let definition = self.definition(
            identity,
            label,
            source,
            Origin::File {
                location,
                bundle_root,
            },
            stack,
        )?;
        self.files.insert(key, definition.clone());
        self.files.insert(request_key, definition.clone());
        Ok(definition)
    }
}

fn file_identity(key: &(String, Option<PathBuf>)) -> String {
    format!("file:{key:?}")
}

fn check_bundle_path(path: &Path, root: Option<&Path>, label: &str) -> Result<(), Diagnostic> {
    if root.is_some_and(|root| !path.starts_with(root)) {
        return Err(Diagnostic::new(
            "resolve",
            label,
            "relative import escapes its bundle root",
        ));
    }
    Ok(())
}

fn absolute_location(source: &str) -> Result<Location, Diagnostic> {
    let location = Location::parse(source);
    match location {
        Location::Local(path) if path.is_relative() => Ok(Location::Local(
            std::env::current_dir()
                .map_err(|e| Diagnostic::new("resolve", source, e))?
                .join(path),
        )
        .normalized()),
        location => Ok(location.normalized()),
    }
}

fn registry_name(name: &str) -> Result<String, Diagnostic> {
    let mut parts = Vec::new();
    for part in Path::new(name).components() {
        match part {
            Component::Normal(p) => parts.push(p.to_string_lossy().into_owned()),
            Component::CurDir => {}
            Component::ParentDir if !parts.is_empty() => {
                parts.pop();
            }
            _ => {
                return Err(Diagnostic::new(
                    "context",
                    name,
                    "registry names must stay within the context's relative namespace",
                ));
            }
        }
    }
    if parts.is_empty() || flatppl_core::text::uri_scheme(name).is_some() {
        return Err(Diagnostic::new(
            "context",
            name,
            "expected a nonempty relative registry name",
        ));
    }
    Ok(parts.join("/"))
}

pub(crate) fn resolve_ref(module: &Module, mut id: NodeId) -> NodeId {
    let mut seen = HashSet::new();
    while seen.insert(id) {
        if let Node::Ref(r) = module.node(id)
            && r.ns == RefNs::SelfMod
            && let Some(b) = module.binding_by_name(r.name)
        {
            id = module.binding(b).rhs;
        } else {
            break;
        }
    }
    id
}

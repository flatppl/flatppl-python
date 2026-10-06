//! Explicit query ABI and host metadata, using the existing StableHLO builder.

use std::collections::{HashMap, HashSet};

use flatppl_core::{
    Call, CallHead, Dim, Idx, Module, Node, NodeId, Phase, RefNs, Scalar, Symbol, Type,
};
use flatppl_stablehlo::{
    Dtype, ElemKind, EmitError, EmitOptions, Emitter, MlirTy, Value, mlir_type_of,
};
use serde::Serialize;

#[derive(Serialize)]
pub struct Export {
    pub stablehlo: String,
    pub entry_point: &'static str,
    pub inputs: Vec<Field>,
    pub output: Schema,
    pub multiple_outputs: bool,
    pub output_names: Vec<Option<String>>,
}

#[derive(Serialize)]
pub struct Field {
    pub name: String,
    pub value: Schema,
}

#[derive(Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Schema {
    Tensor {
        index: usize,
        dtype: &'static str,
        shape: Vec<u64>,
        value_type: String,
    },
    Record {
        fields: Vec<Field>,
    },
    Table {
        fields: Vec<Field>,
        rows: u64,
    },
    Tuple {
        items: Vec<Schema>,
    },
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
enum Selector {
    Field(String),
    Index(usize),
}

type InputPath = (Symbol, Vec<Selector>);

struct Leaf {
    node: NodeId,
    ty: MlirTy,
    elem: ElemKind,
}

pub fn emit_query(
    mut module: Module,
    source: &Module,
    options: &EmitOptions,
) -> Result<Export, EmitError> {
    let dtype = options.dtype;
    flatppl_determinizer::is_flatpdl_with_options(&module, &flatppl_stablehlo::LOWERING_OPTIONS)
        .map_err(|errors| {
            EmitError::whole(
                errors
                    .into_iter()
                    .map(|e| e.reason)
                    .collect::<Vec<_>>()
                    .join("; "),
            )
        })?;
    let output_node = binding(&module, "outputs")
        .ok_or_else(|| EmitError::whole("declare at least one output"))?;
    let input_nodes = binding(&module, "inputs")
        .map(|id| signature(&module, id))
        .unwrap_or_default();
    let mut names = HashSet::new();
    let mut declared = Vec::new();
    for node in input_nodes {
        let Node::Ref(r) = module.node(node) else {
            return Err(EmitError::at(node, "each input must name a binding"));
        };
        if r.ns != RefNs::SelfMod || !names.insert(r.name) {
            return Err(EmitError::at(
                node,
                "each input must name a distinct local binding",
            ));
        }
        let rhs = module
            .binding(
                module
                    .binding_by_name(r.name)
                    .ok_or_else(|| EmitError::at(node, "input binding is absent"))?,
            )
            .rhs;
        if !builtin(&module, rhs, "elementof") && module.phase_of(rhs) != Some(Phase::Fixed) {
            return Err(EmitError::at(
                rhs,
                "an input must be an elementof leaf or a fixed value",
            ));
        }
        declared.push((module.resolve(r.name).to_string(), node));
    }
    check_dependencies(&module, output_node, &names)?;
    let roots = names;
    let mut arguments = Vec::new();
    let mut inputs = Vec::new();
    for (name, node) in declared {
        let ty = inferred(&module, node)?;
        let value = layout(&mut module, node, ty, &roots, true, dtype, &mut arguments)?;
        inputs.push(Field { name, value });
    }
    // Canonicalization may inline output bindings. Describe their authored names.
    let declared_output = binding(source, "outputs")
        .ok_or_else(|| EmitError::whole("declare at least one output"))?;
    let multiple_outputs = builtin(source, declared_output, "tuple");
    let output_names = signature(source, declared_output)
        .iter()
        .map(|node| match source.node(*node) {
            Node::Ref(r) if r.ns == RefNs::SelfMod => Some(source.resolve(r.name).into()),
            _ => None,
        })
        .collect();
    let output_type = inferred(&module, output_node)?;
    let mut results = Vec::new();
    let output = layout(
        &mut module,
        output_node,
        output_type,
        &roots,
        false,
        dtype,
        &mut results,
    )?;
    if results.is_empty() {
        return Err(EmitError::at(
            output_node,
            "the output has no tensor leaves",
        ));
    }

    let mut paths = HashMap::new();
    let mut values = HashMap::new();
    let mut args = Vec::new();
    for (index, leaf) in arguments.iter().enumerate() {
        let path = input_path(&module, leaf.node, &roots, &mut paths).ok_or_else(|| {
            EmitError::at(
                leaf.node,
                "cannot map an input leaf to its declared argument",
            )
        })?;
        let value = Value {
            ssa: format!("%arg{index}"),
            ty: leaf.ty.clone(),
            elem: leaf.elem,
        };
        args.push((value.ssa.clone(), value.ty.clone(), value.elem));
        values.insert(path, value);
    }
    let mut emitter = Emitter::with_options(&module, options);
    // A declared aggregate has no tensor value. Bind its leaf projections,
    // including aliases and the output projections created by layout().
    for i in 0..module.node_count() {
        let node = NodeId::from_usize(i);
        if let Some(path) = input_path(&module, node, &roots, &mut paths)
            && let Some(value) = values.get(&path)
        {
            emitter.bind(node, value.clone());
        }
    }
    let mut returned = Vec::new();
    for leaf in results {
        let value = emitter.lower_node(leaf.node)?;
        if value.ty != leaf.ty || value.elem != leaf.elem {
            return Err(EmitError::at(
                leaf.node,
                "emitted result type disagrees with the query schema",
            ));
        }
        returned.push(value);
    }
    let stablehlo = emitter.finish("main", &args, &returned.iter().collect::<Vec<_>>());
    Ok(Export {
        stablehlo,
        entry_point: "main",
        inputs,
        output,
        multiple_outputs,
        output_names,
    })
}

fn binding(module: &Module, name: &str) -> Option<NodeId> {
    module
        .bindings()
        .find(|(_, b)| module.resolve(b.name) == name)
        .map(|(_, b)| b.rhs)
}

fn builtin(module: &Module, node: NodeId, name: &str) -> bool {
    matches!(module.node(node), Node::Call(c) if matches!(c.head, CallHead::Builtin(s) if module.resolve(s) == name))
}

fn signature(module: &Module, node: NodeId) -> Vec<NodeId> {
    if builtin(module, node, "tuple")
        && let Node::Call(c) = module.node(node)
    {
        c.args.to_vec()
    } else {
        vec![node]
    }
}

fn inferred(module: &Module, node: NodeId) -> Result<Type, EmitError> {
    module
        .type_of(node)
        .cloned()
        .ok_or_else(|| EmitError::at(node, "query value has no inferred type"))
}

fn dimension(node: NodeId, dim: Dim) -> Result<u64, EmitError> {
    match dim {
        Dim::Static(n) => Ok(n.into()),
        Dim::Dynamic => Err(EmitError::at(
            node,
            "the JAX interface requires a static input and output shape",
        )),
    }
}

fn check_dependencies(
    module: &Module,
    output: NodeId,
    inputs: &HashSet<Symbol>,
) -> Result<(), EmitError> {
    let mut pending = vec![output];
    let mut seen = HashSet::new();
    while let Some(node) = pending.pop() {
        if !seen.insert(node) {
            continue;
        }
        if let Node::Ref(r) = module.node(node)
            && r.ns == RefNs::SelfMod
        {
            if !inputs.contains(&r.name)
                && let Some(binding) = module.binding_by_name(r.name)
            {
                pending.push(module.binding(binding).rhs);
            }
            continue;
        }
        if ["elementof", "external", "load_data"]
            .iter()
            .any(|head| builtin(module, node, head))
        {
            return Err(EmitError::at(
                node,
                "a reached runtime value must be declared in inputs",
            ));
        }
        module.for_each_child(node, |child| pending.push(child));
    }
    Ok(())
}

fn layout(
    module: &mut Module,
    node: NodeId,
    ty: Type,
    roots: &HashSet<Symbol>,
    input: bool,
    dtype: Dtype,
    leaves: &mut Vec<Leaf>,
) -> Result<Schema, EmitError> {
    match ty {
        Type::Record(fields) => {
            let mut result = Vec::new();
            for (name, ty) in fields.iter() {
                let name = module.resolve(*name).to_string();
                let child = project(
                    module,
                    node,
                    Selector::Field(name.clone()),
                    ty.clone(),
                    roots,
                    input,
                );
                result.push(Field {
                    name,
                    value: layout(module, child, ty.clone(), roots, input, dtype, leaves)?,
                });
            }
            Ok(Schema::Record { fields: result })
        }
        Type::Table { columns, nrows } => {
            let rows = dimension(node, nrows)?;
            let mut fields = Vec::new();
            for (name, ty) in columns.iter() {
                let name = module.resolve(*name).to_string();
                let column = match ty {
                    Type::Record(fields) => Type::Table {
                        columns: fields.clone(),
                        nrows,
                    },
                    ty => Type::Array {
                        shape: Box::new([nrows]),
                        elem: Box::new(ty.clone()),
                    },
                };
                let child = project(
                    module,
                    node,
                    Selector::Field(name.clone()),
                    column.clone(),
                    roots,
                    input,
                );
                fields.push(Field {
                    name,
                    value: layout(module, child, column, roots, input, dtype, leaves)?,
                });
            }
            Ok(Schema::Table { fields, rows })
        }
        Type::Tuple(items) => {
            let mut result = Vec::new();
            for (index, ty) in items.iter().enumerate() {
                let child = project(
                    module,
                    node,
                    Selector::Index(index),
                    ty.clone(),
                    roots,
                    input,
                );
                result.push(layout(
                    module,
                    child,
                    ty.clone(),
                    roots,
                    input,
                    dtype,
                    leaves,
                )?);
            }
            Ok(Schema::Tuple { items: result })
        }
        ty => {
            let (tensor, elem) = mlir_type_of(module, node, dtype)?;
            let shape = match &tensor {
                MlirTy::Scalar => Vec::new(),
                MlirTy::Key => vec![2],
                MlirTy::Ranked(dims) => dims
                    .iter()
                    .map(|d| {
                        d.ok_or_else(|| {
                            EmitError::at(node, "the JAX interface requires static tensor shapes")
                        })
                    })
                    .collect::<Result<_, _>>()?,
                MlirTy::Tuple(_) => {
                    return Err(EmitError::at(node, "tuple tensors must be flattened"));
                }
            };
            let kind = if tensor == MlirTy::Key {
                "uint64"
            } else {
                match (elem, dtype) {
                    (ElemKind::Bool, _) => "bool",
                    (ElemKind::Int, Dtype::F32) => "int32",
                    (ElemKind::Int, Dtype::F64) => "int64",
                    (ElemKind::Real, Dtype::F32) => "float32",
                    (ElemKind::Real, Dtype::F64) => "float64",
                }
            };
            let result = Schema::Tensor {
                index: leaves.len(),
                dtype: kind,
                shape,
                value_type: module.display_type(&ty),
            };
            leaves.push(Leaf {
                node,
                ty: tensor,
                elem,
            });
            Ok(result)
        }
    }
}

fn project(
    module: &mut Module,
    node: NodeId,
    selector: Selector,
    ty: Type,
    roots: &HashSet<Symbol>,
    input: bool,
) -> NodeId {
    let mut node = node;
    while let Node::Ref(r) = module.node(node)
        && r.ns == RefNs::SelfMod
        && !roots.contains(&r.name)
        && let Some(binding) = module.binding_by_name(r.name)
    {
        node = module.binding(binding).rhs;
    }
    if !input && let Node::Call(c) = module.node(node) {
        match &selector {
            Selector::Field(name)
                if builtin(module, node, "record") || builtin(module, node, "table") =>
            {
                if let Some(field) = c.named.iter().find(|f| module.resolve(f.name) == name) {
                    return field.value;
                }
            }
            Selector::Index(index) if builtin(module, node, "tuple") => {
                if let Some(&value) = c.args.get(*index) {
                    return value;
                }
            }
            _ => {}
        }
    }
    let selector = module.alloc(Node::Lit(match selector {
        Selector::Field(name) => Scalar::Str(name.into()),
        Selector::Index(index) => Scalar::Int(index as i64 + 1),
    }));
    let get = module.intern("get");
    let child = module.alloc(Node::Call(Call {
        head: CallHead::Builtin(get),
        args: Box::new([node, selector]),
        named: Box::new([]),
        inputs: None,
    }));
    module.set_type(child, ty);
    if let Some(phase) = module.phase_of(node) {
        module.set_phase(child, phase);
    }
    child
}

fn input_path(
    module: &Module,
    node: NodeId,
    roots: &HashSet<Symbol>,
    memo: &mut HashMap<NodeId, Option<InputPath>>,
) -> Option<InputPath> {
    if let Some(path) = memo.get(&node) {
        return path.clone();
    }
    let path = match module.node(node) {
        Node::Ref(r) if r.ns == RefNs::SelfMod && roots.contains(&r.name) => {
            Some((r.name, Vec::new()))
        }
        Node::Ref(r) if r.ns == RefNs::SelfMod => module
            .binding_by_name(r.name)
            .and_then(|b| input_path(module, module.binding(b).rhs, roots, memo)),
        Node::Call(c)
            if (builtin(module, node, "get") || builtin(module, node, "get0"))
                && c.args.len() == 2 =>
        {
            let selector = match (module.type_of(c.args[0]), module.node(c.args[1])) {
                (Some(Type::Record(_) | Type::Table { .. }), Node::Lit(Scalar::Str(s))) => {
                    Some(Selector::Field(s.to_string()))
                }
                (Some(Type::Tuple(_)), Node::Lit(Scalar::Int(i))) => {
                    let index = i - i64::from(builtin(module, node, "get"));
                    usize::try_from(index).ok().map(Selector::Index)
                }
                _ => None,
            };
            selector.and_then(|selector| {
                input_path(module, c.args[0], roots, memo).map(|(root, mut path)| {
                    path.push(selector);
                    (root, path)
                })
            })
        }
        _ => None,
    };
    memo.insert(node, path.clone());
    path
}

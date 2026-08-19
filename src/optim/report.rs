//! [`MuonPlan::describe`]: which optimizer each of a model's parameters lands
//! on — the quickest way to check a plan against a real model.

use burn::module::{Module, ModuleVisitor, Param, ParamId};
use burn::prelude::*;
use core::fmt::Write as _;

use super::MuonPlan;

/// One visited parameter.
struct Row {
    path: String,
    dims: Vec<usize>,
    id: ParamId,
}

#[derive(Default)]
struct Collect {
    path: Vec<String>,
    rows: Vec<Row>,
}

impl ModuleVisitor for Collect {
    fn enter_module(&mut self, name: &str, _container_type: &str) {
        self.path.push(name.to_string());
    }
    fn exit_module(&mut self, _name: &str, _container_type: &str) {
        self.path.pop();
    }
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<D>>) {
        self.rows.push(Row {
            path: self.path.join("."),
            dims: param.val().shape().dims::<D>().to_vec(),
            id: param.id,
        });
    }
}

impl MuonPlan {
    /// A per-parameter report of this plan applied to `module`: path, shape, and
    /// the owning optimizer (with the column segments for a fused weight, `*`
    /// marking the ones left on AdamW), plus the share of parameters on Muon.
    ///
    /// Purely diagnostic — print it once to confirm the plan matches the model
    /// you actually built.
    pub fn describe(&self, module: &impl Module) -> String {
        let mut collect = Collect::default();
        module.visit(&mut collect);

        let mut out = String::new();
        let (mut total, mut on_muon) = (0usize, 0usize);

        for row in &collect.rows {
            let count: usize = row.dims.iter().product();
            total += count;

            let spec = self
                .specs
                .iter()
                .rev()
                .find(|spec| spec.param_group().matches(&row.id, Some(row.path.as_str())));

            let owner = match spec {
                None => "adamw".to_string(),
                Some(spec) => {
                    let muon_width: usize =
                        spec.segments.iter().filter(|s| s.muon).map(|s| s.width).sum();
                    on_muon += muon_width * row.dims[0];
                    let segments = spec
                        .segments
                        .iter()
                        .map(|s| {
                            let mark = if s.muon { "" } else { "*" };
                            format!("{}:{}{mark}", s.name, s.width)
                        })
                        .collect::<Vec<_>>()
                        .join(" ");
                    format!("muon[{segments}]")
                }
            };

            let _ = writeln!(
                out,
                "{:>2}D {:<62} {:?} ({count})  {owner}",
                row.dims.len(),
                row.path,
                row.dims
            );
        }

        let share = if total == 0 {
            0.0
        } else {
            100.0 * on_muon as f64 / total as f64
        };
        let _ = writeln!(
            out,
            "total params: {total}; on muon: {on_muon} ({share:.1}%)  (* = segment left on adamw)"
        );
        out
    }
}

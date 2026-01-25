use std::sync::Arc;

use arrow::array::RecordBatch;
use datafusion_common::{Result, ScalarValue};
use datafusion_execution::{SendableRecordBatchStream, TaskContext};
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr_common::{
    metrics::ExecutionPlanMetricsSet, physical_expr::ExprExecutionContext,
};
use futures::TryStreamExt;
use parking_lot::Mutex;

use crate::ExecutionPlan;

/// Describes a plan execution context.
#[derive(Default)]
pub struct PlanExecutionContext {
    /// User passed parameters.
    external_params: Arc<[ScalarValue]>,
    /// Associated task context.
    task_context: Arc<TaskContext>,
}

impl PlanExecutionContext {
    /// Make a new [`PlanExecutionContext`].  
    pub fn new(
        task_context: Arc<TaskContext>,
        external_params: impl Into<Arc<[ScalarValue]>>,
    ) -> Self {
        Self {
            external_params: external_params.into(),
            task_context,
        }
    }

    /// Build a context for the particular node.
    pub fn build_node_context(
        self: &Arc<Self>,
        node: Arc<dyn ExecutionPlan>,
    ) -> Result<PlanNodeExecutionContext> {
        let expr_context = ExprExecutionContext::new(Arc::clone(&self.external_params));
        let exprs = node
            .exprs()
            .iter()
            .map(|expr| Arc::clone(expr).execute(&expr_context))
            .collect::<Result<Vec<_>>>()?;
        let num_children = node.children().len();
        Ok(PlanNodeExecutionContext {
            plan_context: Arc::clone(self),
            node,
            metrics: ExecutionPlanMetricsSet::new(),
            children: Mutex::new((0..num_children).map(|_| None).collect()),
            exprs,
        })
    }

    /// Project associated task context.
    pub fn task_context(&self) -> &Arc<TaskContext> {
        &self.task_context
    }

    pub fn execute(
        self: &Arc<Self>,
        plan: &Arc<dyn ExecutionPlan>,
    ) -> Result<SendableRecordBatchStream> {
        let node_context = self.build_node_context(Arc::clone(plan)).map(Arc::new)?;
        plan.execute_with(0, &node_context)
    }

    pub async fn collect(
        self: &Arc<Self>,
        plan: &Arc<dyn ExecutionPlan>,
    ) -> Result<Vec<RecordBatch>> {
        self.execute(plan)?.try_collect().await
    }
}

/// Describes a particular node execution context.
pub struct PlanNodeExecutionContext {
    plan_context: Arc<PlanExecutionContext>,
    node: Arc<dyn ExecutionPlan>,
    metrics: ExecutionPlanMetricsSet,
    /// Context for each plan child initialized lazy.
    children: Mutex<Box<[Option<Arc<PlanNodeExecutionContext>>]>>,
    /// Executable form of expressions.
    exprs: Vec<Arc<dyn PhysicalExpr>>,
}

impl PlanNodeExecutionContext {
    /// Execute child `idx` of the current node.
    pub fn execute_child(
        &self,
        idx: usize,
        partition: usize,
    ) -> Result<SendableRecordBatchStream> {
        let child_context = self.get_or_build_child(idx)?;
        self.node.children()[idx].execute_with(partition, &child_context)
    }

    /// Project a plan context.
    pub fn plan_context(&self) -> &Arc<PlanExecutionContext> {
        &self.plan_context
    }

    /// Project executable expressions.
    pub fn exprs(&self) -> &[Arc<dyn PhysicalExpr>] {
        &self.exprs
    }

    /// Project a plan node.
    pub fn node(&self) -> &Arc<dyn ExecutionPlan> {
        &self.node
    }

    /// Return metrics to fill.
    pub fn metrics(&self) -> &ExecutionPlanMetricsSet {
        &self.metrics
    }

    fn get_or_build_child(&self, idx: usize) -> Result<Arc<PlanNodeExecutionContext>> {
        let mut children = self.children.lock();
        if let Some(context) = children[idx].as_ref() {
            return Ok(Arc::clone(context));
        }
        let context = self
            .plan_context
            .build_node_context(Arc::clone(&self.node.children()[idx]))
            .map(Arc::new)?;
        children[idx] = Some(Arc::clone(&context));
        Ok(context)
    }
}

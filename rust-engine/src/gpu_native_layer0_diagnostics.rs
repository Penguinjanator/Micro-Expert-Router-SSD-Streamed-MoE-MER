use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::backend::GpuDeviceIdentity;
use crate::gpu_native_diagnostics::CaseEvidence;
use crate::gpu_native_token_loop::{GpuNativeModelGeometry, GpuNativeTokenLoopError};
use crate::numerical_diagnostics::{compare_vectors, VectorComparisonEvidence};
use crate::qualification::{BuildProvenance, ExpertMetadataEvidence};

pub const LAYER0_SCHEMA_VERSION: &str = "mer.gpu-native-layer0-attention-first-divergence.v1";
pub const LAYER0_MODE: &str = "diagnose-gpu-native-layer0-attention-first-divergence";

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "stage", rename_all = "snake_case")]
pub enum Layer0AttentionStage {
    Embedding,
    AttentionPreNorm,
    QRaw,
    KRaw,
    VRaw,
    QAfterNorm,
    KAfterNorm,
    QAfterRope,
    KAfterRope,
    AttentionContext,
    OProjection,
    PostAttentionResidual,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Layer0AttentionDiagnosticTraceLayout {
    pub d_model: usize,
    pub q_width: usize,
    pub kv_width: usize,

    pub embedding_offset: usize,
    pub embedding_bytes: usize,

    pub attention_pre_norm_offset: usize,
    pub attention_pre_norm_bytes: usize,

    pub q_raw_offset: usize,
    pub q_raw_bytes: usize,

    pub k_raw_offset: usize,
    pub k_raw_bytes: usize,

    pub v_raw_offset: usize,
    pub v_raw_bytes: usize,

    pub q_after_norm_offset: usize,
    pub q_after_norm_bytes: usize,

    pub k_after_norm_offset: usize,
    pub k_after_norm_bytes: usize,

    pub q_after_rope_offset: usize,
    pub q_after_rope_bytes: usize,

    pub k_after_rope_offset: usize,
    pub k_after_rope_bytes: usize,

    pub attention_context_offset: usize,
    pub attention_context_bytes: usize,

    pub o_projection_offset: usize,
    pub o_projection_bytes: usize,

    pub post_attention_residual_offset: usize,
    pub post_attention_residual_bytes: usize,

    pub status_offset: usize,
    pub status_bytes: usize,

    pub total_bytes: u64,
}

impl Layer0AttentionDiagnosticTraceLayout {
    pub fn try_new(
        d_model: usize,
        q_width: usize,
        kv_width: usize,
    ) -> Result<Self, GpuNativeTokenLoopError> {
        if d_model == 0 {
            return Err(GpuNativeTokenLoopError::InvalidBoundaryReport {
                detail: "d_model must be > 0".into(),
            });
        }
        if q_width == 0 {
            return Err(GpuNativeTokenLoopError::InvalidBoundaryReport {
                detail: "q_width must be > 0".into(),
            });
        }
        if kv_width == 0 {
            return Err(GpuNativeTokenLoopError::InvalidBoundaryReport {
                detail: "kv_width must be > 0".into(),
            });
        }

        let f32_bytes = std::mem::size_of::<f32>();
        let u32_bytes = std::mem::size_of::<u32>();

        let d_model_bytes = d_model.checked_mul(f32_bytes).ok_or_else(|| {
            GpuNativeTokenLoopError::InvalidBoundaryReport {
                detail: "d_model_bytes overflow".into(),
            }
        })?;
        let q_bytes = q_width.checked_mul(f32_bytes).ok_or_else(|| {
            GpuNativeTokenLoopError::InvalidBoundaryReport {
                detail: "q_bytes overflow".into(),
            }
        })?;
        let kv_bytes = kv_width.checked_mul(f32_bytes).ok_or_else(|| {
            GpuNativeTokenLoopError::InvalidBoundaryReport {
                detail: "kv_bytes overflow".into(),
            }
        })?;

        let embedding_offset = 0usize;
        let embedding_bytes = d_model_bytes;

        let attention_pre_norm_offset =
            embedding_offset
                .checked_add(embedding_bytes)
                .ok_or_else(|| GpuNativeTokenLoopError::InvalidBoundaryReport {
                    detail: "attention_pre_norm_offset overflow".into(),
                })?;
        let attention_pre_norm_bytes = d_model_bytes;

        let q_raw_offset = attention_pre_norm_offset
            .checked_add(attention_pre_norm_bytes)
            .ok_or_else(|| GpuNativeTokenLoopError::InvalidBoundaryReport {
                detail: "q_raw_offset overflow".into(),
            })?;
        let q_raw_bytes = q_bytes;

        let k_raw_offset = q_raw_offset.checked_add(q_raw_bytes).ok_or_else(|| {
            GpuNativeTokenLoopError::InvalidBoundaryReport {
                detail: "k_raw_offset overflow".into(),
            }
        })?;
        let k_raw_bytes = kv_bytes;

        let v_raw_offset = k_raw_offset.checked_add(k_raw_bytes).ok_or_else(|| {
            GpuNativeTokenLoopError::InvalidBoundaryReport {
                detail: "v_raw_offset overflow".into(),
            }
        })?;
        let v_raw_bytes = kv_bytes;

        let q_after_norm_offset = v_raw_offset.checked_add(v_raw_bytes).ok_or_else(|| {
            GpuNativeTokenLoopError::InvalidBoundaryReport {
                detail: "q_after_norm_offset overflow".into(),
            }
        })?;
        let q_after_norm_bytes = q_bytes;

        let k_after_norm_offset = q_after_norm_offset
            .checked_add(q_after_norm_bytes)
            .ok_or_else(|| GpuNativeTokenLoopError::InvalidBoundaryReport {
                detail: "k_after_norm_offset overflow".into(),
            })?;
        let k_after_norm_bytes = kv_bytes;

        let q_after_rope_offset = k_after_norm_offset
            .checked_add(k_after_norm_bytes)
            .ok_or_else(|| GpuNativeTokenLoopError::InvalidBoundaryReport {
                detail: "q_after_rope_offset overflow".into(),
            })?;
        let q_after_rope_bytes = q_bytes;

        let k_after_rope_offset = q_after_rope_offset
            .checked_add(q_after_rope_bytes)
            .ok_or_else(|| GpuNativeTokenLoopError::InvalidBoundaryReport {
                detail: "k_after_rope_offset overflow".into(),
            })?;
        let k_after_rope_bytes = kv_bytes;

        let attention_context_offset = k_after_rope_offset
            .checked_add(k_after_rope_bytes)
            .ok_or_else(|| GpuNativeTokenLoopError::InvalidBoundaryReport {
                detail: "attention_context_offset overflow".into(),
            })?;
        let attention_context_bytes = q_bytes;

        let o_projection_offset = attention_context_offset
            .checked_add(attention_context_bytes)
            .ok_or_else(|| GpuNativeTokenLoopError::InvalidBoundaryReport {
                detail: "o_projection_offset overflow".into(),
            })?;
        let o_projection_bytes = d_model_bytes;

        let post_attention_residual_offset = o_projection_offset
            .checked_add(o_projection_bytes)
            .ok_or_else(|| GpuNativeTokenLoopError::InvalidBoundaryReport {
                detail: "post_attention_residual_offset overflow".into(),
            })?;
        let post_attention_residual_bytes = d_model_bytes;

        let status_offset = post_attention_residual_offset
            .checked_add(post_attention_residual_bytes)
            .ok_or_else(|| GpuNativeTokenLoopError::InvalidBoundaryReport {
                detail: "status_offset overflow".into(),
            })?;
        let status_bytes = u32_bytes;

        let total_bytes_usize = status_offset.checked_add(status_bytes).ok_or_else(|| {
            GpuNativeTokenLoopError::InvalidBoundaryReport {
                detail: "total_bytes overflow".into(),
            }
        })?;

        Ok(Self {
            d_model,
            q_width,
            kv_width,
            embedding_offset,
            embedding_bytes,
            attention_pre_norm_offset,
            attention_pre_norm_bytes,
            q_raw_offset,
            q_raw_bytes,
            k_raw_offset,
            k_raw_bytes,
            v_raw_offset,
            v_raw_bytes,
            q_after_norm_offset,
            q_after_norm_bytes,
            k_after_norm_offset,
            k_after_norm_bytes,
            q_after_rope_offset,
            q_after_rope_bytes,
            k_after_rope_offset,
            k_after_rope_bytes,
            attention_context_offset,
            attention_context_bytes,
            o_projection_offset,
            o_projection_bytes,
            post_attention_residual_offset,
            post_attention_residual_bytes,
            status_offset,
            status_bytes,
            total_bytes: total_bytes_usize as u64,
        })
    }

    pub fn parse(&self, bytes: &[u8]) -> Result<Layer0AttentionDiagnosticTrace, String> {
        if bytes.len() < self.total_bytes as usize {
            return Err(format!(
                "buffer too short for Layer0 diagnostic trace: expected >= {} bytes, got {}",
                self.total_bytes,
                bytes.len()
            ));
        }

        let parse_f32_vec = |offset: usize, len: usize| -> Vec<f32> {
            let byte_slice = &bytes[offset..offset + len * 4];
            byte_slice
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                .collect()
        };

        let embedding = parse_f32_vec(self.embedding_offset, self.d_model);
        let attention_pre_norm = parse_f32_vec(self.attention_pre_norm_offset, self.d_model);
        let q_raw = parse_f32_vec(self.q_raw_offset, self.q_width);
        let k_raw = parse_f32_vec(self.k_raw_offset, self.kv_width);
        let v_raw = parse_f32_vec(self.v_raw_offset, self.kv_width);
        let q_after_norm = parse_f32_vec(self.q_after_norm_offset, self.q_width);
        let k_after_norm = parse_f32_vec(self.k_after_norm_offset, self.kv_width);
        let q_after_rope = parse_f32_vec(self.q_after_rope_offset, self.q_width);
        let k_after_rope = parse_f32_vec(self.k_after_rope_offset, self.kv_width);
        let attention_context = parse_f32_vec(self.attention_context_offset, self.q_width);
        let o_projection = parse_f32_vec(self.o_projection_offset, self.d_model);
        let post_attention_residual =
            parse_f32_vec(self.post_attention_residual_offset, self.d_model);

        let status = u32::from_le_bytes(
            bytes[self.status_offset..self.status_offset + 4]
                .try_into()
                .unwrap(),
        );

        Ok(Layer0AttentionDiagnosticTrace {
            embedding,
            attention_pre_norm,
            q_raw,
            k_raw,
            v_raw,
            q_after_norm,
            k_after_norm,
            q_after_rope,
            k_after_rope,
            attention_context,
            o_projection,
            post_attention_residual,
            status,
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Layer0AttentionDiagnosticTrace {
    pub embedding: Vec<f32>,
    pub attention_pre_norm: Vec<f32>,
    pub q_raw: Vec<f32>,
    pub k_raw: Vec<f32>,
    pub v_raw: Vec<f32>,
    pub q_after_norm: Vec<f32>,
    pub k_after_norm: Vec<f32>,
    pub q_after_rope: Vec<f32>,
    pub k_after_rope: Vec<f32>,
    pub attention_context: Vec<f32>,
    pub o_projection: Vec<f32>,
    pub post_attention_residual: Vec<f32>,
    pub status: u32,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Layer0AttentionStageComparison {
    pub stage: Layer0AttentionStage,
    pub exact_match: bool,
    pub vector_comparison: VectorComparisonEvidence,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Layer0AttentionFirstDivergenceEvidence {
    pub position: usize,
    pub token_id: u32,
    pub stage: Layer0AttentionStage,
    pub reason: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_absolute_error: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rms_error: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub worst_index: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub worst_reference_value: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub worst_gpu_native_value: Option<f32>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Layer0AttentionPositionComparison {
    pub position: usize,
    pub token_id: u32,
    pub exact_match: bool,
    pub first_divergence: Option<Layer0AttentionFirstDivergenceEvidence>,
    pub stages: Vec<Layer0AttentionStageComparison>,
}

pub fn compare_layer0_attention_traces(
    position: usize,
    token_id: u32,
    reference: &crate::transformer::Layer0AttentionCpuDiagnosticSink,
    gpu_native: &Layer0AttentionDiagnosticTrace,
) -> Result<Layer0AttentionPositionComparison, String> {
    let mut stages = Vec::with_capacity(12);
    let mut first_divergence: Option<Layer0AttentionFirstDivergenceEvidence> = None;

    let mut compare_one_stage = |stage: Layer0AttentionStage,
                                 ref_opt: Option<&Vec<f32>>,
                                 gpu_vec: &[f32]|
     -> Result<(), String> {
        let ref_vec = ref_opt.ok_or_else(|| {
            format!("reference diagnostic trace missing stage {stage:?} at position {position}")
        })?;
        let comp = compare_vectors(ref_vec, gpu_vec)?;
        let exact = comp.exact_f32_bits;
        let nonfinite = comp.cpu_nonfinite_count > 0 || comp.hybrid_nonfinite_count > 0;

        if (!exact || nonfinite) && first_divergence.is_none() {
            let reason = if nonfinite {
                format!(
                    "nonfinite value: reference nonfinite={}, gpu_native nonfinite={}",
                    comp.cpu_nonfinite_count, comp.hybrid_nonfinite_count
                )
            } else {
                format!(
                    "exact f32 bits differ (max absolute error {:?}, RMS error {:?})",
                    comp.max_absolute_error, comp.rms_error
                )
            };

            first_divergence = Some(Layer0AttentionFirstDivergenceEvidence {
                position,
                token_id,
                stage: stage.clone(),
                reason,
                max_absolute_error: comp.max_absolute_error,
                rms_error: comp.rms_error,
                worst_index: comp.max_error.as_ref().map(|e| e.index),
                worst_reference_value: comp.max_error.as_ref().and_then(|e| e.cpu.value),
                worst_gpu_native_value: comp.max_error.as_ref().and_then(|e| e.hybrid.value),
            });
        }

        stages.push(Layer0AttentionStageComparison {
            stage,
            exact_match: exact && !nonfinite,
            vector_comparison: comp,
        });
        Ok(())
    };

    compare_one_stage(
        Layer0AttentionStage::Embedding,
        reference.embedding.as_ref(),
        &gpu_native.embedding,
    )?;
    compare_one_stage(
        Layer0AttentionStage::AttentionPreNorm,
        reference.attention_pre_norm.as_ref(),
        &gpu_native.attention_pre_norm,
    )?;
    compare_one_stage(
        Layer0AttentionStage::QRaw,
        reference.q_raw.as_ref(),
        &gpu_native.q_raw,
    )?;
    compare_one_stage(
        Layer0AttentionStage::KRaw,
        reference.k_raw.as_ref(),
        &gpu_native.k_raw,
    )?;
    compare_one_stage(
        Layer0AttentionStage::VRaw,
        reference.v_raw.as_ref(),
        &gpu_native.v_raw,
    )?;
    compare_one_stage(
        Layer0AttentionStage::QAfterNorm,
        reference.q_after_norm.as_ref(),
        &gpu_native.q_after_norm,
    )?;
    compare_one_stage(
        Layer0AttentionStage::KAfterNorm,
        reference.k_after_norm.as_ref(),
        &gpu_native.k_after_norm,
    )?;
    compare_one_stage(
        Layer0AttentionStage::QAfterRope,
        reference.q_after_rope.as_ref(),
        &gpu_native.q_after_rope,
    )?;
    compare_one_stage(
        Layer0AttentionStage::KAfterRope,
        reference.k_after_rope.as_ref(),
        &gpu_native.k_after_rope,
    )?;
    compare_one_stage(
        Layer0AttentionStage::AttentionContext,
        reference.attention_context.as_ref(),
        &gpu_native.attention_context,
    )?;
    compare_one_stage(
        Layer0AttentionStage::OProjection,
        reference.o_projection.as_ref(),
        &gpu_native.o_projection,
    )?;
    compare_one_stage(
        Layer0AttentionStage::PostAttentionResidual,
        reference.post_attention_residual.as_ref(),
        &gpu_native.post_attention_residual,
    )?;

    let exact_match = first_divergence.is_none();
    Ok(Layer0AttentionPositionComparison {
        position,
        token_id,
        exact_match,
        first_divergence,
        stages,
    })
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Slice11bCrossCheckEvidence {
    pub report_path: String,
    pub cross_check_status: String,
    pub identity_match: bool,
    pub stage_evidence_match: bool,
    pub slice11b_schema: Option<String>,
    pub slice11b_mode: Option<String>,
    pub case_name: Option<String>,
    pub prompt_sha256: Option<String>,
    pub prompt_token_ids_sha256: Option<String>,
    pub resolved_config_sha256: Option<String>,
    pub expected_adapter_name: Option<String>,
    pub slice11b_layer0_post_attn_max_abs: Option<f32>,
    pub slice11b_layer0_post_attn_rms: Option<f64>,
    pub slice11c_layer0_post_attn_max_abs: Option<f32>,
    pub slice11c_layer0_post_attn_rms: Option<f64>,
    pub discrepancy: Option<String>,
}

pub fn first_layer0_divergence_across_positions(
    position_comparisons: &[Layer0AttentionPositionComparison],
) -> (
    Option<usize>,
    Option<Layer0AttentionFirstDivergenceEvidence>,
) {
    for pos_comp in position_comparisons {
        if !pos_comp.exact_match {
            if let Some(ref div) = pos_comp.first_divergence {
                return (Some(pos_comp.position), Some(div.clone()));
            }
        }
    }
    (None, None)
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Layer0AttentionFirstDivergenceReport {
    pub schema_version: String,
    pub mode: String,
    pub diagnostic_complete: bool,
    pub qualification_pass: bool,
    pub failure: Option<String>,

    pub provenance: BuildProvenance,
    pub build_git_sha: String,
    pub executable_sha256: String,
    pub resolved_config_sha256: String,
    pub model_geometry: GpuNativeModelGeometry,
    pub expert_metadata: ExpertMetadataEvidence,
    pub case: CaseEvidence,
    pub prompt_token_ids_sha256: String,
    pub prompt_token_ids: Vec<u32>,
    pub prompt_token_count: usize,
    pub expected_adapter_name: String,
    pub actual_adapter: Option<GpuDeviceIdentity>,

    pub earliest_divergent_position: Option<usize>,
    pub first_internal_divergence: Option<Layer0AttentionFirstDivergenceEvidence>,
    pub position_comparisons: Vec<Layer0AttentionPositionComparison>,
    pub final_position_post_attention: Option<Layer0AttentionStageComparison>,
    pub slice11b_cross_check: Option<Slice11bCrossCheckEvidence>,
}

impl Layer0AttentionFirstDivergenceReport {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        provenance: BuildProvenance,
        build_git_sha: String,
        executable_sha256: String,
        resolved_config_sha256: String,
        model_geometry: GpuNativeModelGeometry,
        expert_metadata: ExpertMetadataEvidence,
        case: CaseEvidence,
        prompt_token_ids_sha256: String,
        prompt_token_ids: Vec<u32>,
        expected_adapter_name: String,
    ) -> Self {
        let prompt_token_count = prompt_token_ids.len();
        Self {
            schema_version: LAYER0_SCHEMA_VERSION.to_string(),
            mode: LAYER0_MODE.to_string(),
            diagnostic_complete: false,
            qualification_pass: false,
            failure: None,
            provenance,
            build_git_sha,
            executable_sha256,
            resolved_config_sha256,
            model_geometry,
            expert_metadata,
            case,
            prompt_token_ids_sha256,
            prompt_token_ids,
            prompt_token_count,
            expected_adapter_name,
            actual_adapter: None,
            earliest_divergent_position: None,
            first_internal_divergence: None,
            position_comparisons: Vec::new(),
            final_position_post_attention: None,
            slice11b_cross_check: None,
        }
    }

    pub fn fail(&mut self, reason: String) {
        self.diagnostic_complete = true;
        self.qualification_pass = false;
        self.failure = Some(reason);
    }
}

pub fn cross_check_slice11b_report(
    slice11b_path: &Path,
    current_report: &Layer0AttentionFirstDivergenceReport,
    final_post_attn: &Layer0AttentionStageComparison,
) -> Result<Slice11bCrossCheckEvidence, String> {
    let file_str = std::fs::read_to_string(slice11b_path).map_err(|e| {
        format!(
            "failed to read Slice11B report at {}: {e}",
            slice11b_path.display()
        )
    })?;
    let val: serde_json::Value = serde_json::from_str(&file_str)
        .map_err(|e| format!("failed to parse Slice11B report JSON: {e}"))?;

    let slice11b_schema = val
        .get("schema_version")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let slice11b_mode = val
        .get("mode")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let case_name = val
        .get("case")
        .and_then(|c| c.get("case_name"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let prompt_sha256 = val
        .get("case")
        .and_then(|c| c.get("prompt_sha256"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let prompt_token_count = val
        .get("case")
        .and_then(|c| c.get("prompt_token_count"))
        .and_then(|v| v.as_u64())
        .map(|u| u as usize);
    let prompt_token_ids_sha256 = val
        .get("prompt_token_ids_sha256")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let resolved_config_sha256 = val
        .get("resolved_config_sha256")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let expected_adapter_name = val
        .get("expected_adapter_name")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());
    let diagnostic_complete = val.get("diagnostic_complete").and_then(|v| v.as_bool());
    let failure = val.get("failure");

    let mut identity_discrepancy = None;
    if slice11b_schema.as_deref() != Some("mer.gpu-native-q4-first-divergence-diagnostic.v1") {
        identity_discrepancy = Some(format!(
            "Slice11B schema_version mismatch: expected 'mer.gpu-native-q4-first-divergence-diagnostic.v1', got {:?}",
            slice11b_schema
        ));
    } else if slice11b_mode.as_deref() != Some("diagnose-gpu-native-q4-first-divergence") {
        identity_discrepancy = Some(format!(
            "Slice11B mode mismatch: expected 'diagnose-gpu-native-q4-first-divergence', got {:?}",
            slice11b_mode
        ));
    } else if diagnostic_complete != Some(true) {
        identity_discrepancy = Some(format!(
            "Slice11B diagnostic_complete mismatch: expected true, got {:?}",
            diagnostic_complete
        ));
    } else if failure.is_some() && !failure.unwrap().is_null() {
        identity_discrepancy = Some(format!("Slice11B report recorded a failure: {:?}", failure));
    } else if case_name.as_deref() != Some(&current_report.case.case_name) {
        identity_discrepancy = Some(format!(
            "Slice11B case.case_name mismatch: expected {:?}, got {:?}",
            current_report.case.case_name, case_name
        ));
    } else if prompt_sha256.as_deref() != Some(&current_report.case.prompt_sha256) {
        identity_discrepancy = Some(format!(
            "Slice11B case.prompt_sha256 mismatch: expected {:?}, got {:?}",
            current_report.case.prompt_sha256, prompt_sha256
        ));
    } else if prompt_token_count != Some(current_report.case.prompt_token_count) {
        identity_discrepancy = Some(format!(
            "Slice11B case.prompt_token_count mismatch: expected {:?}, got {:?}",
            current_report.case.prompt_token_count, prompt_token_count
        ));
    } else if prompt_token_ids_sha256.as_deref() != Some(&current_report.prompt_token_ids_sha256) {
        identity_discrepancy = Some(format!(
            "Slice11B prompt_token_ids_sha256 mismatch: expected {:?}, got {:?}",
            current_report.prompt_token_ids_sha256, prompt_token_ids_sha256
        ));
    } else if resolved_config_sha256.as_deref() != Some(&current_report.resolved_config_sha256) {
        identity_discrepancy = Some(format!(
            "Slice11B resolved_config_sha256 mismatch: expected {:?}, got {:?}",
            current_report.resolved_config_sha256, resolved_config_sha256
        ));
    } else if expected_adapter_name.as_deref() != Some(&current_report.expected_adapter_name) {
        identity_discrepancy = Some(format!(
            "Slice11B expected_adapter_name mismatch: expected {:?}, got {:?}",
            current_report.expected_adapter_name, expected_adapter_name
        ));
    } else {
        let current_geom_val =
            serde_json::to_value(&current_report.model_geometry).map_err(|e| e.to_string())?;
        if let Some(old_geom) = val.get("model_geometry") {
            if old_geom != &current_geom_val {
                identity_discrepancy = Some(format!(
                    "Slice11B model_geometry mismatch: expected {}, got {}",
                    current_geom_val, old_geom
                ));
            }
        } else {
            identity_discrepancy = Some("Slice11B report missing 'model_geometry'".to_string());
        }
    }

    let identity_match = identity_discrepancy.is_none();

    let stages = val
        .get("stages")
        .and_then(|s| s.as_array())
        .ok_or_else(|| "Slice11B report missing 'stages' array".to_string())?;

    let mut matched_stages: Vec<&serde_json::Value> = Vec::new();
    for s in stages {
        if let Some(stage_val) = s.get("stage") {
            if stage_val.get("stage").and_then(|v| v.as_str()) == Some("layer_post_attention")
                && stage_val.get("layer").and_then(|l| l.as_u64()) == Some(0)
            {
                matched_stages.push(s);
            }
        }
    }

    if matched_stages.len() != 1 {
        let msg = format!(
            "Slice11B report must contain exactly 1 LayerPostAttention {{ layer: 0 }} stage, found {}",
            matched_stages.len()
        );
        return Ok(Slice11bCrossCheckEvidence {
            report_path: slice11b_path.display().to_string(),
            cross_check_status: "mismatch".to_string(),
            identity_match,
            stage_evidence_match: false,
            slice11b_schema,
            slice11b_mode,
            case_name,
            prompt_sha256,
            prompt_token_ids_sha256,
            resolved_config_sha256,
            expected_adapter_name,
            slice11b_layer0_post_attn_max_abs: None,
            slice11b_layer0_post_attn_rms: None,
            slice11c_layer0_post_attn_max_abs: final_post_attn.vector_comparison.max_absolute_error,
            slice11c_layer0_post_attn_rms: final_post_attn.vector_comparison.rms_error,
            discrepancy: Some(identity_discrepancy.unwrap_or(msg)),
        });
    }

    let slice11b_stage = matched_stages[0];
    let slice11b_comp = slice11b_stage
        .get("vector_comparison")
        .ok_or_else(|| "Slice11B layer 0 stage missing 'vector_comparison'".to_string())?;

    let b_max_abs = slice11b_comp
        .get("max_absolute_error")
        .and_then(|v| v.as_f64())
        .map(|f| f as f32);
    let b_rms = slice11b_comp.get("rms_error").and_then(|v| v.as_f64());

    let c_max_abs = final_post_attn.vector_comparison.max_absolute_error;
    let c_rms = final_post_attn.vector_comparison.rms_error;

    let b_exact_match = slice11b_stage
        .get("exact_match")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let c_exact_match = final_post_attn.exact_match;

    let current_comp_val = serde_json::to_value(&final_post_attn.vector_comparison)
        .map_err(|e| format!("failed to serialize Slice11C vector_comparison: {e}"))?;

    let mut stage_evidence_discrepancy = None;
    if b_exact_match != c_exact_match {
        stage_evidence_discrepancy = Some(format!(
            "exact_match mismatch: Slice11B={}, Slice11C={}",
            b_exact_match, c_exact_match
        ));
    } else if slice11b_comp != &current_comp_val {
        stage_evidence_discrepancy = Some(format!(
            "vector_comparison JSON mismatch: Slice11B={}, Slice11C={}",
            slice11b_comp, current_comp_val
        ));
    }

    let stage_evidence_match = stage_evidence_discrepancy.is_none();
    let discrepancy = identity_discrepancy.or(stage_evidence_discrepancy);
    let verified = identity_match && stage_evidence_match;

    Ok(Slice11bCrossCheckEvidence {
        report_path: slice11b_path.display().to_string(),
        cross_check_status: if verified {
            "verified_match".to_string()
        } else {
            "mismatch".to_string()
        },
        identity_match,
        stage_evidence_match,
        slice11b_schema,
        slice11b_mode,
        case_name,
        prompt_sha256,
        prompt_token_ids_sha256,
        resolved_config_sha256,
        expected_adapter_name,
        slice11b_layer0_post_attn_max_abs: b_max_abs,
        slice11b_layer0_post_attn_rms: b_rms,
        slice11c_layer0_post_attn_max_abs: c_max_abs,
        slice11c_layer0_post_attn_rms: c_rms,
        discrepancy,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layer0_trace_layout_offset_arithmetic_is_exact_and_4byte_aligned() {
        let layout = Layer0AttentionDiagnosticTraceLayout::try_new(2048, 4096, 512).unwrap();

        assert_eq!(layout.embedding_offset % 4, 0);
        assert_eq!(layout.attention_pre_norm_offset % 4, 0);
        assert_eq!(layout.q_raw_offset % 4, 0);
        assert_eq!(layout.k_raw_offset % 4, 0);
        assert_eq!(layout.v_raw_offset % 4, 0);
        assert_eq!(layout.q_after_norm_offset % 4, 0);
        assert_eq!(layout.k_after_norm_offset % 4, 0);
        assert_eq!(layout.q_after_rope_offset % 4, 0);
        assert_eq!(layout.k_after_rope_offset % 4, 0);
        assert_eq!(layout.attention_context_offset % 4, 0);
        assert_eq!(layout.o_projection_offset % 4, 0);
        assert_eq!(layout.post_attention_residual_offset % 4, 0);
        assert_eq!(layout.status_offset % 4, 0);

        assert_eq!(layout.embedding_bytes, 2048 * 4);
        assert_eq!(layout.attention_pre_norm_bytes, 2048 * 4);
        assert_eq!(layout.q_raw_bytes, 4096 * 4);
        assert_eq!(layout.k_raw_bytes, 512 * 4);
        assert_eq!(layout.v_raw_bytes, 512 * 4);
        assert_eq!(layout.q_after_norm_bytes, 4096 * 4);
        assert_eq!(layout.k_after_norm_bytes, 512 * 4);
        assert_eq!(layout.q_after_rope_bytes, 4096 * 4);
        assert_eq!(layout.k_after_rope_bytes, 512 * 4);
        assert_eq!(layout.attention_context_bytes, 4096 * 4);
        assert_eq!(layout.o_projection_bytes, 2048 * 4);
        assert_eq!(layout.post_attention_residual_bytes, 2048 * 4);
        assert_eq!(layout.status_bytes, 4);

        let sum_bytes = (2048 * 4)
            + (2048 * 4)
            + (4096 * 4)
            + (512 * 4)
            + (512 * 4)
            + (4096 * 4)
            + (512 * 4)
            + (4096 * 4)
            + (512 * 4)
            + (4096 * 4)
            + (2048 * 4)
            + (2048 * 4)
            + 4;
        assert_eq!(layout.total_bytes, sum_bytes as u64);
    }

    #[test]
    fn layer0_trace_layout_parse_roundtrip() {
        let layout = Layer0AttentionDiagnosticTraceLayout::try_new(4, 8, 2).unwrap();
        let mut buffer = vec![0u8; layout.total_bytes as usize];

        let q_vals = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        for (i, &v) in q_vals.iter().enumerate() {
            let off = layout.q_raw_offset + i * 4;
            buffer[off..off + 4].copy_from_slice(&v.to_le_bytes());
        }

        buffer[layout.status_offset..layout.status_offset + 4]
            .copy_from_slice(&42u32.to_le_bytes());

        let trace = layout.parse(&buffer).unwrap();
        assert_eq!(trace.q_raw, q_vals);
        assert_eq!(trace.status, 42);
        assert_eq!(trace.embedding.len(), 4);
        assert_eq!(trace.k_raw.len(), 2);
    }

    fn sample_cpu_sink(
        d_model: usize,
        q_width: usize,
        kv_width: usize,
    ) -> crate::transformer::Layer0AttentionCpuDiagnosticSink {
        crate::transformer::Layer0AttentionCpuDiagnosticSink {
            embedding: Some(vec![1.0; d_model]),
            attention_pre_norm: Some(vec![2.0; d_model]),
            q_raw: Some(vec![3.0; q_width]),
            k_raw: Some(vec![4.0; kv_width]),
            v_raw: Some(vec![5.0; kv_width]),
            q_after_norm: Some(vec![6.0; q_width]),
            k_after_norm: Some(vec![7.0; kv_width]),
            q_after_rope: Some(vec![8.0; q_width]),
            k_after_rope: Some(vec![9.0; kv_width]),
            attention_context: Some(vec![10.0; q_width]),
            o_projection: Some(vec![11.0; d_model]),
            post_attention_residual: Some(vec![12.0; d_model]),
        }
    }

    fn cpu_sink_to_gpu_trace(
        s: &crate::transformer::Layer0AttentionCpuDiagnosticSink,
    ) -> Layer0AttentionDiagnosticTrace {
        Layer0AttentionDiagnosticTrace {
            embedding: s.embedding.clone().unwrap(),
            attention_pre_norm: s.attention_pre_norm.clone().unwrap(),
            q_raw: s.q_raw.clone().unwrap(),
            k_raw: s.k_raw.clone().unwrap(),
            v_raw: s.v_raw.clone().unwrap(),
            q_after_norm: s.q_after_norm.clone().unwrap(),
            k_after_norm: s.k_after_norm.clone().unwrap(),
            q_after_rope: s.q_after_rope.clone().unwrap(),
            k_after_rope: s.k_after_rope.clone().unwrap(),
            attention_context: s.attention_context.clone().unwrap(),
            o_projection: s.o_projection.clone().unwrap(),
            post_attention_residual: s.post_attention_residual.clone().unwrap(),
            status: 0,
        }
    }

    #[test]
    fn comparator_exact_match_yields_no_divergence() {
        let cpu = sample_cpu_sink(4, 8, 2);
        let gpu = cpu_sink_to_gpu_trace(&cpu);

        let pos_comp = compare_layer0_attention_traces(0, 100, &cpu, &gpu).unwrap();
        assert!(pos_comp.exact_match);
        assert!(pos_comp.first_divergence.is_none());
        assert!(pos_comp.stages.iter().all(|s| s.exact_match));
    }

    #[test]
    fn comparator_flags_earliest_q_raw_divergence() {
        let cpu = sample_cpu_sink(4, 8, 2);
        let mut gpu = cpu_sink_to_gpu_trace(&cpu);
        gpu.q_raw[3] += 1e-5;
        gpu.attention_context[0] += 1.0;

        let pos_comp = compare_layer0_attention_traces(0, 100, &cpu, &gpu).unwrap();
        assert!(!pos_comp.exact_match);
        assert!(pos_comp.first_divergence.is_some());
        let div = pos_comp.first_divergence.unwrap();
        assert_eq!(div.stage, Layer0AttentionStage::QRaw);
        assert_eq!(div.worst_index, Some(3));
    }

    #[test]
    fn comparator_flags_attention_context_divergence_when_qkv_exact() {
        let cpu = sample_cpu_sink(4, 8, 2);
        let mut gpu = cpu_sink_to_gpu_trace(&cpu);
        gpu.attention_context[1] += 1e-4;

        let pos_comp = compare_layer0_attention_traces(1, 200, &cpu, &gpu).unwrap();
        assert!(!pos_comp.exact_match);
        let div = pos_comp.first_divergence.unwrap();
        assert_eq!(div.position, 1);
        assert_eq!(div.stage, Layer0AttentionStage::AttentionContext);
    }

    #[test]
    fn position_order_position_0_exact_position_1_divergent_selects_pos_1() {
        let cpu = sample_cpu_sink(4, 8, 2);
        let gpu0 = cpu_sink_to_gpu_trace(&cpu);
        let mut gpu1 = cpu_sink_to_gpu_trace(&cpu);
        gpu1.q_after_norm[0] += 1e-4;
        let mut gpu2 = cpu_sink_to_gpu_trace(&cpu);
        gpu2.q_raw[0] += 1e-3;

        let comp0 = compare_layer0_attention_traces(0, 10, &cpu, &gpu0).unwrap();
        let comp1 = compare_layer0_attention_traces(1, 11, &cpu, &gpu1).unwrap();
        let comp2 = compare_layer0_attention_traces(2, 12, &cpu, &gpu2).unwrap();

        let (pos, div) = first_layer0_divergence_across_positions(&[comp0, comp1, comp2]);
        assert_eq!(pos, Some(1));
        assert!(div.is_some());
        let div = div.unwrap();
        assert_eq!(div.position, 1);
        assert_eq!(div.stage, Layer0AttentionStage::QAfterNorm);
    }

    #[test]
    fn position_order_position_0_divergent_and_later_divergent_preserves_pos_0() {
        let cpu = sample_cpu_sink(4, 8, 2);
        let mut gpu0 = cpu_sink_to_gpu_trace(&cpu);
        gpu0.k_raw[0] += 1e-4;
        let mut gpu1 = cpu_sink_to_gpu_trace(&cpu);
        gpu1.attention_pre_norm[0] += 1e-3;

        let comp0 = compare_layer0_attention_traces(0, 10, &cpu, &gpu0).unwrap();
        let comp1 = compare_layer0_attention_traces(1, 11, &cpu, &gpu1).unwrap();

        let (pos, div) = first_layer0_divergence_across_positions(&[comp0, comp1]);
        assert_eq!(pos, Some(0));
        assert!(div.is_some());
        let div = div.unwrap();
        assert_eq!(div.position, 0);
        assert_eq!(div.stage, Layer0AttentionStage::KRaw);
    }

    struct TempFile {
        path: std::path::PathBuf,
    }

    impl TempFile {
        fn new(name: &str, content: &str) -> Self {
            static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let id = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "mer-test-slice11c-{name}-{id}-{}.json",
                std::process::id()
            ));
            let _ = std::fs::remove_file(&path);
            std::fs::write(&path, content).unwrap();
            Self { path }
        }
    }

    impl Drop for TempFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.path);
        }
    }

    fn sample_test_evidence() -> (
        Layer0AttentionFirstDivergenceReport,
        Layer0AttentionStageComparison,
    ) {
        let geom = GpuNativeModelGeometry {
            num_layers: 48,
            d_model: 2048,
            d_ff: 768,
            num_experts: 128,
            top_k: 8,
            num_heads: 32,
            num_kv_heads: 4,
            head_dim: 128,
            rope_dim: 128,
            vocab_size: 151936,
            max_seq_len: 2048,
            rms_eps: 1e-6,
            rope_base: 10000000.0,
        };
        let report = Layer0AttentionFirstDivergenceReport::new(
            BuildProvenance::embedded(),
            "git_sha_123".to_string(),
            "exec_sha_123".to_string(),
            "resolved_cfg_hash_123".to_string(),
            geom,
            ExpertMetadataEvidence {
                dtype: Some("q4_0".to_string()),
                q4_0_layout: Some("canonical".to_string()),
                conversion_mode: Some("native".to_string()),
                source: Some("checkpoint".to_string()),
                explicitly_synthetic: false,
            },
            CaseEvidence {
                case_name: "json-transformation".to_string(),
                prompt_sha256: "prompt_hash_123".to_string(),
                prompt_token_count: 4,
            },
            "prompt_token_ids_hash_123".to_string(),
            vec![1, 2, 3, 4],
            "NVIDIA L4".to_string(),
        );

        let comp = crate::numerical_diagnostics::compare_vectors(
            &[1.0, 2.0, 3.0, 4.0],
            &[1.001, 2.0, 3.0, 4.0],
        )
        .unwrap();

        let stage_comp = Layer0AttentionStageComparison {
            stage: Layer0AttentionStage::PostAttentionResidual,
            exact_match: false,
            vector_comparison: comp,
        };

        (report, stage_comp)
    }

    fn sample_slice11b_report_json(
        case_name: &str,
        prompt_token_ids_sha256: &str,
        geom: &GpuNativeModelGeometry,
        comp: &VectorComparisonEvidence,
    ) -> String {
        serde_json::json!({
            "schema_version": "mer.gpu-native-q4-first-divergence-diagnostic.v1",
            "mode": "diagnose-gpu-native-q4-first-divergence",
            "diagnostic_complete": true,
            "qualification_pass": false,
            "failure": null,
            "case": {
                "case_name": case_name,
                "prompt_sha256": "prompt_hash_123",
                "prompt_token_count": 4
            },
            "prompt_token_ids_sha256": prompt_token_ids_sha256,
            "resolved_config_sha256": "resolved_cfg_hash_123",
            "model_geometry": geom,
            "expected_adapter_name": "NVIDIA L4",
            "stages": [
                {
                    "stage": {
                        "stage": "layer_post_attention",
                        "layer": 0
                    },
                    "exact_match": comp.exact_f32_bits,
                    "tolerance_pass": false,
                    "vector_comparison": comp
                }
            ]
        })
        .to_string()
    }

    #[test]
    fn cross_check_exact_matching_slice11b_identity_and_evidence_passes() {
        let (report, stage_comp) = sample_test_evidence();
        let json_str = sample_slice11b_report_json(
            &report.case.case_name,
            &report.prompt_token_ids_sha256,
            &report.model_geometry,
            &stage_comp.vector_comparison,
        );
        let temp = TempFile::new("exact_match", &json_str);

        let res = cross_check_slice11b_report(&temp.path, &report, &stage_comp).unwrap();
        assert_eq!(res.cross_check_status, "verified_match");
        assert!(res.identity_match);
        assert!(res.stage_evidence_match);
        assert!(res.discrepancy.is_none());
    }

    #[test]
    fn cross_check_wrong_case_name_fails() {
        let (report, stage_comp) = sample_test_evidence();
        let json_str = sample_slice11b_report_json(
            "rust-generation",
            &report.prompt_token_ids_sha256,
            &report.model_geometry,
            &stage_comp.vector_comparison,
        );
        let temp = TempFile::new("wrong_case", &json_str);

        let res = cross_check_slice11b_report(&temp.path, &report, &stage_comp).unwrap();
        assert_eq!(res.cross_check_status, "mismatch");
        assert!(!res.identity_match);
        assert!(res.discrepancy.unwrap().contains("case.case_name"));
    }

    #[test]
    fn cross_check_wrong_prompt_token_hash_fails() {
        let (report, stage_comp) = sample_test_evidence();
        let json_str = sample_slice11b_report_json(
            &report.case.case_name,
            "wrong_prompt_hash",
            &report.model_geometry,
            &stage_comp.vector_comparison,
        );
        let temp = TempFile::new("wrong_prompt_hash", &json_str);

        let res = cross_check_slice11b_report(&temp.path, &report, &stage_comp).unwrap();
        assert_eq!(res.cross_check_status, "mismatch");
        assert!(!res.identity_match);
        assert!(res.discrepancy.unwrap().contains("prompt_token_ids_sha256"));
    }

    #[test]
    fn cross_check_wrong_model_geometry_fails() {
        let (report, stage_comp) = sample_test_evidence();
        let mut wrong_geom = report.model_geometry;
        wrong_geom.num_layers = 24; // differs from 48
        let json_str = sample_slice11b_report_json(
            &report.case.case_name,
            &report.prompt_token_ids_sha256,
            &wrong_geom,
            &stage_comp.vector_comparison,
        );
        let temp = TempFile::new("wrong_geom", &json_str);

        let res = cross_check_slice11b_report(&temp.path, &report, &stage_comp).unwrap();
        assert_eq!(res.cross_check_status, "mismatch");
        assert!(!res.identity_match);
        assert!(res.discrepancy.unwrap().contains("model_geometry"));
    }

    #[test]
    fn cross_check_same_aggregates_different_vector_evidence_fails() {
        let (report, stage_comp) = sample_test_evidence();

        // Build comp2 with identical max_abs (0.001) but different worst-error index and bits
        let comp2 = crate::numerical_diagnostics::compare_vectors(
            &[1.0, 2.0, 3.0, 4.0],
            &[1.0, 2.0, 3.0, 4.001], // error at index 3 instead of index 0
        )
        .unwrap();

        let json_str = sample_slice11b_report_json(
            &report.case.case_name,
            &report.prompt_token_ids_sha256,
            &report.model_geometry,
            &comp2,
        );
        let temp = TempFile::new("different_vector", &json_str);

        let res = cross_check_slice11b_report(&temp.path, &report, &stage_comp).unwrap();
        assert_eq!(res.cross_check_status, "mismatch");
        assert!(res.identity_match);
        assert!(!res.stage_evidence_match);
        assert!(res
            .discrepancy
            .unwrap()
            .contains("vector_comparison JSON mismatch"));
    }
}

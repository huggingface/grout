#![allow(clippy::too_many_arguments)]

#[repr(u8)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KernelKind {
    EmbeddingBatch,
    Gemm,
    Gemv,
    RmsNorm,
    RopeSeq,
    KvCacheUpdateSeq,
    FlashAttnCausalSeq,
    AddVec,
    SiluMul,
    GatherRow,
    ArgmaxBlocks,
}

impl KernelKind {
    pub const COUNT: usize = 11;

    pub const fn idx(self) -> usize {
        self as usize
    }
}

pub const TILE_KERNEL_KINDS: [KernelKind; 11] = [
    KernelKind::EmbeddingBatch,
    KernelKind::Gemm,
    KernelKind::Gemv,
    KernelKind::RmsNorm,
    KernelKind::RopeSeq,
    KernelKind::KvCacheUpdateSeq,
    KernelKind::FlashAttnCausalSeq,
    KernelKind::AddVec,
    KernelKind::SiluMul,
    KernelKind::GatherRow,
    KernelKind::ArgmaxBlocks,
];

#[allow(clippy::too_many_arguments)]
#[cutile::module]
pub mod kernels {
    use cutile::core::*;

    #[cutile::entry(print_ir=false,
                       optimization_hints = ())]
    fn gemm_f16<const BM: i32, const BN: i32, const BK: i32, const K: i32>(
        z: &mut Tensor<f16, { [BM, BN] }>,
        x: &Tensor<f16, { [-1, K] }>,
        y: &Tensor<f16, { [K, -1] }>,
    ) {
        let part_x = x.partition(const_shape![BM, BK]);
        let part_y = y.partition(const_shape![BK, BN]);
        let pid: (i32, i32, i32) = get_tile_block_id();
        let mut tile_z: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
        for i in 0i32..(K / BK) {
            let tile_x: Tile<f16, { [BM, BK] }> = part_x.load([pid.0, i]);
            let tile_y: Tile<f16, { [BK, BN] }> = part_y.load([i, pid.1]);
            let tile_x: Tile<f32, { [BM, BK] }> = convert_tile(tile_x);
            let tile_y: Tile<f32, { [BK, BN] }> = convert_tile(tile_y);
            tile_z = mma(tile_x, tile_y, tile_z);
            continue;
        }
        let tile_z: Tile<f16, { [BM, BN] }> = convert_tile(tile_z);
        z.store(tile_z);
    }

    #[cutile::entry(print_ir=false,
                       optimization_hints = ())]
    fn add_vec_f16<const S: [i32; 1]>(
        out: &mut Tensor<f16, S>,
        lhs: &Tensor<f16, { [-1] }>,
        rhs: &Tensor<f16, { [-1] }>,
    ) {
        let lhs_tile = load_tile_like(lhs, out);
        let rhs_tile = load_tile_like(rhs, out);
        out.store(lhs_tile + rhs_tile);
    }

    #[cutile::entry(print_ir=false,
                       optimization_hints = ())]
    fn add_2d_f16<const BLOCK_SIZE: i32>(
        out: &mut Tensor<f16, { [1, BLOCK_SIZE] }>,
        lhs: &Tensor<f16, { [-1, -1] }>,
        rhs: &Tensor<f16, { [-1, -1] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let row = pid.0;
        let col = pid.1;
        let lhs_part: Partition<f16, { [1, BLOCK_SIZE] }> =
            lhs.partition(const_shape![1, BLOCK_SIZE]);
        let rhs_part: Partition<f16, { [1, BLOCK_SIZE] }> =
            rhs.partition(const_shape![1, BLOCK_SIZE]);
        let lhs_tile: Tile<f16, { [1, BLOCK_SIZE] }> = lhs_part.load([row, col]);
        let rhs_tile: Tile<f16, { [1, BLOCK_SIZE] }> = rhs_part.load([row, col]);
        out.store(lhs_tile + rhs_tile);
    }

    #[cutile::entry(print_ir=false,
                       optimization_hints = ())]
    fn silu_mul_vec_f16<const S: [i32; 1]>(
        out: &mut Tensor<f16, S>,
        gate: &Tensor<f16, { [-1] }>,
        up: &Tensor<f16, { [-1] }>,
    ) {
        let gate_f16 = load_tile_like(gate, out);
        let up_f16 = load_tile_like(up, out);
        let gate_f32: Tile<f32, S> = convert_tile(gate_f16);
        let up_f32: Tile<f32, S> = convert_tile(up_f16);
        let one: Tile<f32, S> = constant(1.0f32, out.shape());
        let zero: Tile<f32, S> = constant(0.0f32, out.shape());
        let neg_gate: Tile<f32, S> = zero - gate_f32;
        let exp_neg_gate: Tile<f32, S> = exp(neg_gate);
        let denom: Tile<f32, S> = one + exp_neg_gate;
        let sigmoid: Tile<f32, S> = true_div(one, denom);
        let y: Tile<f32, S> = sigmoid * gate_f32 * up_f32;
        let y_f16: Tile<f16, S> = convert_tile(y);
        out.store(y_f16);
    }

    #[cutile::entry(print_ir=false,
                       optimization_hints = ())]
    fn silu_mul_2d_f16<const BLOCK_SIZE: i32>(
        out: &mut Tensor<f16, { [1, BLOCK_SIZE] }>,
        gate: &Tensor<f16, { [-1, -1] }>,
        up: &Tensor<f16, { [-1, -1] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let row = pid.0;
        let col = pid.1;
        let gate_part: Partition<f16, { [1, BLOCK_SIZE] }> =
            gate.partition(const_shape![1, BLOCK_SIZE]);
        let up_part: Partition<f16, { [1, BLOCK_SIZE] }> =
            up.partition(const_shape![1, BLOCK_SIZE]);
        let gate_f16: Tile<f16, { [1, BLOCK_SIZE] }> = gate_part.load([row, col]);
        let up_f16: Tile<f16, { [1, BLOCK_SIZE] }> = up_part.load([row, col]);
        let gate_f32: Tile<f32, { [1, BLOCK_SIZE] }> = convert_tile(gate_f16);
        let up_f32: Tile<f32, { [1, BLOCK_SIZE] }> = convert_tile(up_f16);
        let one: Tile<f32, { [1, BLOCK_SIZE] }> = constant(1.0f32, const_shape![1, BLOCK_SIZE]);
        let zero: Tile<f32, { [1, BLOCK_SIZE] }> = constant(0.0f32, const_shape![1, BLOCK_SIZE]);
        let neg_gate: Tile<f32, { [1, BLOCK_SIZE] }> = zero - gate_f32;
        let exp_neg_gate: Tile<f32, { [1, BLOCK_SIZE] }> = exp(neg_gate);
        let denom: Tile<f32, { [1, BLOCK_SIZE] }> = one + exp_neg_gate;
        let sigmoid: Tile<f32, { [1, BLOCK_SIZE] }> = true_div(one, denom);
        let y: Tile<f32, { [1, BLOCK_SIZE] }> = sigmoid * gate_f32 * up_f32;
        let y_f16: Tile<f16, { [1, BLOCK_SIZE] }> = convert_tile(y);
        out.store(y_f16);
    }

    #[cutile::entry(print_ir=false,
                       optimization_hints = ())]
    fn rms_norm_f16<const N: i32, const BLOCK_SIZE: i32>(
        x: &Tensor<f16, { [-1, N] }>,
        w: &Tensor<f16, { [N] }>,
        out: &mut Tensor<f16, { [1, N] }>,
        eps: f32,
    ) {
        let tile_shape: Shape<{ [1, BLOCK_SIZE] }> = const_shape![1, BLOCK_SIZE];
        let num_tiles: i32 = N / BLOCK_SIZE;
        let pid: (i32, i32, i32) = get_tile_block_id();
        let row = pid.0;

        let x_part: Partition<f16, { [1, BLOCK_SIZE] }> = x.partition(tile_shape);
        let mut rms: Tile<f32, { [1, BLOCK_SIZE] }> = constant(0.0, tile_shape);
        for j in 0i32..num_tiles {
            let tx_f16: Tile<f16, { [1, BLOCK_SIZE] }> = x_part.load([row, j]);
            let tx: Tile<f32, { [1, BLOCK_SIZE] }> = convert_tile(tx_f16);
            rms = rms + tx * tx;
        }
        let rms: Tile<f32, { [1] }> = reduce_sum(rms, 1i32);
        let rms: Tile<f32, { [] }> = rms.reshape(const_shape![]);
        let rms: f32 = tile_to_scalar(rms);
        let n: f32 = convert_scalar(N);
        let inv_rms: f32 = rms / n + eps;
        let inv_rms: Tile<f32, { [] }> = rsqrt(scalar_to_tile(inv_rms), ftz::Disabled);
        let inv_rms: f32 = tile_to_scalar(inv_rms);
        let inv_rms: Tile<f32, { [1, BLOCK_SIZE] }> = inv_rms.broadcast(tile_shape);

        let w_part: Partition<f16, { [BLOCK_SIZE] }> = w.partition(const_shape![BLOCK_SIZE]);
        let mut out_part: PartitionMut<f16, { [1, BLOCK_SIZE] }> =
            unsafe { out.partition_mut(tile_shape) };
        for j in 0i32..num_tiles {
            let tx_f16: Tile<f16, { [1, BLOCK_SIZE] }> = x_part.load([row, j]);
            let tw_f16: Tile<f16, { [1, BLOCK_SIZE] }> = w_part.load([j]).reshape(tile_shape);
            let tx: Tile<f32, { [1, BLOCK_SIZE] }> = convert_tile(tx_f16);
            let tw: Tile<f32, { [1, BLOCK_SIZE] }> = convert_tile(tw_f16);
            let tout: Tile<f32, { [1, BLOCK_SIZE] }> = tx * inv_rms * tw;
            let tout_f16: Tile<f16, { [1, BLOCK_SIZE] }> = convert_tile(tout);
            unsafe { out_part.store(tout_f16, [0i32, j]) };
        }
    }

    #[cutile::entry(print_ir=false,
                       optimization_hints = ())]
    fn argmax_blocks_f16<const BLOCK_SIZE: i32>(
        logits: &Tensor<f16, { [-1] }>,
        block_max: &mut Tensor<f32, { [1] }>,
        block_idx: &mut Tensor<u32, { [1] }>,
        len: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let block = pid.0;

        let logits_part = logits.partition(const_shape![BLOCK_SIZE]);
        let logits_f16: Tile<f16, { [BLOCK_SIZE] }> = logits_part.load([block]);
        let logits: Tile<f32, { [BLOCK_SIZE] }> = convert_tile(logits_f16);

        let base: i32 = block * BLOCK_SIZE;
        let base_tile: Tile<i32, { [BLOCK_SIZE] }> = base.broadcast(const_shape![BLOCK_SIZE]);
        let offs: Tile<i32, { [BLOCK_SIZE] }> = iota(const_shape![BLOCK_SIZE]);
        let indices: Tile<i32, { [BLOCK_SIZE] }> = base_tile + offs;

        let len_tile: Tile<i32, { [BLOCK_SIZE] }> = len.broadcast(const_shape![BLOCK_SIZE]);
        let valid: Tile<bool, { [BLOCK_SIZE] }> = lt_tile(indices, len_tile);

        let mask_mag: Tile<f32, { [BLOCK_SIZE] }> = constant(1.0e30f32, const_shape![BLOCK_SIZE]);
        let zero: Tile<f32, { [BLOCK_SIZE] }> = constant(0.0f32, const_shape![BLOCK_SIZE]);
        let neg_inf: Tile<f32, { [BLOCK_SIZE] }> = zero - mask_mag;
        let masked_logits: Tile<f32, { [BLOCK_SIZE] }> = select(valid, logits, neg_inf);

        let max_tile: Tile<f32, { [1] }> = reduce_max(masked_logits, 0i32);
        let max_scalar: f32 = tile_to_scalar(max_tile.reshape(const_shape![]));

        let max_bcast: Tile<f32, { [BLOCK_SIZE] }> = max_scalar.broadcast(const_shape![BLOCK_SIZE]);
        let is_max: Tile<bool, { [BLOCK_SIZE] }> = eq_tile(masked_logits, max_bcast);
        let invalid_idx: i32 = len + 1i32;
        let invalid_idx: Tile<i32, { [BLOCK_SIZE] }> =
            invalid_idx.broadcast(const_shape![BLOCK_SIZE]);
        let candidate_idx: Tile<i32, { [BLOCK_SIZE] }> = select(is_max, indices, invalid_idx);
        let idx_tile: Tile<i32, { [1] }> = reduce_min(candidate_idx, 0i32);
        let idx_scalar: i32 = tile_to_scalar(idx_tile.reshape(const_shape![]));

        let out_max_scalar: Tile<f32, { [] }> = scalar_to_tile(max_scalar);
        let out_max_tile: Tile<f32, { [1] }> = out_max_scalar.reshape(const_shape![1]);
        let out_idx_scalar: Tile<i32, { [] }> = scalar_to_tile(idx_scalar);
        let out_idx_i32: Tile<i32, { [1] }> = out_idx_scalar.reshape(const_shape![1]);
        let out_idx_tile: Tile<u32, { [1] }> = bitcast(out_idx_i32);
        block_max.store(out_max_tile);
        block_idx.store(out_idx_tile);
    }

    #[cutile::entry(print_ir=false,
                       optimization_hints = ())]
    fn gather_row_f16<const BLOCK_SIZE: i32>(
        src: &Tensor<f16, { [-1, -1] }>,
        out: &mut Tensor<f16, { [BLOCK_SIZE] }>,
        row_idx: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let block = pid.0;
        let src_part: Partition<f16, { [1, BLOCK_SIZE] }> =
            src.partition(const_shape![1, BLOCK_SIZE]);
        let tile: Tile<f16, { [1, BLOCK_SIZE] }> = src_part.load([row_idx, block]);
        out.store(tile.reshape(const_shape![BLOCK_SIZE]));
    }

    #[cutile::entry(print_ir=false,
                       optimization_hints = ())]
    fn rope_f16<const D: i32, const HALF_D: i32>(
        x: &mut Tensor<f16, { [1, D] }>,
        inv_freq: &Tensor<f32, { [HALF_D] }>,
        position: i32,
    ) {
        // Qwen3 text uses chunked/GPT-NeoX RoPE layout: [x0, x1] where each chunk is D/2.
        let x_part: Partition<f16, { [1, HALF_D] }> = x.partition(const_shape![1, HALF_D]);
        let x_lo_f16: Tile<f16, { [1, HALF_D] }> = x_part.load([0i32, 0i32]);
        let x_hi_f16: Tile<f16, { [1, HALF_D] }> = x_part.load([0i32, 1i32]);
        let x_lo: Tile<f32, { [1, HALF_D] }> = convert_tile(x_lo_f16);
        let x_hi: Tile<f32, { [1, HALF_D] }> = convert_tile(x_hi_f16);

        let inv_part = inv_freq.partition(const_shape![HALF_D]);
        let freq: Tile<f32, { [HALF_D] }> = inv_part.load([0i32]);
        let pos: f32 = convert_scalar(position);
        let pos: Tile<f32, { [HALF_D] }> = pos.broadcast(const_shape![HALF_D]);
        let theta: Tile<f32, { [HALF_D] }> = pos * freq;
        let theta: Tile<f32, { [1, HALF_D] }> = theta.reshape(const_shape![1, HALF_D]);
        let cos_t = cos(theta);
        let sin_t = sin(theta);

        let y_lo: Tile<f32, { [1, HALF_D] }> = x_lo * cos_t - x_hi * sin_t;
        let y_hi: Tile<f32, { [1, HALF_D] }> = x_hi * cos_t + x_lo * sin_t;
        let y_lo_f16: Tile<f16, { [1, HALF_D] }> = convert_tile(y_lo);
        let y_hi_f16: Tile<f16, { [1, HALF_D] }> = convert_tile(y_hi);

        let mut x_out = unsafe { x.partition_mut(const_shape![1, HALF_D]) };
        unsafe {
            x_out.store(y_lo_f16, [0i32, 0i32]);
            x_out.store(y_hi_f16, [0i32, 1i32]);
        }
    }

    #[cutile::entry(print_ir=false,
                       optimization_hints = ())]
    fn rope_seq_f16<const D: i32, const HALF_D: i32>(
        x: &Tensor<f16, { [-1, -1, D] }>,
        inv_freq: &Tensor<f32, { [HALF_D] }>,
        out: &mut Tensor<f16, { [1, 1, HALF_D] }>,
        position_start: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let seq_idx = pid.0;
        let head_idx = pid.1;
        let half_idx = pid.2;

        // Qwen3 text uses chunked/GPT-NeoX RoPE layout: [x0, x1] where each chunk is D/2.
        let x_part: Partition<f16, { [1, 1, HALF_D] }> = x.partition(const_shape![1, 1, HALF_D]);
        let x_lo_f16: Tile<f16, { [1, 1, HALF_D] }> = x_part.load([seq_idx, head_idx, 0i32]);
        let x_hi_f16: Tile<f16, { [1, 1, HALF_D] }> = x_part.load([seq_idx, head_idx, 1i32]);
        let x_lo: Tile<f32, { [1, 1, HALF_D] }> = convert_tile(x_lo_f16);
        let x_hi: Tile<f32, { [1, 1, HALF_D] }> = convert_tile(x_hi_f16);

        let inv_part = inv_freq.partition(const_shape![HALF_D]);
        let freq: Tile<f32, { [HALF_D] }> = inv_part.load([0i32]);
        let pos_i: i32 = position_start + seq_idx;
        let pos: f32 = convert_scalar(pos_i);
        let pos: Tile<f32, { [HALF_D] }> = pos.broadcast(const_shape![HALF_D]);
        let theta: Tile<f32, { [HALF_D] }> = pos * freq;
        let theta: Tile<f32, { [1, 1, HALF_D] }> = theta.reshape(const_shape![1, 1, HALF_D]);
        let cos_t = cos(theta);
        let sin_t = sin(theta);

        let y_lo: Tile<f32, { [1, 1, HALF_D] }> = x_lo * cos_t - x_hi * sin_t;
        let y_hi: Tile<f32, { [1, 1, HALF_D] }> = x_hi * cos_t + x_lo * sin_t;
        let y_lo_f16: Tile<f16, { [1, 1, HALF_D] }> = convert_tile(y_lo);
        let y_hi_f16: Tile<f16, { [1, 1, HALF_D] }> = convert_tile(y_hi);

        if half_idx == 0i32 {
            out.store(y_lo_f16);
        } else {
            out.store(y_hi_f16);
        }
    }

    #[cutile::entry(print_ir=false,
                       optimization_hints = ())]
    fn rope_seq_dynpos_f16<const D: i32, const HALF_D: i32>(
        x: &Tensor<f16, { [-1, -1, D] }>,
        inv_freq: &Tensor<f32, { [HALF_D] }>,
        position_start: &Tensor<u32, { [1] }>,
        out: &mut Tensor<f16, { [1, 1, HALF_D] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let seq_idx = pid.0;
        let head_idx = pid.1;
        let half_idx = pid.2;

        // Qwen3 text uses chunked/GPT-NeoX RoPE layout: [x0, x1] where each chunk is D/2.
        let x_part: Partition<f16, { [1, 1, HALF_D] }> = x.partition(const_shape![1, 1, HALF_D]);
        let x_lo_f16: Tile<f16, { [1, 1, HALF_D] }> = x_part.load([seq_idx, head_idx, 0i32]);
        let x_hi_f16: Tile<f16, { [1, 1, HALF_D] }> = x_part.load([seq_idx, head_idx, 1i32]);
        let x_lo: Tile<f32, { [1, 1, HALF_D] }> = convert_tile(x_lo_f16);
        let x_hi: Tile<f32, { [1, 1, HALF_D] }> = convert_tile(x_hi_f16);

        let pos_part = position_start.partition(const_shape![1]);
        let base_pos_t_u32: Tile<u32, { [1] }> = pos_part.load([0i32]);
        let base_pos_t: Tile<i32, { [1] }> = bitcast(base_pos_t_u32);
        let base_pos: i32 = tile_to_scalar(base_pos_t.reshape(const_shape![]));

        let inv_part = inv_freq.partition(const_shape![HALF_D]);
        let freq: Tile<f32, { [HALF_D] }> = inv_part.load([0i32]);
        let pos_i: i32 = base_pos + seq_idx;
        let pos: f32 = convert_scalar(pos_i);
        let pos: Tile<f32, { [HALF_D] }> = pos.broadcast(const_shape![HALF_D]);
        let theta: Tile<f32, { [HALF_D] }> = pos * freq;
        let theta: Tile<f32, { [1, 1, HALF_D] }> = theta.reshape(const_shape![1, 1, HALF_D]);
        let cos_t = cos(theta);
        let sin_t = sin(theta);

        let y_lo: Tile<f32, { [1, 1, HALF_D] }> = x_lo * cos_t - x_hi * sin_t;
        let y_hi: Tile<f32, { [1, 1, HALF_D] }> = x_hi * cos_t + x_lo * sin_t;
        let y_lo_f16: Tile<f16, { [1, 1, HALF_D] }> = convert_tile(y_lo);
        let y_hi_f16: Tile<f16, { [1, 1, HALF_D] }> = convert_tile(y_hi);

        if half_idx == 0i32 {
            out.store(y_lo_f16);
        } else {
            out.store(y_hi_f16);
        }
    }

    #[cutile::entry(print_ir=false,
                       optimization_hints = ())]
    fn embedding_f16<const D: i32, const BLOCK_SIZE: i32>(
        token_ids: &Tensor<u32, { [-1] }>,
        table: &Tensor<f16, { [-1, D] }>,
        out: &mut Tensor<f16, { [1, D] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let row = pid.0;
        let ids_part = token_ids.partition(const_shape![1]);
        let token_tile: Tile<u32, { [1] }> = ids_part.load([row]);
        let token_idx_tile: Tile<i32, { [1] }> = bitcast(token_tile);
        let token_idx: i32 = tile_to_scalar(token_idx_tile.reshape(const_shape![]));

        let emb_part: Partition<f16, { [1, BLOCK_SIZE] }> =
            table.partition(const_shape![1, BLOCK_SIZE]);
        let mut out_part: PartitionMut<f16, { [1, BLOCK_SIZE] }> =
            unsafe { out.partition_mut(const_shape![1, BLOCK_SIZE]) };
        for j in 0i32..(D / BLOCK_SIZE) {
            let emb: Tile<f16, { [1, BLOCK_SIZE] }> = emb_part.load([token_idx, j]);
            unsafe { out_part.store(emb, [0i32, j]) };
        }
    }

    #[cutile::entry(print_ir=false,
                       optimization_hints = ())]
    fn embedding_batch_f16<const D: i32, const BLOCK_SIZE: i32>(
        token_ids: &Tensor<u32, { [-1] }>,
        table: &Tensor<f16, { [-1, D] }>,
        out: &mut Tensor<f16, { [1, BLOCK_SIZE] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let row = pid.0;
        let d_block = pid.1;

        let ids_part = token_ids.partition(const_shape![1]);
        let token_tile: Tile<u32, { [1] }> = ids_part.load([row]);
        let token_idx_tile: Tile<i32, { [1] }> = bitcast(token_tile);
        let token_idx: i32 = tile_to_scalar(token_idx_tile.reshape(const_shape![]));

        let emb_part: Partition<f16, { [1, BLOCK_SIZE] }> =
            table.partition(const_shape![1, BLOCK_SIZE]);
        let emb: Tile<f16, { [1, BLOCK_SIZE] }> = emb_part.load([token_idx, d_block]);
        out.store(emb);
    }

    #[cutile::entry(print_ir=false,
                       optimization_hints = ())]
    fn kv_cache_update_f16<const D: i32, const BLOCK_SIZE: i32, const MAX_SEQ: i32>(
        new_k: &Tensor<f16, { [-1, D] }>,
        new_v: &Tensor<f16, { [-1, D] }>,
        k_cache: &mut Tensor<f16, { [1, MAX_SEQ, BLOCK_SIZE] }>,
        v_cache: &mut Tensor<f16, { [1, MAX_SEQ, BLOCK_SIZE] }>,
        position: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let head = pid.0;
        let d_block = pid.2;

        let new_k_part = new_k.partition(const_shape![1, BLOCK_SIZE]);
        let new_v_part = new_v.partition(const_shape![1, BLOCK_SIZE]);
        let mut k_cache_part = unsafe { k_cache.partition_mut(const_shape![1, 1, BLOCK_SIZE]) };
        let mut v_cache_part = unsafe { v_cache.partition_mut(const_shape![1, 1, BLOCK_SIZE]) };

        let k_tile = new_k_part
            .load([head, d_block])
            .reshape(const_shape![1, 1, BLOCK_SIZE]);
        let v_tile = new_v_part
            .load([head, d_block])
            .reshape(const_shape![1, 1, BLOCK_SIZE]);
        unsafe {
            k_cache_part.store(k_tile, [0i32, position, 0i32]);
            v_cache_part.store(v_tile, [0i32, position, 0i32]);
        }
    }

    #[cutile::entry(print_ir=false,
                       optimization_hints = ())]
    fn kv_cache_update_seq_f16<const D: i32, const BLOCK_SIZE: i32, const MAX_SEQ: i32>(
        new_k: &Tensor<f16, { [-1, -1, D] }>,
        new_v: &Tensor<f16, { [-1, -1, D] }>,
        k_cache: &mut Tensor<f16, { [1, MAX_SEQ, BLOCK_SIZE] }>,
        v_cache: &mut Tensor<f16, { [1, MAX_SEQ, BLOCK_SIZE] }>,
        position_start: i32,
        seq_len: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let head = pid.0;
        let d_block = pid.2;

        let new_k_part = new_k.partition(const_shape![1, 1, BLOCK_SIZE]);
        let new_v_part = new_v.partition(const_shape![1, 1, BLOCK_SIZE]);
        let mut k_cache_part = unsafe { k_cache.partition_mut(const_shape![1, 1, BLOCK_SIZE]) };
        let mut v_cache_part = unsafe { v_cache.partition_mut(const_shape![1, 1, BLOCK_SIZE]) };

        for s in 0i32..seq_len {
            let k_tile = new_k_part
                .load([s, head, d_block])
                .reshape(const_shape![1, 1, BLOCK_SIZE]);
            let v_tile = new_v_part
                .load([s, head, d_block])
                .reshape(const_shape![1, 1, BLOCK_SIZE]);
            let cache_pos = position_start + s;
            unsafe {
                k_cache_part.store(k_tile, [0i32, cache_pos, 0i32]);
                v_cache_part.store(v_tile, [0i32, cache_pos, 0i32]);
            }
        }
    }

    #[cutile::entry(print_ir=false,
                       optimization_hints = ())]
    fn kv_cache_update_seq_dynpos_f16<const D: i32, const BLOCK_SIZE: i32, const MAX_SEQ: i32>(
        new_k: &Tensor<f16, { [-1, -1, D] }>,
        new_v: &Tensor<f16, { [-1, -1, D] }>,
        k_cache: &mut Tensor<f16, { [1, MAX_SEQ, BLOCK_SIZE] }>,
        v_cache: &mut Tensor<f16, { [1, MAX_SEQ, BLOCK_SIZE] }>,
        position_start: &Tensor<u32, { [1] }>,
        seq_len: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let head = pid.0;
        let d_block = pid.2;

        let pos_part = position_start.partition(const_shape![1]);
        let pos_t_u32: Tile<u32, { [1] }> = pos_part.load([0i32]);
        let pos_t: Tile<i32, { [1] }> = bitcast(pos_t_u32);
        let pos_start: i32 = tile_to_scalar(pos_t.reshape(const_shape![]));

        let new_k_part = new_k.partition(const_shape![1, 1, BLOCK_SIZE]);
        let new_v_part = new_v.partition(const_shape![1, 1, BLOCK_SIZE]);
        let mut k_cache_part = unsafe { k_cache.partition_mut(const_shape![1, 1, BLOCK_SIZE]) };
        let mut v_cache_part = unsafe { v_cache.partition_mut(const_shape![1, 1, BLOCK_SIZE]) };

        for s in 0i32..seq_len {
            let k_tile = new_k_part
                .load([s, head, d_block])
                .reshape(const_shape![1, 1, BLOCK_SIZE]);
            let v_tile = new_v_part
                .load([s, head, d_block])
                .reshape(const_shape![1, 1, BLOCK_SIZE]);
            let cache_pos = pos_start + s;
            unsafe {
                k_cache_part.store(k_tile, [0i32, cache_pos, 0i32]);
                v_cache_part.store(v_tile, [0i32, cache_pos, 0i32]);
            }
        }
    }

    #[cutile::entry(print_ir=false,
                       unchecked_accesses=true,
                       optimization_hints = ())]
    unsafe fn flash_attn_f16<const BM: i32, const BN: i32, const D: i32>(
        q: &Tensor<f16, { [-1, -1, D] }>,
        k: &Tensor<f16, { [-1, -1, D] }>,
        v: &Tensor<f16, { [-1, -1, D] }>,
        out: &mut Tensor<f16, { [1, BM, D] }>,
        qk_scale: f32,
        query_group_size: i32,
        kv_len: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let q_head_idx = pid.0;
        let q_m_idx = pid.1;
        let kv_head_idx = q_head_idx / query_group_size;
        let qk_scale: Tile<f32, { [BM, BN] }> = qk_scale.broadcast(const_shape![BM, BN]);

        let mask_mag: Tile<f32, { [BM, BN] }> = constant(1.0e30f32, const_shape![BM, BN]);
        let mask_false: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]) - mask_mag;
        let offs_n_tile: Tile<i32, { [BN] }> = iota(const_shape![BN]);
        let offs_n_tile: Tile<i32, { [BM, BN] }> = offs_n_tile
            .reshape(const_shape![1, BN])
            .broadcast(const_shape![BM, BN]);

        let max_mag: Tile<f32, { [BM, 1] }> = constant(1.0e30f32, const_shape![BM, 1]);
        let mut m_i: Tile<f32, { [BM, 1] }> = constant(0.0f32, const_shape![BM, 1]) - max_mag;
        let mut l_i: Tile<f32, { [BM, 1] }> = constant(0.0f32, const_shape![BM, 1]);
        let mut acc: Tile<f32, { [BM, D] }> = constant(0.0f32, const_shape![BM, D]);

        let q_part: Partition<f16, { [1, BM, D] }> = q.partition(const_shape![1, BM, D]);
        let tq: Tile<f16, { [1, BM, D] }> = q_part.load([q_head_idx, q_m_idx, 0i32]);
        let tq: Tile<f32, { [BM, D] }> = convert_tile(tq.reshape(const_shape![BM, D]));

        let n: i32 = kv_len;
        let num_tiles: i32 = (n + BN - 1i32) / BN;
        let k_part = k.partition(const_shape![1, BN, D]);
        let v_part = v.partition(const_shape![1, BN, D]);
        let transpose: Array<{ [1, 0] }> = Array::<{ [1, 0] }> {
            dims: &[1i32, 0i32],
        };

        for j in 0i32..num_tiles {
            let k_tile: Tile<f16, { [1, BN, D] }> = k_part.load([kv_head_idx, j, 0i32]);
            let k_tile: Tile<f16, { [BN, D] }> = k_tile.reshape(const_shape![BN, D]);
            let k_tile_trans: Tile<f16, { [D, BN] }> = permute(k_tile, transpose);
            let k_tile_trans: Tile<f32, { [D, BN] }> = convert_tile(k_tile_trans);
            let qk: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
            let qk: Tile<f32, { [BM, BN] }> = mma(tq, k_tile_trans, qk);
            let qk: Tile<f32, { [BM, BN] }> = qk * qk_scale;

            let offs_n: i32 = j * BN;
            let offs_n: Tile<i32, { [BM, BN] }> = offs_n.broadcast(const_shape![BM, BN]);
            let offs_n: Tile<i32, { [BM, BN] }> = offs_n + offs_n_tile;
            let kv_len_t: Tile<i32, { [BM, BN] }> = n.broadcast(const_shape![BM, BN]);
            let valid: Tile<bool, { [BM, BN] }> = lt_tile(offs_n, kv_len_t);
            let qk: Tile<f32, { [BM, BN] }> = select(valid, qk, mask_false);

            let qk_max: Tile<f32, { [BM] }> = reduce_max(qk, 1i32);
            let qk_max: Tile<f32, { [BM, 1] }> = qk_max.reshape(const_shape![BM, 1]);
            let m_ij: Tile<f32, { [BM, 1] }> = max_tile(m_i, qk_max);
            let qk: Tile<f32, { [BM, BN] }> = qk - m_ij.broadcast(const_shape![BM, BN]);

            let p: Tile<f32, { [BM, BN] }> = exp(qk);
            let l_ij: Tile<f32, { [BM] }> = reduce_sum(p, 1i32);
            let l_ij: Tile<f32, { [BM, 1] }> = l_ij.reshape(const_shape![BM, 1]);
            let alpha: Tile<f32, { [BM, 1] }> = exp(m_i - m_ij);
            l_i = fma(l_i, alpha, l_ij, rounding::NearestEven, ftz::Disabled);
            let alpha: Tile<f32, { [BM, D] }> = alpha.broadcast(const_shape![BM, D]);
            acc = acc * alpha;

            let v_tile: Tile<f16, { [1, BN, D] }> = v_part.load([kv_head_idx, j, 0i32]);
            let v_tile: Tile<f32, { [BN, D] }> = convert_tile(v_tile.reshape(const_shape![BN, D]));
            acc = mma(p, v_tile, acc);
            m_i = m_ij;
        }

        let eps: Tile<f32, { [BM, 1] }> = constant(1.0e-8f32, const_shape![BM, 1]);
        let l_i: Tile<f32, { [BM, 1] }> = max_tile(l_i, eps);
        acc = true_div(acc, l_i.broadcast(const_shape![BM, D]));
        let acc: Tile<f16, { [1, BM, D] }> = convert_tile(acc.reshape(const_shape![1, BM, D]));
        out.store(acc);
    }

    #[cutile::entry(print_ir=false,
                       unchecked_accesses=true,
                       optimization_hints = ())]
    unsafe fn flash_attn_causal_f16<const BM: i32, const BN: i32, const D: i32>(
        q: &Tensor<f16, { [-1, -1, D] }>,
        k: &Tensor<f16, { [-1, -1, D] }>,
        v: &Tensor<f16, { [-1, -1, D] }>,
        out: &mut Tensor<f16, { [1, BM, D] }>,
        qk_scale: f32,
        query_group_size: i32,
        kv_len: i32,
        query_start: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let q_head_idx = pid.0;
        let q_m_idx = pid.1;
        let kv_head_idx = q_head_idx / query_group_size;
        let qk_scale: Tile<f32, { [BM, BN] }> = qk_scale.broadcast(const_shape![BM, BN]);

        let mask_mag: Tile<f32, { [BM, BN] }> = constant(1.0e30f32, const_shape![BM, BN]);
        let mask_false: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]) - mask_mag;
        let offs_n_tile: Tile<i32, { [BN] }> = iota(const_shape![BN]);
        let offs_n_tile: Tile<i32, { [BM, BN] }> = offs_n_tile
            .reshape(const_shape![1, BN])
            .broadcast(const_shape![BM, BN]);
        let offs_m_base: i32 = query_start + q_m_idx * BM;
        let offs_m: Tile<i32, { [BM] }> = offs_m_base.broadcast(const_shape![BM]);
        let m_arange: Tile<i32, { [BM] }> = iota(const_shape![BM]);
        let offs_m: Tile<i32, { [BM] }> = offs_m + m_arange;
        let offs_m: Tile<i32, { [BM, BN] }> = offs_m
            .reshape(const_shape![BM, 1])
            .broadcast(const_shape![BM, BN]);

        let max_mag: Tile<f32, { [BM, 1] }> = constant(1.0e30f32, const_shape![BM, 1]);
        let mut m_i: Tile<f32, { [BM, 1] }> = constant(0.0f32, const_shape![BM, 1]) - max_mag;
        let mut l_i: Tile<f32, { [BM, 1] }> = constant(0.0f32, const_shape![BM, 1]);
        let mut acc: Tile<f32, { [BM, D] }> = constant(0.0f32, const_shape![BM, D]);

        let q_part: Partition<f16, { [1, BM, D] }> = q.partition(const_shape![1, BM, D]);
        let tq: Tile<f16, { [1, BM, D] }> = q_part.load([q_head_idx, q_m_idx, 0i32]);
        let tq: Tile<f32, { [BM, D] }> = convert_tile(tq.reshape(const_shape![BM, D]));

        let n: i32 = kv_len;
        let num_tiles: i32 = (n + BN - 1i32) / BN;
        let k_part = k.partition(const_shape![1, BN, D]);
        let v_part = v.partition(const_shape![1, BN, D]);
        let transpose: Array<{ [1, 0] }> = Array::<{ [1, 0] }> {
            dims: &[1i32, 0i32],
        };

        for j in 0i32..num_tiles {
            let k_tile: Tile<f16, { [1, BN, D] }> = k_part.load([kv_head_idx, j, 0i32]);
            let k_tile: Tile<f16, { [BN, D] }> = k_tile.reshape(const_shape![BN, D]);
            let k_tile_trans: Tile<f16, { [D, BN] }> = permute(k_tile, transpose);
            let k_tile_trans: Tile<f32, { [D, BN] }> = convert_tile(k_tile_trans);
            let qk: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
            let qk: Tile<f32, { [BM, BN] }> = mma(tq, k_tile_trans, qk);
            let qk: Tile<f32, { [BM, BN] }> = qk * qk_scale;

            let offs_n: i32 = j * BN;
            let offs_n: Tile<i32, { [BM, BN] }> = offs_n.broadcast(const_shape![BM, BN]);
            let offs_n: Tile<i32, { [BM, BN] }> = offs_n + offs_n_tile;
            let kv_len_t: Tile<i32, { [BM, BN] }> = n.broadcast(const_shape![BM, BN]);
            let valid_k: Tile<bool, { [BM, BN] }> = lt_tile(offs_n, kv_len_t);
            let valid_causal: Tile<bool, { [BM, BN] }> = ge_tile(offs_m, offs_n);
            let valid: Tile<bool, { [BM, BN] }> = valid_k & valid_causal;
            let qk: Tile<f32, { [BM, BN] }> = select(valid, qk, mask_false);

            let qk_max: Tile<f32, { [BM] }> = reduce_max(qk, 1i32);
            let qk_max: Tile<f32, { [BM, 1] }> = qk_max.reshape(const_shape![BM, 1]);
            let m_ij: Tile<f32, { [BM, 1] }> = max_tile(m_i, qk_max);
            let qk: Tile<f32, { [BM, BN] }> = qk - m_ij.broadcast(const_shape![BM, BN]);

            let p: Tile<f32, { [BM, BN] }> = exp(qk);
            let l_ij: Tile<f32, { [BM] }> = reduce_sum(p, 1i32);
            let l_ij: Tile<f32, { [BM, 1] }> = l_ij.reshape(const_shape![BM, 1]);
            let alpha: Tile<f32, { [BM, 1] }> = exp(m_i - m_ij);
            l_i = fma(l_i, alpha, l_ij, rounding::NearestEven, ftz::Disabled);
            let alpha: Tile<f32, { [BM, D] }> = alpha.broadcast(const_shape![BM, D]);
            acc = acc * alpha;

            let v_tile: Tile<f16, { [1, BN, D] }> = v_part.load([kv_head_idx, j, 0i32]);
            let v_tile: Tile<f32, { [BN, D] }> = convert_tile(v_tile.reshape(const_shape![BN, D]));
            acc = mma(p, v_tile, acc);
            m_i = m_ij;
        }

        let eps: Tile<f32, { [BM, 1] }> = constant(1.0e-8f32, const_shape![BM, 1]);
        let l_i: Tile<f32, { [BM, 1] }> = max_tile(l_i, eps);
        acc = true_div(acc, l_i.broadcast(const_shape![BM, D]));
        let acc: Tile<f16, { [1, BM, D] }> = convert_tile(acc.reshape(const_shape![1, BM, D]));
        out.store(acc);
    }

    #[cutile::entry(print_ir=false,
                       unchecked_accesses=true,
                       optimization_hints = ())]
    unsafe fn flash_attn_causal_seq_f16<const BM: i32, const BN: i32, const D: i32>(
        q: &Tensor<f16, { [-1, -1, D] }>,      // [q_len, q_heads, d]
        k: &Tensor<f16, { [-1, -1, D] }>,      // [kv_heads, kv_len, d]
        v: &Tensor<f16, { [-1, -1, D] }>,      // [kv_heads, kv_len, d]
        out: &mut Tensor<f16, { [BM, 1, D] }>, // [q_len, q_heads, d]
        qk_scale: f32,
        query_group_size: i32,
        kv_len: i32,
        query_start: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let q_m_idx = pid.0;
        let q_head_idx = pid.1;
        let kv_head_idx = q_head_idx / query_group_size;
        let qk_scale: Tile<f32, { [BM, BN] }> = qk_scale.broadcast(const_shape![BM, BN]);

        let mask_mag: Tile<f32, { [BM, BN] }> = constant(1.0e30f32, const_shape![BM, BN]);
        let mask_false: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]) - mask_mag;
        let offs_n_tile: Tile<i32, { [BN] }> = iota(const_shape![BN]);
        let offs_n_tile: Tile<i32, { [BM, BN] }> = offs_n_tile
            .reshape(const_shape![1, BN])
            .broadcast(const_shape![BM, BN]);
        let offs_m_base: i32 = query_start + q_m_idx * BM;
        let offs_m: Tile<i32, { [BM] }> = offs_m_base.broadcast(const_shape![BM]);
        let m_arange: Tile<i32, { [BM] }> = iota(const_shape![BM]);
        let offs_m: Tile<i32, { [BM] }> = offs_m + m_arange;
        let offs_m: Tile<i32, { [BM, BN] }> = offs_m
            .reshape(const_shape![BM, 1])
            .broadcast(const_shape![BM, BN]);

        let max_mag: Tile<f32, { [BM, 1] }> = constant(1.0e30f32, const_shape![BM, 1]);
        let mut m_i: Tile<f32, { [BM, 1] }> = constant(0.0f32, const_shape![BM, 1]) - max_mag;
        let mut l_i: Tile<f32, { [BM, 1] }> = constant(0.0f32, const_shape![BM, 1]);
        let mut acc: Tile<f32, { [BM, D] }> = constant(0.0f32, const_shape![BM, D]);

        let q_part: Partition<f16, { [BM, 1, D] }> = q.partition(const_shape![BM, 1, D]);
        let tq: Tile<f16, { [BM, 1, D] }> = q_part.load([q_m_idx, q_head_idx, 0i32]);
        let tq: Tile<f32, { [BM, D] }> = convert_tile(tq.reshape(const_shape![BM, D]));

        let n: i32 = kv_len;
        let num_tiles: i32 = (n + BN - 1i32) / BN;
        let k_part = k.partition(const_shape![1, BN, D]);
        let v_part = v.partition(const_shape![1, BN, D]);
        let transpose: Array<{ [1, 0] }> = Array::<{ [1, 0] }> {
            dims: &[1i32, 0i32],
        };

        for j in 0i32..num_tiles {
            let k_tile: Tile<f16, { [1, BN, D] }> = k_part.load([kv_head_idx, j, 0i32]);
            let k_tile: Tile<f16, { [BN, D] }> = k_tile.reshape(const_shape![BN, D]);
            let k_tile_trans: Tile<f16, { [D, BN] }> = permute(k_tile, transpose);
            let k_tile_trans: Tile<f32, { [D, BN] }> = convert_tile(k_tile_trans);
            let qk: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
            let qk: Tile<f32, { [BM, BN] }> = mma(tq, k_tile_trans, qk);
            let qk: Tile<f32, { [BM, BN] }> = qk * qk_scale;

            let offs_n: i32 = j * BN;
            let offs_n: Tile<i32, { [BM, BN] }> = offs_n.broadcast(const_shape![BM, BN]);
            let offs_n: Tile<i32, { [BM, BN] }> = offs_n + offs_n_tile;
            let kv_len_t: Tile<i32, { [BM, BN] }> = n.broadcast(const_shape![BM, BN]);
            let valid_k: Tile<bool, { [BM, BN] }> = lt_tile(offs_n, kv_len_t);
            let valid_causal: Tile<bool, { [BM, BN] }> = ge_tile(offs_m, offs_n);
            let valid: Tile<bool, { [BM, BN] }> = valid_k & valid_causal;
            let qk: Tile<f32, { [BM, BN] }> = select(valid, qk, mask_false);

            let qk_max: Tile<f32, { [BM] }> = reduce_max(qk, 1i32);
            let qk_max: Tile<f32, { [BM, 1] }> = qk_max.reshape(const_shape![BM, 1]);
            let m_ij: Tile<f32, { [BM, 1] }> = max_tile(m_i, qk_max);
            let qk: Tile<f32, { [BM, BN] }> = qk - m_ij.broadcast(const_shape![BM, BN]);

            let p: Tile<f32, { [BM, BN] }> = exp(qk);
            let l_ij: Tile<f32, { [BM] }> = reduce_sum(p, 1i32);
            let l_ij: Tile<f32, { [BM, 1] }> = l_ij.reshape(const_shape![BM, 1]);
            let alpha: Tile<f32, { [BM, 1] }> = exp(m_i - m_ij);
            l_i = fma(l_i, alpha, l_ij, rounding::NearestEven, ftz::Disabled);
            let alpha: Tile<f32, { [BM, D] }> = alpha.broadcast(const_shape![BM, D]);
            acc = acc * alpha;

            let v_tile: Tile<f16, { [1, BN, D] }> = v_part.load([kv_head_idx, j, 0i32]);
            let v_tile: Tile<f32, { [BN, D] }> = convert_tile(v_tile.reshape(const_shape![BN, D]));
            acc = mma(p, v_tile, acc);
            m_i = m_ij;
        }

        let eps: Tile<f32, { [BM, 1] }> = constant(1.0e-8f32, const_shape![BM, 1]);
        let l_i: Tile<f32, { [BM, 1] }> = max_tile(l_i, eps);
        acc = true_div(acc, l_i.broadcast(const_shape![BM, D]));
        let acc: Tile<f16, { [BM, 1, D] }> = convert_tile(acc.reshape(const_shape![BM, 1, D]));
        out.store(acc);
    }

    #[cutile::entry(print_ir=false,
                    unchecked_accesses=true,
                    optimization_hints = (sm_120 = (occupancy=4,),))]
    unsafe fn flash_attn_causal_seq_dynpos_f16<const BM: i32, const BN: i32, const D: i32>(
        q: &Tensor<f16, { [-1, -1, D] }>,      // [q_len, q_heads, d]
        k: &Tensor<f16, { [-1, -1, D] }>,      // [kv_heads, kv_len, d]
        v: &Tensor<f16, { [-1, -1, D] }>,      // [kv_heads, kv_len, d]
        out: &mut Tensor<f16, { [BM, 1, D] }>, // [q_len, q_heads, d]
        qk_scale: f32,
        query_group_size: i32,
        position_start: &Tensor<u32, { [1] }>,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let q_m_idx = pid.0;
        let q_head_idx = pid.1;
        let kv_head_idx = q_head_idx / query_group_size;
        let qk_scale: Tile<f32, { [BM, BN] }> = qk_scale.broadcast(const_shape![BM, BN]);

        let pos_part = position_start.partition(const_shape![1]);
        let pos_t_u32: Tile<u32, { [1] }> = pos_part.load([0i32]);
        let pos_t: Tile<i32, { [1] }> = bitcast(pos_t_u32);
        let query_start: i32 = tile_to_scalar(pos_t.reshape(const_shape![]));

        // Decode graph uses q_len=1, BM=1: kv_len is position+1.
        let kv_len: i32 = query_start + 1i32;

        let mask_mag: Tile<f32, { [BM, BN] }> = constant(1.0e30f32, const_shape![BM, BN]);
        let mask_false: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]) - mask_mag;
        let offs_n_tile: Tile<i32, { [BN] }> = iota(const_shape![BN]);
        let offs_n_tile: Tile<i32, { [BM, BN] }> = offs_n_tile
            .reshape(const_shape![1, BN])
            .broadcast(const_shape![BM, BN]);
        let offs_m_base: i32 = query_start + q_m_idx * BM;
        let offs_m: Tile<i32, { [BM] }> = offs_m_base.broadcast(const_shape![BM]);
        let m_arange: Tile<i32, { [BM] }> = iota(const_shape![BM]);
        let offs_m: Tile<i32, { [BM] }> = offs_m + m_arange;
        let offs_m: Tile<i32, { [BM, BN] }> = offs_m
            .reshape(const_shape![BM, 1])
            .broadcast(const_shape![BM, BN]);

        let max_mag: Tile<f32, { [BM, 1] }> = constant(1.0e30f32, const_shape![BM, 1]);
        let mut m_i: Tile<f32, { [BM, 1] }> = constant(0.0f32, const_shape![BM, 1]) - max_mag;
        let mut l_i: Tile<f32, { [BM, 1] }> = constant(0.0f32, const_shape![BM, 1]);
        let mut acc: Tile<f32, { [BM, D] }> = constant(0.0f32, const_shape![BM, D]);

        let q_part: Partition<f16, { [BM, 1, D] }> = q.partition(const_shape![BM, 1, D]);
        let tq: Tile<f16, { [BM, 1, D] }> = q_part.load([q_m_idx, q_head_idx, 0i32]);
        let tq: Tile<f32, { [BM, D] }> = convert_tile(tq.reshape(const_shape![BM, D]));

        let n: i32 = kv_len;
        let num_tiles: i32 = (n + BN - 1i32) / BN;
        let k_part = k.partition(const_shape![1, BN, D]);
        let v_part = v.partition(const_shape![1, BN, D]);
        let transpose: Array<{ [1, 0] }> = Array::<{ [1, 0] }> {
            dims: &[1i32, 0i32],
        };

        for j in 0i32..num_tiles {
            let k_tile: Tile<f16, { [1, BN, D] }> = k_part.load([kv_head_idx, j, 0i32]);
            let k_tile: Tile<f16, { [BN, D] }> = k_tile.reshape(const_shape![BN, D]);
            let k_tile_trans: Tile<f16, { [D, BN] }> = permute(k_tile, transpose);
            let k_tile_trans: Tile<f32, { [D, BN] }> = convert_tile(k_tile_trans);
            let qk: Tile<f32, { [BM, BN] }> = constant(0.0f32, const_shape![BM, BN]);
            let qk: Tile<f32, { [BM, BN] }> = mma(tq, k_tile_trans, qk);
            let qk: Tile<f32, { [BM, BN] }> = qk * qk_scale;

            let offs_n: i32 = j * BN;
            let offs_n: Tile<i32, { [BM, BN] }> = offs_n.broadcast(const_shape![BM, BN]);
            let offs_n: Tile<i32, { [BM, BN] }> = offs_n + offs_n_tile;
            let kv_len_t: Tile<i32, { [BM, BN] }> = n.broadcast(const_shape![BM, BN]);
            let valid_k: Tile<bool, { [BM, BN] }> = lt_tile(offs_n, kv_len_t);
            let valid_causal: Tile<bool, { [BM, BN] }> = ge_tile(offs_m, offs_n);
            let valid: Tile<bool, { [BM, BN] }> = valid_k & valid_causal;
            let qk: Tile<f32, { [BM, BN] }> = select(valid, qk, mask_false);

            let qk_max: Tile<f32, { [BM] }> = reduce_max(qk, 1i32);
            let qk_max: Tile<f32, { [BM, 1] }> = qk_max.reshape(const_shape![BM, 1]);
            let m_ij: Tile<f32, { [BM, 1] }> = max_tile(m_i, qk_max);
            let qk: Tile<f32, { [BM, BN] }> = qk - m_ij.broadcast(const_shape![BM, BN]);

            let p: Tile<f32, { [BM, BN] }> = exp(qk);
            let l_ij: Tile<f32, { [BM] }> = reduce_sum(p, 1i32);
            let l_ij: Tile<f32, { [BM, 1] }> = l_ij.reshape(const_shape![BM, 1]);
            let alpha: Tile<f32, { [BM, 1] }> = exp(m_i - m_ij);
            l_i = fma(l_i, alpha, l_ij, rounding::NearestEven, ftz::Disabled);
            let alpha: Tile<f32, { [BM, D] }> = alpha.broadcast(const_shape![BM, D]);
            acc = acc * alpha;

            let v_tile: Tile<f16, { [1, BN, D] }> = v_part.load([kv_head_idx, j, 0i32]);
            let v_tile: Tile<f32, { [BN, D] }> = convert_tile(v_tile.reshape(const_shape![BN, D]));
            acc = mma(p, v_tile, acc);
            m_i = m_ij;
        }

        let eps: Tile<f32, { [BM, 1] }> = constant(1.0e-8f32, const_shape![BM, 1]);
        let l_i: Tile<f32, { [BM, 1] }> = max_tile(l_i, eps);
        acc = true_div(acc, l_i.broadcast(const_shape![BM, D]));
        let acc: Tile<f16, { [BM, 1, D] }> = convert_tile(acc.reshape(const_shape![BM, 1, D]));
        out.store(acc);
    }
}

#[allow(unused_imports)]
pub use kernels::{
    add_2d_f16, add_vec_f16, argmax_blocks_f16, embedding_batch_f16, embedding_f16,
    flash_attn_causal_f16, flash_attn_causal_seq_dynpos_f16, flash_attn_causal_seq_f16,
    flash_attn_f16, gather_row_f16, gemm_f16, kv_cache_update_f16, kv_cache_update_seq_dynpos_f16,
    kv_cache_update_seq_f16, rms_norm_f16, rope_f16, rope_seq_dynpos_f16, rope_seq_f16,
    silu_mul_2d_f16, silu_mul_vec_f16,
};

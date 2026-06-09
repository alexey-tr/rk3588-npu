#ifndef NPU_MATMUL_H
#define NPU_MATMUL_H

/*
 * Copyright (C) 2024  Jasbir Matharu, <jasjnuk@gmail.com>
 *
 * This file is part of rk3588-npu.
 *
 * rk3588-npu is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.

 * rk3588-npu is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.

 * You should have received a copy of the GNU General Public License
 * along with rk3588-npu.  If not, see <https://www.gnu.org/licenses/>.
 *
 */

typedef struct {
  uint16_t  m;
  uint16_t  k;
  uint16_t  n;

  uint32_t  input_dma;
  uint32_t  weights_dma;
  uint32_t  output_dma;

  uint64_t  *tasks;

  uint8_t   fp32tofp16;

  /* Override for plane-stride between output channel surfaces, in elements of M
   * (i.e. number of rows in the full destination plane). 0 = use this tile's m
   * (current behaviour: single-tile contiguous output). Non-zero lets a tile
   * write its (m, n) sub-rectangle into a larger pre-allocated output buffer
   * whose row count is `out_surf_stride`. */
  uint32_t  out_surf_stride;

  /* Fuse a ReLU (max(0, x)) epilogue into the DPU's BS stage. 0 = bypass (plain
   * matmul, default); 1 = clamp negatives to zero on the accumulator before the
   * output downcast. No extra DMA/op-count cost — see gen_matmul_task BS bits. */
  uint8_t   relu;

  /* Fuse a SCALAR bias add into the BS-stage ALU (operand from register, no DMA):
   * out = x + bias, applied before ReLU when both are set (relu(x + bias)).
   * bias_en=0 (default) bypasses it. bias_bits is the scalar reinterpreted in the
   * accumulator's dtype: FP32 bits for the fp16/bf16 paths, int32 for int8. */
  uint8_t   bias_en;
  uint32_t  bias_bits;

  /* Per-channel bias: bias[N] as fp32 must be appended by the caller right after the
   * packed weights (byte off = weight_bytes) in the weights_dma buffer. gen_matmul then
   * emits the BS BRDMA path (full DPU_RDMA block, BS_BASE = weights_dma + weight_bytes,
   * qd_en=0, enable 0x1d). Mutually exclusive with scalar bias_en. */
  uint8_t   pcbias_en;
  /* OUT: regcfg_amount the generated task uses (footer offset). 104 plain, 122 w/ pcbias. */
  uint16_t  regcfg_amount;
} matmul_params_t;

int gen_matmul_fp16(matmul_params_t *params);
int gen_matmul_bf16(matmul_params_t *params);
int gen_matmul_int8(matmul_params_t *params);
int feature_data(int C, int H, int W, int C2, int c, int h, int w);
int weight_fp16(int C, int k, int c);
int weight_int8(int C, int k, int c);

#endif // NPU_MATMUL_H

#ifndef NPU_DPU_H
#define NPU_DPU_H

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

#include <stdint.h>

enum tp_precision { tp_precision_8bit, tp_precision_16bit};

/* DPU EW_CFG 0x4070 [2] */
enum ew_op_type_t {
    ew_op_alu = 0,  /* ALU path: use ew_alu_algo */
    ew_op_mul = 1,  /* MUL path: element-wise multiply */
};

/* DPU EW_CFG 0x4070 [19:16] — only applies when ew_op_type == ew_op_alu */
enum ew_alu_algo_t {
    ew_alu_max   = 0,
    ew_alu_min   = 1,
    ew_alu_add   = 2,
    ew_alu_div   = 3,
    ew_alu_minus = 4,
    ew_alu_abs   = 5,
    ew_alu_neg   = 6,
    ew_alu_floor = 7,
    ew_alu_ceil  = 8,
};

typedef struct npu_dpu_desc {
 uint8_t burst_len;         // 0x400C [8:5]
 uint8_t conv_mode;         // 0x400C [4:3]
 uint8_t output_mode;       // 0x400C [2:1]
 uint8_t flying_mode;       // 0x400C [0]
 uint8_t out_precision;     // 0x4010 [31:29]
 uint8_t in_precision;      // 0x4010 [28:26]
 uint8_t proc_precision;    // 0x4010 [2:0]
 uint32_t dst_base_addr;    // 0x4020
 uint32_t dst_surf_stride;  // 0x4024
 uint16_t width;            // 0x4030
 uint16_t height;           // 0x4034
 uint16_t channel;          // 0x403C [28:16] [12:0]
 uint8_t bs_bypass;         // 0x4040 [0]
 uint8_t bs_alu_bypass;     // 0x4040 [1]
 uint8_t bs_mul_bypass;     // 0x4040 [4]
 uint8_t bs_relu_bypass;    // 0x4040 [6]
 uint8_t tp_org_en;         // 0x4050 [27]
 uint8_t od_bypass;         // 0x4050 [1]
 uint8_t size_e_2;          // 0x4050 [10:8]
 uint8_t size_e_1;          // 0x4050 [7:5]
 uint8_t size_e_0;          // 0x4050 [4:2]
 enum tp_precision tp_precision; // 0x4058 [27]
 uint16_t size_c_wdma;      // 0x4058 [26:16]
 uint16_t channel_wdma;     // 0x4058 [12:0]
 uint16_t height_wdma;      // 0x405C [28:16]
 uint16_t width_wdma;       // 0x405C [12:0]
 uint8_t bn_relu_bypass;    // 0x4060 [6]
 uint8_t bn_mul_bypass;     // 0x4060 [4]
 uint8_t bn_alu_bypass;     // 0x4060 [1]
 uint8_t bn_bypass;         // 0x4060 [0]
 uint8_t ew_bypass;         // 0x4070 [0]
 uint8_t ew_op_bypass;      // 0x4070 [1]
 enum ew_op_type_t ew_op_type;  // 0x4070 [2]
 uint8_t ew_mul_prelu;      // 0x4070 [5]
 uint8_t ew_op_src;         // 0x4070 [6]
 uint8_t ew_lut_bypass;     // 0x4070 [7]
 uint8_t ew_op_cvt_bypass;  // 0x4070 [8]
 uint8_t ew_relu_bypass;    // 0x4070 [9]
 uint8_t ew_relux_en;       // 0x4070 [10]
 enum ew_alu_algo_t ew_alu_algo; // 0x4070 [19:16]
 uint8_t ew_binary_en;      // 0x4070 [20]
 uint8_t ew_equal_en;       // 0x4070 [21]
 uint8_t edata_size;        // 0x4070 [23:22]
 uint8_t ew_data_mode;      // 0x4070 [29:28]
 uint8_t ew_cvt_round;      // 0x4070 [30]
 uint8_t ew_cvt_type;       // 0x4070 [31]
 uint8_t fp32tofp16_en;     // 0x4084 [16]
 uint16_t out_cvt_scale;    // 0x4084 [15:0]
 uint32_t surf_add;         // 0x40C0
} npu_dpu_desc;

typedef struct npu_dpu_rdma_desc {
 uint16_t width;               // 0x500C
 uint16_t height;              // 0x5010
 uint16_t channel;             // 0x5014
 uint32_t src_base_addr;       // 0x5018
 uint8_t erdma_disable;        // 0x5034 [0]
 uint8_t ov4k_bypass;          // 0x5034 [1]
 uint8_t erdma_data_size;      // 0x5034 [3:2]
 uint8_t erdma_nonalign;       // 0x5034 [28]
 uint8_t erdma_surf_mode;      // 0x5034 [29]
 uint8_t erdma_data_mode;      // 0x5034 [31:30]
 uint32_t ew_base_addr;        // 0x5038
 uint32_t ew_surf_stride;      // 0x5040 [31:4]
 uint8_t flying_mode;          // 0x5044 [0]
 uint8_t conv_mode;            // 0x5044 [2:1]
 uint8_t mrdma_fp16tofp32_en;  // 0x5044 [3]
 uint8_t mrdma_disable;        // 0x5044 [4]
 uint8_t proc_precision;       // 0x5044 [7:5]
 uint8_t comb_use;             // 0x5044 [10:8]
 uint8_t burst_len;            // 0x5044 [14:11]
 uint8_t in_precision;         // 0x5044 [17:15]
 uint8_t m_weight;             // 0x5068 [7:0]
 uint8_t b_weight;             // 0x5068 [15:8]
 uint8_t n_weight;             // 0x5068 [23:16]
 uint8_t e_weight;             // 0x5068 [31:24]
 uint32_t ew_surf_notch;       // 0x506C
} npu_dpu_rdma_desc;

#endif // NPU_DPU_H

/*
 * layer_template_L3.c
 * Alessio Burrello <alessio.burrello@unibo.it>
 *
 * Copyright (C) 2019-2020 University of Bologna
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. 
 * SPDX-License-Identifier: Apache-2.0
 *
 * Modified by:
 * Yoga Esa Wibowo, ETH Zurich (ywibowo@student.ethz.ch)
 * Cristian Cioflan, ETH Zurich (cioflanc@iis.ee.ethz.ch)
 * Thorir Mar Ingolfsson, ETH Zurich (thoriri@iis.ee.ethz.ch)
 * Michael Hersche, IBM Research Zurich (her@zurich.ibm.com)
 * Leo Zhao, ETH Zurich (lezhao@student.ethz.ch)
 */
#include "BackpropFullyConnected63.h"
#include "BackpropFullyConnected63_L2.h"
#include "dory_get_tile.h"
#include "pmsis.h"
#include "bsp/fs.h"
#include "bsp/fs/readfs.h"
#include "bsp/flash.h"
#include "bsp/ram.h"
#include "bsp/flash/hyperflash.h"
#include "bsp/ram/hyperram.h"
#include "net_utils.h"

#define LEN_IN 1280
#define LEN_OUT 8
#define ALL_IN 1280
#define ALL_OUT 256
#define BYTE_IN 1
#define BYTE_OUT 4
#define BYTE_W 4
#define BYTE_B 4

void __attribute__ ((noinline)) BackpropFullyConnected63(void *args)
{
  layer_args_t *layer_args = (layer_args_t *)args;
  const unsigned int l3_x = layer_args->L3_input;
  const unsigned int l3_y = layer_args->L3_output;
  const unsigned int l3_WG= layer_args->L3_after_weights;
  const unsigned int l2_x = layer_args->L2_input;
  const unsigned int l2_y = layer_args->L2_output;
  const unsigned int l2_WG= layer_args->L2_weights;
  const unsigned int batch_size = layer_args->bypass;
  const pi_device_t * ram = (pi_device_t *)layer_args->ram;

  layer_args_t tile_args = *layer_args;

  const struct {
    void *bias_grad;
    void *w_grad;
  } db[2] = {
    {
      .bias_grad = l2_WG + LEN_IN*LEN_OUT*BYTE_W, //location after w_grad db[0]
      .w_grad = l2_WG,
    },
    {
      .bias_grad = l2_WG + 2*LEN_IN*LEN_OUT*BYTE_W + LEN_OUT*BYTE_B, //location w_grad db[1]
      .w_grad = l2_WG + LEN_IN*LEN_OUT*BYTE_W + LEN_OUT*BYTE_B, //location after bias_grad db[0]
    }
  };

  int i_db_w = 0;

  // weight L3 tiling. Parameters
  pi_cl_ram_req_t req_wg_rd, req_biasg_rd;
  pi_cl_ram_req_t req_wg_wr, req_biasg_wr;
  // first tile transfer. Weights, k, lambda
  if(pi_core_id()==0)
  {
    pi_cl_ram_read(ram, l3_WG, db[i_db_w].w_grad, LEN_IN*LEN_OUT*BYTE_W, &req_wg_rd); //10240 = 8*1280
    pi_cl_ram_read(ram, l3_WG+ALL_IN*ALL_OUT*BYTE_W, db[i_db_w].bias_grad, LEN_OUT*BYTE_B, &req_biasg_rd);
    pi_cl_ram_read_wait(&req_wg_rd);
    pi_cl_ram_read_wait(&req_biasg_rd);
  }
  // switching buffers

  int j = 0;

  // loop over weight tiles
  int offset_w_rd = 0;
  int offset_b_rd = ALL_IN*ALL_OUT*BYTE_W;  
  int offset_w_wr = 0;
  int offset_b_wr = ALL_IN*ALL_OUT*BYTE_W;
  for(int k = 0; k < ALL_OUT/LEN_OUT; k++) {
    if (k < ALL_OUT/LEN_OUT-1) {
      // Fetch next weights
      if(pi_core_id()==0) {
        offset_w_rd += LEN_IN*LEN_OUT*BYTE_W;
        offset_b_rd += LEN_OUT*BYTE_B;
        pi_cl_ram_read(ram, l3_WG + offset_w_rd, db[!i_db_w].w_grad, LEN_IN*LEN_OUT*BYTE_W, &req_wg_rd);
        pi_cl_ram_read(ram, l3_WG + offset_b_rd, db[!i_db_w].bias_grad, LEN_OUT*BYTE_B, &req_biasg_rd);
      }
    }  

    tile_args.L2_input = l2_x;
    tile_args.L2_output = dory_get_tile_3d(l2_y, j, 0, k, 1, 1, LEN_OUT, 1, ALL_OUT, 0, 0, 0, 0, 0, 0, 8*BYTE_OUT);
    tile_args.L2_weights = db[i_db_w].w_grad;

    // execution of L2-L1 layer. Either top, middle or bottom layer.
    pi_cl_team_barrier(0);

    if (j==0) {
      BackpropFullyConnected63_L2((void*)&tile_args);
    } else if (j == (0)) {
      BackpropFullyConnected63_L2((void*)&tile_args);
    } else {
      BackpropFullyConnected63_L2((void*)&tile_args);
    }    

      pi_cl_team_barrier(0);
        if(pi_core_id()==0)
        {
          // waiting for weights, lambda, and k
          pi_cl_ram_read_wait(&req_wg_rd);
          pi_cl_ram_read_wait(&req_biasg_rd);
          // if (j > 0){
          //   pi_cl_ram_write_wait(&req_wg_wr);
          //   pi_cl_ram_write_wait(&req_biasg_wr);
          // }
          pi_cl_ram_write(ram, l3_WG + offset_w_wr, db[i_db_w].w_grad, LEN_IN*LEN_OUT*BYTE_W, &req_wg_wr);
          pi_cl_ram_write(ram, l3_WG + offset_b_wr, db[i_db_w].bias_grad, LEN_OUT*BYTE_B, &req_biasg_wr);
          offset_w_wr += LEN_IN*LEN_OUT*BYTE_W;
          offset_b_wr += LEN_OUT*BYTE_B;
        }
        i_db_w = !i_db_w;
      }   
  if(pi_core_id()==0) {
    pi_cl_ram_write_wait(&req_wg_wr);
    pi_cl_ram_write_wait(&req_biasg_wr);
  }
  
  pi_cl_team_barrier(0);
}

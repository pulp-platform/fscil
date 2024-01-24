/*
 * layer_template.c
 * Alessio Burrello <alessio.burrello@unibo.it>
 * Francesco Conti <f.conti@unibo.it>
 *
 * Copyright (C) 2018-2020 University of Bologna
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

#include "BackpropFullyConnected63_L2.h"
#include "pulp.h"
#include "pmsis.h"
#include "dory_get_tile.h"
#include "dory_dma.h"
#include "pulp_nn_kernels.h"
#include "pulp_nn_utils.h"

#define LEN_IN 1280
#define LEN_OUT 8
#define ALL_IN 1280
#define ALL_OUT 256
#define BYTE_IN 1
#define BYTE_OUT 4
#define BYTE_W 4
#define BYTE_B 4

#if BYTE_IN==4
  typedef uint32_t type_in;
#elif BYTE_IN==2
  typedef uint16_t type_in;
#elif BYTE_IN==1
  typedef uint8_t type_in;
#endif

#if BYTE_OUT==4
  typedef int32_t type_out;
#elif BYTE_OUT==2
  typedef int16_t type_out;
#elif BYTE_OUT==1
  typedef int8_t type_out;
#endif

#if BYTE_W==4
  typedef int32_t type_w;
#elif BYTE_W==2
  typedef int16_t type_w;
#elif BYTE_W==1
  typedef int8_t type_w;
#endif

#if BYTE_B==4
  typedef int32_t type_b;
#elif BYTE_B==2
  typedef int16_t type_b;
#elif BYTE_B==1
  typedef int8_t type_b;
#endif

// void BackpropFullyConnected63_L2(
void calculate_grad(
  void *args
) {
  
  unsigned int *real_arg = (unsigned int *) args;
  type_in *pIn = (type_in *) real_arg[0];
  type_out *pOutGrad = (type_out *) real_arg[1];
  type_w *pWeightGrad = (type_w *) real_arg[2];
  type_b *pBiasGrad = (type_b *)  real_arg[3];
  unsigned int dim_vec = real_arg[4];
  unsigned int num_o_neurons = real_arg[5];
  unsigned int batch_size =  real_arg[6];
  
  // unsigned int *real_arg = (unsigned int *) args;
  // uint8_t *pIn = (uint8_t *) real_arg[3];
  // int32_t *pOutGrad = (int32_t *) real_arg[5];
  // int32_t *pWeightGrad = (int32_t *) real_arg[6];
  // int32_t *pBiasGrad = (int32_t *) pWeightGrad + 10240;
  // unsigned int dim_vec = 1280;
  // unsigned int num_o_neurons = 8;


  int core_id = pi_core_id();
  int chunk = (num_o_neurons / NUM_CORES) + ((num_o_neurons & (NUM_CORES-1))!=0);
  int start = min(chunk * core_id, num_o_neurons);
  int stop = min(start + chunk, num_o_neurons);

  int i;
  for(i=start; i<stop; i+=1)
  {
    // if (pBiasGrad != NULL)
    //   pBiasGrad[i] += ((int32_t *) pOutGrad)[i];

    type_in *pA = pIn;
    type_out *pB = &(pOutGrad[i]);
    type_w *pC = &(pWeightGrad[i * dim_vec]);
    type_b *pD = &(pBiasGrad[i]);

    for (int k=0; k<batch_size; k++){
      for (int j=0; j<dim_vec; j++)
      {
        pC[j] = pC[j] + pA[j+k*dim_vec] * pB[k*num_o_neurons]; //accumulate gradient of the weight
      }
      *pD += pB[k*num_o_neurons]; //accumulate gradient of the bias
    }
  }
  pi_cl_team_barrier(0);
}

void BackpropFullyConnected63_L2(
  void *args
) {
  unsigned int *real_arg = (unsigned int *) args;
  uint8_t *l2_x = (uint8_t *) real_arg[3];
  uint8_t *l2_y = (int32_t *) real_arg[5];
  uint8_t *l2_wg = (int32_t *) real_arg[6];
  uint8_t *l2_bg = (int32_t *) l2_wg + LEN_IN*LEN_OUT;
  uint8_t *l1_buffer = (uint8_t *) real_arg[7];
  int32_t batch_size = (int32_t *) real_arg[4];

  uint32_t dory_dma_channel = dory_dma_allocate();
  const int offset_x = 0;
  const int offset_y = batch_size*(LEN_IN/2)*BYTE_IN; 
  const int offset_wg = offset_y + batch_size*LEN_OUT*BYTE_OUT;
  const int offset_bg = offset_wg + LEN_OUT*(LEN_IN/2)*BYTE_W;

  volatile DMA_copy DMA_copy_y, DMA_copy_x;
  volatile DMA_copy DMA_copy_W_in, DMA_copy_W_out;
  volatile DMA_copy DMA_copy_b_in, DMA_copy_b_out;

  DMA_copy_x.hwc_to_chw = 0;
  DMA_copy_x.stride_2d = ALL_IN*BYTE_IN;
  DMA_copy_x.stride_1d = ALL_IN*BYTE_IN;
  DMA_copy_x.dir = 1;
  DMA_copy_x.tid = dory_dma_channel;
  DMA_copy_x.ext = l2_x;
  DMA_copy_x.loc = (l1_buffer + offset_x);
  DMA_copy_x.number_of_2d_copies = batch_size;
  DMA_copy_x.number_of_1d_copies = 1;
  DMA_copy_x.length_1d_copy = (LEN_IN/2)*BYTE_IN;
  dory_dma_memcpy_async(&DMA_copy_x);
  dory_dma_barrier(&DMA_copy_x);
  
  DMA_copy_y.hwc_to_chw = 0;
  DMA_copy_y.stride_2d = ALL_OUT*BYTE_OUT;
  DMA_copy_y.stride_1d = ALL_OUT*BYTE_OUT;
  DMA_copy_y.dir = 1;
  DMA_copy_y.tid = dory_dma_channel;
  DMA_copy_y.ext = l2_y;
  DMA_copy_y.loc = (l1_buffer + offset_y);
  DMA_copy_y.number_of_2d_copies = batch_size;
  DMA_copy_y.number_of_1d_copies = 1;
  DMA_copy_y.length_1d_copy = LEN_OUT*BYTE_OUT;
  dory_dma_memcpy_async(&DMA_copy_y); 
  dory_dma_barrier(&DMA_copy_y);  

  DMA_copy_W_in.hwc_to_chw = 0;
  DMA_copy_W_in.stride_2d = ALL_IN*BYTE_W;
  DMA_copy_W_in.stride_1d = ALL_IN*BYTE_W;
  DMA_copy_W_in.dir = 1;
  DMA_copy_W_in.tid = dory_dma_channel;
  DMA_copy_W_in.ext = l2_wg;
  DMA_copy_W_in.loc = (l1_buffer + offset_wg);
  DMA_copy_W_in.number_of_2d_copies = LEN_OUT;
  DMA_copy_W_in.number_of_1d_copies = 1;
  DMA_copy_W_in.length_1d_copy = (LEN_IN/2)*BYTE_W;
  dory_dma_memcpy_async(&DMA_copy_W_in);
  dory_dma_barrier(&DMA_copy_W_in);

  DMA_copy_b_in.hwc_to_chw = 0;
  DMA_copy_b_in.stride_2d = 0;
  DMA_copy_b_in.stride_1d = 0;
  DMA_copy_b_in.dir = 1;
  DMA_copy_b_in.tid = dory_dma_channel;
  DMA_copy_b_in.ext = (uint32_t) l2_bg;
  DMA_copy_b_in.loc = (uint32_t) (l1_buffer + offset_bg);
  DMA_copy_b_in.number_of_2d_copies = 1;
  DMA_copy_b_in.number_of_1d_copies = 1;
  DMA_copy_b_in.length_1d_copy = (uint16_t) (LEN_OUT*BYTE_B);
  dory_dma_memcpy_async(&DMA_copy_b_in);
  dory_dma_barrier(&DMA_copy_b_in);

  int input_args[7] = {
    l1_buffer + offset_x,
    l1_buffer + offset_y,
    l1_buffer + offset_wg,
    l1_buffer + offset_bg,
    LEN_IN/2,
    LEN_OUT,
    batch_size
  };
  pi_cl_team_barrier(0);
  asm volatile("": : :"memory");
  calculate_grad(input_args);
  pi_cl_team_barrier(0);

  DMA_copy_W_out.hwc_to_chw = 0;
  DMA_copy_W_out.stride_2d = ALL_IN*BYTE_W;
  DMA_copy_W_out.stride_1d = ALL_IN*BYTE_W;
  DMA_copy_W_out.dir = 0;
  DMA_copy_W_out.tid = dory_dma_channel;
  DMA_copy_W_out.ext = l2_wg;
  DMA_copy_W_out.loc = (l1_buffer + offset_wg);
  DMA_copy_W_out.number_of_2d_copies = LEN_OUT;
  DMA_copy_W_out.number_of_1d_copies = 1;
  DMA_copy_W_out.length_1d_copy = (LEN_IN/2)*BYTE_W;
  dory_dma_memcpy_async(&DMA_copy_W_out);
  dory_dma_barrier(&DMA_copy_W_out);

  DMA_copy_b_out.hwc_to_chw = 0;
  DMA_copy_b_out.stride_2d = 0;
  DMA_copy_b_out.stride_1d = 0;
  DMA_copy_b_out.dir = 0;
  DMA_copy_b_out.tid = dory_dma_channel;
  DMA_copy_b_out.ext = (uint32_t) l2_bg;
  DMA_copy_b_out.loc = (uint32_t) (l1_buffer + offset_bg);
  DMA_copy_b_out.number_of_2d_copies = 1;
  DMA_copy_b_out.number_of_1d_copies = 1;
  DMA_copy_b_out.length_1d_copy = (uint16_t) (LEN_OUT*BYTE_B);
  dory_dma_memcpy_async(&DMA_copy_b_out);
  dory_dma_barrier(&DMA_copy_b_out);


  // refetch x for tiling
  DMA_copy_x.ext = l2_x+(LEN_IN/2)*BYTE_IN;
  dory_dma_memcpy_async(&DMA_copy_x);
  dory_dma_barrier(&DMA_copy_x);

  // refetch w for tiling
  DMA_copy_W_in.ext = l2_wg + (LEN_IN/2)*BYTE_W;
  dory_dma_memcpy_async(&DMA_copy_W_in);
  dory_dma_barrier(&DMA_copy_W_in);

  // dont refetch y for tiling
  // dont refetch b for tiling

  pi_cl_team_barrier(0);
  asm volatile("": : :"memory");
  calculate_grad(input_args);
  pi_cl_team_barrier(0);

  // resave w for tiling
  DMA_copy_W_out.ext = l2_x+(LEN_IN/2)*BYTE_IN;
  dory_dma_memcpy_async(&DMA_copy_W_out);
  dory_dma_barrier(&DMA_copy_W_out);

  // dont resave b for tiling
  
  // Release DMA
  pi_cl_team_barrier(0);
  dory_dma_free(&DMA_copy_x);
}
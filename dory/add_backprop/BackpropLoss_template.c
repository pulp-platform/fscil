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

#include "BackpropLoss.h"
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

void plp_sqrt_f32s_xpulpv2(const float *__restrict__ pSrc, float *__restrict__ pRes) {

    const float threehalfs = 1.5f;
    float x2, y;

    union {
        float f;
        int32_t i;
    } conv;

    if (*pSrc > 0) {
        /* fast inverse square root with proper type punning */
        x2 = *pSrc * 0.5f;
        conv.f = *pSrc;
        conv.i = 0x5f3759df - (conv.i >> 1); /* evil floating point bit level hacking */
        y = conv.f;
        y = y * (threehalfs - (x2 * y * y)); /* newton 1st iter */
        y = y * (threehalfs - (x2 * y * y)); /* newton 2nd iter */
        *pRes = *pSrc * y;                   /* to square root */
    } else {
        *pRes = 0.f;
    }
}

float mysqrt(float x){
  float temp;
  float sq;
  temp = 0;
  sq = x/2;
  while (temp!=sq){
    temp = sq;
    sq = (x/temp + temp)/2;
  }
  return sq;
}

// void BackpropFullyConnected63_L2(
void calculate_grad_cossim( int32_t *gt, int32_t *x, int32_t *out, int arr_len, float eps_x, float *temp_mem){
  int core_id = pi_core_id();
  int chunk = (arr_len / NUM_CORES) + ((arr_len % NUM_CORES - core_id)>0);
  int start = (arr_len * core_id) / NUM_CORES;
  int stop = (arr_len * (core_id+1)) / NUM_CORES;

  float *mag_x_p2_temp = temp_mem;
  float *mag_gt_p2_temp = temp_mem + NUM_CORES;
  float *cossim_temp = temp_mem + 2*NUM_CORES;

  mag_x_p2_temp[core_id] = 0;
  mag_gt_p2_temp[core_id] = 0;
  cossim_temp[core_id] = 0;
  for (int i=start; i<stop; i++){
    mag_x_p2_temp[core_id]  += ((float) (x[i]))*x[i];
    mag_gt_p2_temp[core_id] += ((float) (gt[i]))*gt[i];
    cossim_temp[core_id]    += ((float) (gt[i]))*x[i];
  }
  pi_cl_team_barrier(0);
  float mag_x_p2, mag_gt_p2, cossim;
  mag_x_p2=0; mag_gt_p2=0; cossim=0;
  for(int i=0; i<NUM_CORES; i++){
    mag_x_p2  += mag_x_p2_temp[i];
    mag_gt_p2 += mag_gt_p2_temp[i];
    cossim    += cossim_temp[i];
  }
  // printf("%e, %e, %e, %e, %e\n", mag_x_p2, mag_gt_p2, cossim, (float) (x[0]), (float) (gt[0]));

  // __builtin_sqrtf
  float mag_x, mag_gt;
  // mag_x  = sqrt(mag_x_p2); 
  // mag_gt = sqrt(mag_gt_p2);
  plp_sqrt_f32s_xpulpv2(&mag_x_p2 ,&mag_x ); 
  plp_sqrt_f32s_xpulpv2(&mag_gt_p2,&mag_gt);
  cossim = cossim/(mag_x*mag_gt);
  
  float div_gt = mag_x*mag_gt;
  float div_x = mag_x*mag_x/cossim;
  for (int i=start; i<stop; i++){
    // the sign of this calculation is reversed because we need to maximize cosine similarity
    // we dont want to minimize cosisne similarity
    out[i] = (int32_t) ((x[i]/div_x - gt[i]/div_gt)/(eps_x*eps_x));
  }
  // printf("%e, %e, %e, %e, %e, %e\n", mag_x, mag_gt, cossim, div_gt, div_x, eps_x);

}

void BackpropLossCossim(
  void *args
) {
  unsigned int *real_arg = (unsigned int *) args;
  uint8_t *l2_x = (uint8_t *) real_arg[3];
  uint8_t *l2_y = (int32_t *) real_arg[5];
  uint8_t *l2_T = (int32_t *) real_arg[6];
  uint8_t *l1_buffer = (uint8_t *) real_arg[7];
  int32_t batch_size = (int32_t *) real_arg[4];
  float eps_out = ((float *) (real_arg[11]))[0];

  uint32_t dory_dma_channel = dory_dma_allocate();
  const int all_byte_out = ALL_OUT*BYTE_OUT;
  const int offset_x = 0;
  const int offset_T = ALL_OUT*BYTE_OUT; 
  const int offset_y = 2*ALL_OUT*BYTE_OUT;
  const int offset_temp = 3*ALL_OUT*BYTE_OUT;

  volatile DMA_copy DMA_copy_y, DMA_copy_x, DMA_copy_T;

  for (int i=0; i<batch_size; i++){
    DMA_copy_x.hwc_to_chw = 0;
    DMA_copy_x.stride_2d = all_byte_out;
    DMA_copy_x.stride_1d = all_byte_out;
    DMA_copy_x.dir = 1;
    DMA_copy_x.tid = dory_dma_channel;
    DMA_copy_x.ext = l2_x + i*all_byte_out;
    DMA_copy_x.loc = (l1_buffer + offset_x);
    DMA_copy_x.number_of_2d_copies = 1;
    DMA_copy_x.number_of_1d_copies = 1;
    DMA_copy_x.length_1d_copy = all_byte_out;
    dory_dma_memcpy_async(&DMA_copy_x);
    dory_dma_barrier(&DMA_copy_x);
    
    DMA_copy_T.hwc_to_chw = 0;
    DMA_copy_T.stride_2d = all_byte_out;
    DMA_copy_T.stride_1d = all_byte_out;
    DMA_copy_T.dir = 1;
    DMA_copy_T.tid = dory_dma_channel;
    DMA_copy_T.ext = l2_T + i*all_byte_out;
    DMA_copy_T.loc = (l1_buffer + offset_T);
    DMA_copy_T.number_of_2d_copies = 1;
    DMA_copy_T.number_of_1d_copies = 1;
    DMA_copy_T.length_1d_copy = all_byte_out;
    dory_dma_memcpy_async(&DMA_copy_T); 
    dory_dma_barrier(&DMA_copy_T);  

    pi_cl_team_barrier(0);
    asm volatile("": : :"memory");
    calculate_grad_cossim(
      (l1_buffer + offset_T),
      (l1_buffer + offset_x),
      (l1_buffer + offset_y),
      ALL_OUT,
      eps_out,
      (float *) (l1_buffer + offset_temp)
    );
    pi_cl_team_barrier(0);

    DMA_copy_y.hwc_to_chw = 0;
    DMA_copy_y.stride_2d = all_byte_out;
    DMA_copy_y.stride_1d = all_byte_out;
    DMA_copy_y.dir = 0;
    DMA_copy_y.tid = dory_dma_channel;
    DMA_copy_y.ext = l2_y + i*all_byte_out;
    DMA_copy_y.loc = (l1_buffer + offset_y);
    DMA_copy_y.number_of_2d_copies = 1;
    DMA_copy_y.number_of_1d_copies = 1;
    DMA_copy_y.length_1d_copy = all_byte_out;
    dory_dma_memcpy_async(&DMA_copy_y);
    dory_dma_barrier(&DMA_copy_y);
  }

  // Release DMA
  pi_cl_team_barrier(0);
  dory_dma_free(&DMA_copy_x);
}
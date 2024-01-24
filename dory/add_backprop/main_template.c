/*
 * test_template.c
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
#include "mem.h"
#include "network.h"

#include "pmsis.h"

#define VERBOSE 1
#define L2_INPUT_SIZE 205000
#define MAX_CLASS 110
#define PROTOTYPE_LEN 256
#define LAST_LAYER_INP_LEN 1280
#define TRAINING_PROCESS_BATCH 20 

int main () {
  PMU_set_voltage(1000, 0);
  pi_time_wait_us(10000);
  pi_freq_set(PI_FREQ_DOMAIN_FC, 100000000);
  pi_time_wait_us(10000);
  pi_freq_set(PI_FREQ_DOMAIN_CL, 100000000);
  pi_time_wait_us(10000);

/*
    Opening of Filesystem and Ram
*/
  mem_init();
  network_initialize();
  
  unsigned int W, I, O, WS;
  network_get_mem_addr(&W, &I, &O, &WS);
  backprop_mem_init(W, I, O, WS);
  /*
    Allocating space for input
  */
  int last_layer_sz = 1280;
  int target_label_sz = 1024;
  uint8_t *last_layer_out = pi_l2_malloc(last_layer_sz);
  int32_t *target_label = (int32_t *) pi_l2_malloc(target_label_sz);

  void *l2_sync_in = pi_l2_malloc(L2_INPUT_SIZE);

  int32_t gw_div=1, gb_div=1;
  float eps_in=0.01, eps_out=0.00002, eps_w=0.0003;

  printf("Test back propagation\n");
  backprop_set_eps(&eps_in, &eps_out, &eps_w);
  int32_t output_loc = l2_sync_in+L2_INPUT_SIZE-TRAINING_PROCESS_BATCH*2*4*PROTOTYPE_LEN;
  int32_t target_loc = l2_sync_in+L2_INPUT_SIZE-TRAINING_PROCESS_BATCH*4*PROTOTYPE_LEN;
  int32_t new_L2_INPUT_SIZE = L2_INPUT_SIZE-TRAINING_PROCESS_BATCH*(2*4*PROTOTYPE_LEN+LAST_LAYER_INP_LEN);
  int z =0;
  for (int j=0; j<MAX_CLASS; j++){
      // printf("%d\n", j);
      int32_t batch_loc = z%TRAINING_PROCESS_BATCH;
      
      // Get and storing input
      for(int i=0; i<LAST_LAYER_INP_LEN; i++){
          ((uint8_t *) l2_sync_in)[i+batch_loc*LAST_LAYER_INP_LEN] = (i+2*j)%256;
      }
      // Calculate and storing output
      network_run(l2_sync_in+batch_loc*LAST_LAYER_INP_LEN, new_L2_INPUT_SIZE, output_loc+batch_loc*4*PROTOTYPE_LEN, 0, 2);
      // Get and storing target
      for(int i=0; i<PROTOTYPE_LEN; i++){
          ((uint32_t *) (target_loc+batch_loc*4*PROTOTYPE_LEN))[i] = 0;
      }
      ((uint32_t *) (target_loc+batch_loc*4*PROTOTYPE_LEN))[2*j] = 1;
      //counter
      z++;
      // backprop function : backprop_run(LOUT, LIN, TARGET, BUFFER_SZ, batch_size, 0)
      if ((z%TRAINING_PROCESS_BATCH) == 0){
          backprop_run(output_loc, l2_sync_in, target_loc, L2_INPUT_SIZE, TRAINING_PROCESS_BATCH, 0);
      }
  }
  // handle case total class is not divisible by TRAINING_PROCESS_BATCH
  if ((z%TRAINING_PROCESS_BATCH)!=0){
      backprop_run(output_loc, l2_sync_in, target_loc, L2_INPUT_SIZE, z%TRAINING_PROCESS_BATCH, 0);
  }
  backprop_weight_update(l2_sync_in, L2_INPUT_SIZE, gw_div*z, gb_div*z);
  printf("Test DONE\n");

  printf("Test forward propagation\n");
  network_run(l2_sync_in, L2_INPUT_SIZE, l2_sync_in, 0, 1);
  printf("Test DONE\n\n");
  
  network_terminate();
  backprop_terminate();
  pi_l2_free(l2_sync_in, L2_INPUT_SIZE);
}

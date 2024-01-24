// Copyright (C) 2022-2024 ETH Zurich
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// SPDX-License-Identifier: Apache-2.0
// ==============================================================================
// 
// Authors: 
// Yoga Esa Wibowo, ETH Zurich (ywibowo@student.ethz.ch)
// Cristian Cioflan, ETH Zurich (cioflanc@iis.ee.ethz.ch)
// Thorir Mar Ingolfsson, ETH Zurich (thoriri@iis.ee.ethz.ch)
// Michael Hersche, IBM Research Zurich (her@zurich.ibm.com)
// Leo Zhao, ETH Zurich (lezhao@student.ethz.ch)


#ifndef __NETWORK_H__
#define __NETWORK_H__

// Emmulator functions: it mimic the behaviour of generated neural network from dory
// some variables that can be found in generated network.h 150528
static int activations_size[2] = {3072, 1280};
static int activations_out_size[2] = {1280, 1024};

// some functions that you will find in netowrk.c
void network_initialize();
void network_terminate();
void network_run(void *l2_buffer, size_t l2_buffer_size, void *l2_final_output, int exec, int type_run);
void network_run_cluster(void * args);
void network_run_cluster_conv(void * args);
void network_run_cluster_last(void * args);
void backprop_run_cluster_last(void * args);
void backprop_mem_init(unsigned int W, unsigned int I, unsigned int O, unsigned int WS);
void network_get_mem_addr(unsigned int *W, unsigned int *I, unsigned int *O, unsigned int *WS);
void backprop_run(void *l2_buffer, void *last_layer_out, void *target_label, size_t l2_buffer_size, void *l2_final_output, int exec);
void backprop_weight_update(void *l2_buffer, int l2_buffer_sz, int divider);

#endif  // __NETWORK_H__
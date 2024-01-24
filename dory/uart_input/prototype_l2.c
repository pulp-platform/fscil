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

#include "prototype.h"

void init_prototype(struct prototype *prot, int max_len, int protolen){
  prot->sums = (int32_t **) pi_l2_malloc(sizeof(int32_t *) * max_len);
  prot->lengths = (uint16_t *) pi_l2_malloc(sizeof(uint16_t) * max_len);
  prot->one_proto = (uint32_t *) pi_l2_malloc(sizeof(uint32_t) * protolen);
  for (int i=0; i<max_len; i++){
    (prot->lengths)[i] = 0;
  }
  prot->max_proto = max_len;
  prot->len_proto = protolen;
}

void free_prototype(struct prototype *prot){
  for (int i=0; i<prot->max_proto; i++){
    if ((prot->lengths)[i]!=0){
        pi_l2_free((prot->sums)[i], sizeof(int32_t)*prot->len_proto);
    }
  }
  pi_l2_free(prot->sums, sizeof(int32_t *)*prot->max_proto);
  pi_l2_free(prot->lengths, sizeof(uint16_t)*prot->max_proto);
  pi_l2_free(prot->one_proto, sizeof(uint32_t)*prot->len_proto);
}

void reset_prototype(struct prototype *prot){
  for (int i=0; i<prot->max_proto; i++){
    if ((prot->lengths)[i]!=0){
        pi_l2_free((prot->sums)[i], sizeof(int32_t)*prot->len_proto);
    }
    (prot->lengths)[i] = 0;
  }
}

void update_prototype(struct prototype *prot, void *new_proto, int gt_class){
  // assert (gt_class < prot->max_proto);
  int32_t *int_proto = (int32_t *) new_proto;
  // Not a new prototype
  if ((prot->lengths)[gt_class]!=0){
    (prot->lengths)[gt_class] += 1;
    for (int i=0; i<prot->len_proto; i++){
      (prot->sums)[gt_class][i] += int_proto[i];
    }
  }
  //new class
  else{
    (prot->sums)[gt_class] = (int32_t *) pi_l2_malloc(sizeof(int32_t)*prot->len_proto);

    (prot->lengths)[gt_class] += 1;
    for (int i=0; i<prot->len_proto; i++){
      (prot->sums)[gt_class][i] = int_proto[i];
    }
  }
}


int32_t *get_one_proto(struct prototype *prot, int idx){
  if (prot->lengths[idx]!=0){
    for (int i=0; i<prot->len_proto; i++){
      prot->one_proto[i] = (prot->sums)[idx][i]/(prot->lengths)[idx];
    }
  }
  return prot->one_proto;
}

void set_one_proto(struct prototype *prot, char *inp_proto, int idx){
  // assert (gt_class < prot->max_proto);
  int32_t *int_proto = (int32_t *) inp_proto;
  // new class
  if ((prot->lengths)[idx]==0){
    (prot->sums)[idx] = (int32_t *) pi_l2_malloc(sizeof(int32_t)*prot->len_proto);
  }
  (prot->lengths)[idx] = 1;
  for (int i=0; i<prot->len_proto; i++){
    (prot->sums)[idx][i] = int_proto[i];
  }
}


int64_t check_proto_distance(struct prototype prot, void *check_proto, int idx){
  int64_t the_max = 0x7FFFFFFFFFFFFFFF;
  if (idx >= prot.max_proto){
    return the_max;
  }
  if ((prot.lengths)[idx]==0){
    return the_max;
  }
  
  int64_t dist = 0, temp;
  int32_t *int_proto = (int32_t *) check_proto;
  for (int i=0; i<prot.len_proto; i++){
    temp = (prot.sums)[idx][i]/(prot.lengths)[idx] - int_proto[i];
    dist += temp*temp;
  }
  return dist;
}

uint16_t get_class(struct prototype prot, void *check_proto){
  int64_t min_dist = 0x7FFFFFFFFFFFFFFF;
  int64_t dist;
  uint16_t best_idx = 0;
  for (int i=0; i<prot.max_proto; i++){
    dist = check_proto_distance(prot, check_proto, i);
    if (dist < min_dist){
        min_dist = dist;
        best_idx = i;
    }
  }
  return best_idx;
}

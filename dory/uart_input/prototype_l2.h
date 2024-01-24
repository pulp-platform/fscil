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

#ifndef __PROTOTYPE_H__
#define __PROTOTYPE_H__

#include "pmsis.h"
#include <stdint.h>

struct prototype {
  int32_t **sums;
  int32_t *one_proto; // for tempararily save prototype
  uint16_t *lengths;  // String
  uint16_t len_proto; // lenght of prototype vector
  uint16_t max_proto; // maximum number of prototype to be saved
};

void init_prototype(struct prototype *prot, int max_len, int protolen);

void free_prototype(struct prototype *prot);

void reset_prototype(struct prototype *prot);

void update_prototype(struct prototype *prot, void *new_proto, int gt_class);

int64_t check_proto_distance(struct prototype prot, void *check_proto, int idx);

int32_t *get_one_proto(struct prototype *prot, int idx);

uint16_t get_class(struct prototype prot, void *check_proto);

void set_one_proto(struct prototype *prot, char *inp_proto, int idx);

#endif  // __PROTOTYPE_H__
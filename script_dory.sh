#!/bin/bash
MYHOME="$(dirname $(dirname "$(readlink -f "${BASH_SOURCE[0]}")"))"
APP_ID=001
EXP_ID=001
PLATFORM=gvsoc
MODEL=mnetv2_x4
DORY_HOME=$MYHOME/dory
# "GAP8_SDK", "PULP_SDK", "GAP9_SDK"
PLATFORM="GAP8_SDK" 

if [[ "$PLATFORM" == "PULP_SDK" ]]; then
  # Choose 1 (pulp_sdk)
  # pulp sdk should have been installed
  unset GAP_RISCV_GCC_TOOLCHAIN
  export PULP_RISCV_GCC_TOOLCHAIN=$MYHOME/riscv-nn-toolchain
  source $MYHOME/pulp-sdk/configs/pulp-open-nn.sh
  find $DORY_HOME -path "*Hardware_targets/PULP*HW_description.json" -exec sed -E -i 's/("name" *: *)"gap_sdk"/\1"pulp-sdk"/' {} +
  echo "Compile with $PLATFORM"
elif  [[ "$PLATFORM" == "GAP8_SDK" ]]; then
  # Choose 1 (gap_sdk)
  # gap_sdk should have been installed
  # choose correct board the defuault here is gapuino_v3 with GAP8_V3
  unset PULP_RISCV_GCC_TOOLCHAIN
  export GAP_RISCV_GCC_TOOLCHAIN=$MYHOME/gap_riscv_toolchain_ubuntu
  source $MYHOME/gap_sdk/configs/gapuino_v3.sh
  find $DORY_HOME  -path "*Hardware_targets/PULP*HW_description.json" -exec sed -E -i 's/("name" *: *)"pulp-sdk"/\1"gap_sdk"/' {} +
  export LD_LIBRARY_PATH=$MYHOME/lib/mpfr6/lib/x86_64-linux-gnu/:$LD_LIBRARY_PATH
  echo "Compile with $PLATFORM"
elif [[ "$PLATFORM" == "GAP9_SDK" ]]; then
  # Choose 1 (gap_sdk_private)
  # gap_sdk should have been installed
  # choose correct board the defuault here is gap9_v2
  unset PULP_RISCV_GCC_TOOLCHAIN
  export GAP_RISCV_GCC_TOOLCHAIN=$MYHOME/gap_riscv_toolchain_ubuntu
  source $MYHOME/gap_sdk_private/configs/gap9_evk_audio.sh
  find $DORY_HOME -path "*Hardware_targets/PULP*HW_description.json" -exec sed -E -i 's/("name" *: *)"pulp-sdk"/\1"gap_sdk"/' {} +
  # export LD_LIBRARY_PATH=$MYHOME/lib/mpfr6/lib/x86_64-linux-gnu/:$LD_LIBRARY_PATH
  echo "Compile with $PLATFORM"
else
  echo "Wrong platform name"
fi


# Make sure folder exist
mkdir $MYHOME/dory/application

# # EXAMPLE 1: All example test
# # Pulp compiler
# python3 -m pytest --durations=0 -x test_PULP.py --compat "pulp-sdk" --appdir "application/$APP_ID"
# # GAP compiler
# python3 -m pytest --durations=0 -x test_PULP.py --compat "gap-sdk" --appdir "application/$APP_ID"

# # EXAMPLE 2: Dory example network
# python3 network_generate.py Quantlab PULP.PULP_gvsoc $MYHOME/dory/dory/dory_examples/config_files/config_Quantlab_MV2_8bits.json --optional mixed-sw --app_dir $MYHOME/dory/application/$APP_ID
# python3 network_generate.py Quantlab PULP.PULP_gvsoc $MYHOME/dory/dory/dory_examples/config_files/config_Quantlab_MV1_fast_xpnn.json --optional mixed-hw --app_dir $MYHOME/dory/application/$APP_ID
# cd $MYHOME/dory/application/$APP_ID && make clean all run platform=$PLATFORM CORE=8

# # EXAMPLE 3: Generate code and compile (standard)
# cd $MYHOME/dory && python3 network_generate.py Quantlab PULP.GAP8 $MYHOME/fscil/code/log/$MODEL/$EXP_ID/quantise/config_quantise.json --optional mixed-sw --app_dir $MYHOME/dory/application/$APP_ID
# cd $MYHOME/dory/application/$APP_ID && make all run platform=$PLATFORM CORE=8

# # EXAMPLE 4: Generate GAP8 board and integrate fscil communication
# cd $MYHOME/dory && python3 network_generate.py Quantlab PULP.GAP8 $MYHOME/fscil/code/log/$MODEL/$EXP_ID/quantise/config_quantise.json --optional mixed-sw --app_dir $MYHOME/dory/application/$APP_ID
# python $MYHOME/fscil/dory/uart_input/integrate_fscil.py $MYHOME/fscil/dory/uart_input $MYHOME/dory/application/$APP_ID
# cd $MYHOME/dory/application/$APP_ID && make clean all run platform=$PLATFORM CORE=8

# # EXAMPLE 5: Generate code for GAP8 and add backprop function
# cd $MYHOME/dory && python3 network_generate.py Quantlab PULP.GAP8 $MYHOME/fscil/code/log/$MODEL/$EXP_ID/quantise/config_quantise.json --optional mixed-sw --app_dir $MYHOME/dory/application/$APP_ID
# python $MYHOME/fscil/dory/add_backprop/add_backprop.py $MYHOME/fscil/dory/add_backprop $MYHOME/dory/application/$APP_ID 
# cd $MYHOME/dory/application/$APP_ID && make clean all run platform=$PLATFORM CORE=8

# EXAMPLE 6: Generate GAP8 board and integrate fscil communication and backprop
cd $MYHOME/dory && python3 network_generate.py Quantlab PULP.GAP8 $MYHOME/fscil/code/log/$MODEL/$EXP_ID/quantise/config_quantise.json --optional mixed-sw --app_dir $MYHOME/dory/application/$APP_ID
python $MYHOME/fscil/dory/add_backprop/add_backprop.py $MYHOME/fscil/dory/add_backprop $MYHOME/dory/application/$APP_ID 
python $MYHOME/fscil/dory/uart_input/integrate_fscil.py $MYHOME/fscil/dory/uart_input $MYHOME/dory/application/$APP_ID
cd $MYHOME/dory/application/$APP_ID && make all run platform=$PLATFORM CORE=8

# # EXAMPLE 7: Generate GAP8 Code then transfer it to GAP9 version
# cd $MYHOME/dory && python3 network_generate.py Quantlab PULP.GAP8 $MYHOME/fscil/code/log/$MODEL/$EXP_ID/quantise/config_quantise.json --optional mixed-sw --app_dir $MYHOME/dory/application/$APP_ID
# python $MYHOME/fscil/dory/convert_to_gap9/convert_to_gap9.py $MYHOME/fscil/dory/convert_to_gap9 $MYHOME/dory/application/$APP_ID 
# cd $MYHOME/dory/application/$APP_ID && make clean && make all -j CORE=8 PMSIS_OS=freertos platform=$PLATFORM && make run CORE=8 PMSIS_OS=freertos platform=$PLATFORM

# # EXAMPLE 8: Generate code for GAP9 and add backprop function
# cd $MYHOME/dory && python3 network_generate.py Quantlab PULP.GAP8 $MYHOME/fscil/code/log/$MODEL/$EXP_ID/quantise/config_quantise.json --optional mixed-sw --app_dir $MYHOME/dory/application/$APP_ID
# python $MYHOME/fscil/dory/add_backprop/add_backprop.py $MYHOME/fscil/dory/add_backprop $MYHOME/dory/application/$APP_ID
# python $MYHOME/fscil/dory/convert_to_gap9/convert_to_gap9.py $MYHOME/fscil/dory/convert_to_gap9 $MYHOME/dory/application/$APP_ID 
# cd $MYHOME/dory/application/$APP_ID && make clean && make all -j CORE=8 PMSIS_OS=freertos platform=$PLATFORM && make run CORE=8 PMSIS_OS=freertos platform=$PLATFORM


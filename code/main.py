#
# Copyright 2022- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#
# Modified by:
# Yoga Esa Wibowo, ETH Zurich (ywibowo@student.ethz.ch)
# Cristian Cioflan, ETH Zurich (cioflanc@iis.ee.ethz.ch)
# Thorir Mar Ingolfsson, ETH Zurich (thoriri@iis.ee.ethz.ch)
# Michael Hersche, IBM Research Zurich (her@zurich.ibm.com)
# Leo Zhao, ETH Zurich (lezhao@student.ethz.ch)

# ==================================================================================================
# IMPORTS
# ==================================================================================================
import argparse
import os
import platform
if platform.system() == 'Darwin':
    # This had to be done to resolve a strange issue with OpenMP on MacOS
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import matplotlib
matplotlib.use('Agg')

from lib.param_FSCIL import ParamFSCIL

if __name__ == "__main__":
    # --------------------------------------------------------------------------------------------------
    # Parser
    # --------------------------------------------------------------------------------------------------

    # Argument parser
    verbose_parent_parser = argparse.ArgumentParser(add_help=False)
    verbose_parent_parser.add_argument('-v', '--verbose', help='Increase output verbosity.', action='store_true')

    log_parent_parser = argparse.ArgumentParser(add_help=False)
    log_group = log_parent_parser.add_mutually_exclusive_group()
    log_group.add_argument('-lp', '--logprefix', type=str, help='Specify a logging path prefix (i.e. root).')
    log_group.add_argument('-ls', '--logsuffix', type=str, help='Specify a logging subdirectory.')
    log_group.add_argument('-ld', '--logdir', type=str, help='Specify the whole logging path.')

    main_parser = argparse.ArgumentParser(parents=[verbose_parent_parser])
    # subparsers = main_parser.add_subparsers(help='Test', required=True)
    subparsers = main_parser.add_subparsers()

    # Simulation parser
    simulation_parser = subparsers.add_parser('simulation', parents=[log_parent_parser, verbose_parent_parser],
                                            help='Start a simulation of the model.',
                                            )
    simulation_parser.add_argument('-p', '--parameter', action='append', nargs=2, metavar=('key', 'value'),
                                help='Specify parameters with key-value pairs. Chainable.'
                                )
    simulation_parser.set_defaults(which='simulation')

    # Parse arguments
    args = main_parser.parse_args()

    # --------------------------------------------------------------------------------------------------
    # Execution
    # --------------------------------------------------------------------------------------------------

    # TODO: Make packages and provide them as arguments in the parser
    # TODO: Solve dependent parameters nicer

    if args.which in ['simulation']:
        param = ParamFSCIL(args)

        # Set randomess
        import torch as t
        import random
        import os
        import numpy as np
        random.seed(param.parameters['random_seed'])
        os.environ['PYTHONHASHSEED'] = str(param.parameters['random_seed'])
        np.random.seed(param.parameters['random_seed'])
        t.manual_seed(param.parameters['random_seed'])
        t.cuda.manual_seed(param.parameters['random_seed'])
        t.backends.cudnn.deterministic = True
        t.backends.cudnn.benchmark = False

        if param.parameters['trainstage']=="pretrain_baseFSCIL":
            from lib.run_FSCIL import pretrain_baseFSCIL
            func = pretrain_baseFSCIL
        elif param.parameters['trainstage']=="train_FSCIL":
            from lib.run_FSCIL import train_FSCIL
            func = train_FSCIL
        elif param.parameters['trainstage']=="metatrain_baseFSCIL":
            from lib.run_FSCIL import metatrain_baseFSCIL
            func = metatrain_baseFSCIL
        
        func(verbose=args.verbose, **(param.parameters))

    else:
        raise ValueError
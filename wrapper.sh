#!/bin/bash
BLOCK2_LIBS=/home/loharkar/QuEnAIS-quantum-embedding/quenais-env2/lib/python3.11/site-packages/block2.libs

export LD_PRELOAD="\
$BLOCK2_LIBS/libgomp-a34b3233.so.1.0.0:\
$BLOCK2_LIBS/libmkl_avx2.so.1:\
$BLOCK2_LIBS/libmkl_avx512.so.1:\
$BLOCK2_LIBS/libmkl_core-a1f8e95a.so.1:\
$BLOCK2_LIBS/libmkl_gnu_thread-76126a9d.so.1:\
$BLOCK2_LIBS/libmkl_intel_lp64-eeafede9.so.1"

export MKL_THREADING_LAYER=GNU
export MKL_DEBUG_CPU_TYPE=5
export LD_LIBRARY_PATH=$BLOCK2_LIBS:$LD_LIBRARY_PATH

exec /home/loharkar/QuEnAIS-quantum-embedding/quenais-env2/bin/block2main "$@"
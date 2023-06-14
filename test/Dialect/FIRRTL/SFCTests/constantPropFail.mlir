// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl-imconstprop), canonicalize{top-down region-simplify}, circuit(firrtl.module(firrtl-register-optimizer)))'  %s | FileCheck %s
// XFAIL: *

// These depend on more than constant prop.  They need to move.

  // CHECK-LABEL: module @padZeroReg
  module @padZeroReg(in %clock: !firrtl.clock, out %z: !firrtl.uint<16>) {
      %_r = reg droppable_name %clock  :  !firrtl.uint<8>
      strictconnect %_r, %_r : !firrtl.uint<8>
      %c171_ui8 = constant 171 : !firrtl.uint<8>
      %_n = node droppable_name %c171_ui8  : !firrtl.uint<8>
      %1 = cat %_n, %_r : (!firrtl.uint<8>, !firrtl.uint<8>) -> !firrtl.uint<16>
      strictconnect %z, %1 : !firrtl.uint<16>
    // CHECK: %[[TMP:.+]] = constant 43776 : !firrtl.uint<16>
    // CHECK-NEXT: strictconnect %z, %[[TMP]] : !firrtl.uint<16>
  }

// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl-imconstprop))' --mlir-print-local-scope --mlir-print-debuginfo %s | FileCheck %s

// CHECK-LABEL: circuit "Test"
firrtl.circuit "Test" {
  module private @Consts(out %c2 : !firrtl.uint<3>, out %c4 : !firrtl.uint<3>) {
    %c2_ui3 = constant 2 : !firrtl.uint<3>
    %c4_ui3 = constant 4 : !firrtl.uint<3>
    strictconnect %c2, %c2_ui3 : !firrtl.uint<3>
    strictconnect %c4, %c4_ui3 : !firrtl.uint<3>
  }
  // CHECK-LABEL: module @Test
  module @Test() {
    // CHECK-NEXT: %[[SIX:.+]] = constant 6 : !firrtl.uint<3>
    // CHECK-SAME: loc(fused["test.txt":5:2, "test.txt":4:2])

    %c2, %c4 = instance consts @Consts(out c2: !firrtl.uint<3>, out c4: !firrtl.uint<3>)

    %w_or = wire : !firrtl.uint<3> loc("test.txt":4:2)
    %w_add = wire : !firrtl.uint<3> loc("test.txt":5:2)

    %or = or %c2, %c4: (!firrtl.uint<3>, !firrtl.uint<3>) -> !firrtl.uint<3>
    %add = add %c2, %c4: (!firrtl.uint<3>, !firrtl.uint<3>) -> !firrtl.uint<4>
    %addtrunc = bits %add 2 to 0 : (!firrtl.uint<4>) -> !firrtl.uint<3>

    strictconnect %w_or, %or : !firrtl.uint<3>
    strictconnect %w_add, %addtrunc : !firrtl.uint<3>

    // CHECK: %n_or = node sym @n_or %[[SIX]]
    // CHECK: %n_add = node sym @n_add %[[SIX]]
    %n_or = node sym @n_or %w_or : !firrtl.uint<3>
    %n_add = node sym @n_add %w_add : !firrtl.uint<3>
  }
}

// RUN: circt-opt -cse %s | FileCheck %s

firrtl.circuit "And" {

// CHECK-LABEL: module @And
firrtl.module @And(in %in1: !firrtl.uint<4>, in %in2: !firrtl.uint<4>,
                   out %out1: !firrtl.uint<4>,
                   out %out2: !firrtl.uint<4>) {
  // And operations should get CSE'd.

  // CHECK: %0 = and %in1, %in2
  %0 = and %in1, %in2 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  // CHECK-NEXT: connect %out1, %0
  connect %out1, %0 : !firrtl.uint<4>, !firrtl.uint<4>

  // CHECK-NEXT: connect %out2, %0
  %1 = and %in1, %in2 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  connect %out2, %1 : !firrtl.uint<4>, !firrtl.uint<4>
}

// CHECK-LABEL: module @Wire
firrtl.module @Wire() {

   // CHECK: %_t = wire
   // CHECK-NEXT: %_t_0 = wire
   %w1 = wire {name = "_t"} : !firrtl.uint<1>
   %w2 = wire {name = "_t"} : !firrtl.uint<1>

  // CHECK-NEXT: connect %_t, %_t_0
  connect %w1, %w2 : !firrtl.uint<1>, !firrtl.uint<1>
}

// Invalids do not CSE
// CHECK-LABEL: module @Invalid
firrtl.module @Invalid(in %cond: !firrtl.uint<1>,
                   out %out: !firrtl.uint<4>) {
  // CHECK: invalid_ui4
  %invalid1_ui4 = invalidvalue : !firrtl.uint<4>
  // CHECK-NEXT: invalid_ui4_0
  %invalid2_ui4 = invalidvalue : !firrtl.uint<4>
  %7 = mux (%cond, %invalid1_ui4, %invalid2_ui4) : (!firrtl.uint<1>, !firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  connect %out, %7 : !firrtl.uint<4>, !firrtl.uint<4>

}


}

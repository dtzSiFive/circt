// RUN: circt-opt %s | circt-opt | FileCheck %s
// Basic MLIR operation parser round-tripping

firrtl.circuit "Basic" {
firrtl.extmodule @Basic()

// CHECK-LABEL: module @Intrinsics
firrtl.module @Intrinsics(in %ui : !firrtl.uint, in %clock: !firrtl.clock, in %ui1: !firrtl.uint<1>) {
  // CHECK-NEXT: int.sizeof %ui : (!firrtl.uint) -> !firrtl.uint<32>
  %size = int.sizeof %ui : (!firrtl.uint) -> !firrtl.uint<32>

  // CHECK-NEXT: int.isX %ui : !firrtl.uint
  %isx = int.isX %ui : !firrtl.uint

  // CHECK-NEXT: int.plusargs.test "foo"
  // CHECK-NEXT: int.plusargs.value "bar" : !firrtl.uint<5>
  %foo_found = int.plusargs.test "foo"
  %bar_found, %bar_value = int.plusargs.value "bar" : !firrtl.uint<5>

  // CHECK-NEXT: int.clock_gate %clock, %ui1
  // CHECK-NEXT: int.clock_gate %clock, %ui1, %ui1
  %cg0 = int.clock_gate %clock, %ui1
  %cg1 = int.clock_gate %clock, %ui1, %ui1
}

}

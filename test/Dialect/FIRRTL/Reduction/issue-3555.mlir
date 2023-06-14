// UNSUPPORTED: system-windows
//   See https://github.com/llvm/circt/issues/4129
// RUN: circt-reduce %s --test /bin/sh --test-arg -c --test-arg 'firtool "$0" 2>&1 | grep -q "error: sink \"x1.x\" not fully initialized"' --keep-best=0 --include root-port-pruner | FileCheck %s

// https://github.com/llvm/circt/issues/3555
firrtl.circuit "Foo"  {
  // CHECK-LABEL: module @Foo
  // CHECK-SAME:  () {
  module @Foo(in %x: !firrtl.uint<1>, out %y: !firrtl.uint<1>) {
    %x1_x = wire   : !firrtl.uint<1>
    %invalid_ui1 = invalidvalue : !firrtl.uint<1>
    // CHECK-NOT: strictconnect %y
    strictconnect %y, %invalid_ui1 : !firrtl.uint<1>
  }
  // CHECK: }
}

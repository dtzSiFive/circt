// UNSUPPORTED: system-windows
//   See https://github.com/llvm/circt/issues/4129
// RUN: circt-reduce %s --test /usr/bin/env --test-arg grep --test-arg -q --test-arg "module @Foo" --keep-best=0 --include connect-source-operand-0-forwarder | FileCheck %s
firrtl.circuit "Foo" {
  // CHECK-LABEL: module @Foo
  module @Foo(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %val: !firrtl.uint<2>) {
    %a = wire : !firrtl.uint<1>
    %b = reg %clock : !firrtl.clock, !firrtl.uint<1>
    %c = regreset %clock, %reset, %reset : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>
    %0 = bits %val 0 to 0 : (!firrtl.uint<2>) -> !firrtl.uint<1>
    connect %a, %0 : !firrtl.uint<1>, !firrtl.uint<1>
    connect %b, %0 : !firrtl.uint<1>, !firrtl.uint<1>
    connect %c, %0 : !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK-NEXT:   %a = wire  : !firrtl.uint<2>
    // CHECK-NEXT:   %b = reg %clock  : !firrtl.clock, !firrtl.uint<2>
    // CHECK-NEXT:   %c = reg %clock  : !firrtl.clock, !firrtl.uint<2>
    // CHECK-NEXT:   connect %a, %val : !firrtl.uint<2>, !firrtl.uint<2>
    // CHECK-NEXT:   connect %b, %val : !firrtl.uint<2>, !firrtl.uint<2>
    // CHECK-NEXT:   connect %c, %val : !firrtl.uint<2>, !firrtl.uint<2>
    // CHECK-NEXT: }
  }
}

// UNSUPPORTED: system-windows
//   See https://github.com/llvm/circt/issues/4129
// RUN: circt-reduce %s --test /usr/bin/env --test-arg grep --test-arg -q --test-arg "firrtl.module private @Bar" --keep-best=0 --include firrtl-remove-unused-ports | FileCheck %s

firrtl.circuit "Foo" {
  // CHECK-LABEL: module @Foo
  module @Foo(in %x: !firrtl.uint<1>, out %y: !firrtl.uint<3>) {
    // CHECK-NOT: %bar_a
    // CHECK-NOT: %bar_c
    // CHECK-NOT: %bar_e
    // CHECK: %bar_b, %bar_d = instance bar @Bar
    %bar_a, %bar_b, %bar_c, %bar_d, %bar_e = instance bar @Bar (in a: !firrtl.uint<1>, in b: !firrtl.uint<1>, out c: !firrtl.uint<1>, out d: !firrtl.uint<1>, out e: !firrtl.uint<1>)
    connect %bar_a, %x : !firrtl.uint<1>, !firrtl.uint<1>
    connect %bar_b, %x : !firrtl.uint<1>, !firrtl.uint<1>
    %0 = add %bar_c, %bar_d : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<2>
    %1 = add %0, %bar_e : (!firrtl.uint<2>, !firrtl.uint<1>) -> !firrtl.uint<3>
    connect %y, %1 : !firrtl.uint<3>, !firrtl.uint<3>
  }

  // We're only ever using ports %b and %d -- the rest should be stripped.
  // CHECK-LABEL: module private @Bar
  // CHECK-NOT: in %a
  // CHECK-SAME: in %b
  // CHECK-NOT: out %c
  // CHECK-SAME: out %d
  // CHECK-NOT: out %e
  module private @Bar(
    in %a: !firrtl.uint<1>,
    in %b: !firrtl.uint<1>,
    out %c: !firrtl.uint<1>,
    out %d: !firrtl.uint<1>,
    out %e: !firrtl.uint<1>
  ) {
    %invalid_ui1 = invalidvalue : !firrtl.uint<1>
    %0 = not %b : (!firrtl.uint<1>) -> !firrtl.uint<1>
    connect %c, %invalid_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    connect %d, %0 : !firrtl.uint<1>, !firrtl.uint<1>
    connect %e, %invalid_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
  }
}

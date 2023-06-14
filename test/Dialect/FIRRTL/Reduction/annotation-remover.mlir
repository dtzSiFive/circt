// UNSUPPORTED: system-windows
//   See https://github.com/llvm/circt/issues/4129
// RUN: circt-reduce %s --test /usr/bin/env --test-arg grep --test-arg -q --test-arg "%anotherWire = wire" --keep-best=0 --include annotation-remover | FileCheck %s

firrtl.circuit "Foo" {
  // CHECK: module @Foo
  // CHECK: %anotherWire = wire
  // CHECK-NOT: annotations
  module @Foo() {
    %oneWire = wire : !firrtl.uint<1>
    %anotherWire = wire {annotations = [{a}]} : !firrtl.uint<1>
  }
}

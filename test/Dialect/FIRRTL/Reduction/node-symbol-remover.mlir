// UNSUPPORTED: system-windows
//   See https://github.com/llvm/circt/issues/4129
// RUN: circt-reduce %s --test /usr/bin/env --test-arg grep --test-arg -q --test-arg "%anotherWire = node" --keep-best=0 --include node-symbol-remover | FileCheck %s

firrtl.circuit "Foo" {
  // CHECK: module @Foo
  // CHECK: %oneWire = wire
  // CHECK-NEXT: %anotherWire = node %oneWire
  module @Foo() {
    %oneWire = wire : !firrtl.uint<1>
    %anotherWire = node sym @SYM %oneWire : !firrtl.uint<1>
  }
}

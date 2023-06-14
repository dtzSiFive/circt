// RUN: circt-as %s -o - | circt-opt | FileCheck -strict-whitespace %s
// RUN: circt-opt %s -emit-bytecode | circt-dis | FileCheck -strict-whitespace  %s
// RUN: circt-as %s -o - | circt-dis | FileCheck -strict-whitespace  %s

firrtl.circuit "Top" {
  module @Top(in %in : !firrtl.uint<8>,
                     out %out : !firrtl.uint<8>) {
    strictconnect %out, %in : !firrtl.uint<8>
  }
}

// CHECK-LABEL: circuit "Top" {
// CHECK-NEXT:    module @Top(in %in: !firrtl.uint<8>, out %out: !firrtl.uint<8>) {
// CHECK-NEXT:      strictconnect %out, %in : !firrtl.uint<8>
// CHECK-NEXT:    }
// CHECK-NEXT:  }

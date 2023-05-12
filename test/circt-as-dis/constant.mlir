// Round-trip?
// RUNX: firtool --parse-only -emit-bytecode %s | mlir-opt --allow-unregistered-dialect -o /dev/null
// RUN: circt-as %s -o - | circt-dis -o %t
// Disable the following for now.
// RUNX: circt-as %s -o - | circt-opt | FileCheck -strict-whitespace %s
// RUNX: circt-opt %s -emit-bytecode | circt-dis | FileCheck -strict-whitespace  %s
// RUNX: circt-as %s -o - | circt-dis | FileCheck -strict-whitespace  %s

module {
  firrtl.circuit "ChipTop" {
    firrtl.module @ChipTop() {
      %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    }
  }
}
// CHECK-LABEL: firrtl.circuit "ChipTop" {
// CHECK-NEXT:    firrtl.module @ChipTop() {
// CHECK-NEXT:      %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
// CHECK-NEXT:    }
// CHECK-NEXT:  }

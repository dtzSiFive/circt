// RUN: circt-opt %s -export-verilog --split-input-file | FileCheck %s

module {
hw.design {
// CHECK-LABEL: Generated
// CHECK-NEXT: module Foo(
hw.module @Foo(%a: i1 loc("")) -> () {
  hw.output
}
}
}

// -----

module attributes {circt.loweringOptions = "omitVersionComment"}{
hw.design {
// CHECK-NOT: Generated
// CHECK-LABEL: module Bar(
hw.module @Bar() -> () {
  hw.output
}
}
}

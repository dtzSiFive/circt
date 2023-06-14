// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl.module(vectorization)))' %s | FileCheck %s

firrtl.circuit "ElementWise" {
// CHECK-LABEL: @ElementWise
firrtl.module @ElementWise(in %a: !firrtl.vector<uint<1>, 2>, in %b: !firrtl.vector<uint<1>, 2>, out %c_0: !firrtl.vector<uint<1>, 2>, out %c_1: !firrtl.vector<uint<1>, 2>, out %c_2: !firrtl.vector<uint<1>, 2>) {
  // CHECK-NEXT: %0 = elementwise_or %a, %b : (!firrtl.vector<uint<1>, 2>, !firrtl.vector<uint<1>, 2>) -> !firrtl.vector<uint<1>, 2>
  // CHECK-NEXT: strictconnect %c_0, %0 : !firrtl.vector<uint<1>, 2>
  // CHECK-NEXT: %1 = elementwise_and %a, %b : (!firrtl.vector<uint<1>, 2>, !firrtl.vector<uint<1>, 2>) -> !firrtl.vector<uint<1>, 2>
  // CHECK-NEXT: strictconnect %c_1, %1 : !firrtl.vector<uint<1>, 2>
  // CHECK-NEXT: %2 = elementwise_xor %a, %b : (!firrtl.vector<uint<1>, 2>, !firrtl.vector<uint<1>, 2>) -> !firrtl.vector<uint<1>, 2>
  // CHECK-NEXT: strictconnect %c_2, %2 : !firrtl.vector<uint<1>, 2>
  %0 = subindex %b[1] : !firrtl.vector<uint<1>, 2>
  %1 = subindex %a[1] : !firrtl.vector<uint<1>, 2>
  %2 = subindex %b[0] : !firrtl.vector<uint<1>, 2>
  %3 = subindex %a[0] : !firrtl.vector<uint<1>, 2>
  %4 = or %3, %2 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
  %5 = or %1, %0 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
  %6 = vectorcreate %4, %5 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.vector<uint<1>, 2>
  strictconnect %c_0, %6 : !firrtl.vector<uint<1>, 2>
  %7 = and %3, %2 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
  %8 = and %1, %0 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
  %9 = vectorcreate %7, %8 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.vector<uint<1>, 2>
  strictconnect %c_1, %9 : !firrtl.vector<uint<1>, 2>
  %10 = xor %3, %2 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
  %11 = xor %1, %0 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
  %12 = vectorcreate %10, %11 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.vector<uint<1>, 2>
  strictconnect %c_2, %12 : !firrtl.vector<uint<1>, 2>
}
}


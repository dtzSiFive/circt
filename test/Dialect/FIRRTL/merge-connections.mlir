// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl.module(merge-connections)))' %s | FileCheck %s --check-prefixes=CHECK,COMMON
// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl.module(merge-connections{aggressive-merging=true})))' %s | FileCheck %s --check-prefixes=AGGRESSIVE,COMMON

firrtl.circuit "Test"   {
  // circuit Test :
  //   module Test :
  //     input a : {c: {clock: Clock, valid:UInt<1>}[2]}
  //     output b : {c: {clock: Clock, valid:UInt<1>}[2]}
  //     b <= a
  // COMMON-LABEL: module @Test(
  // COMMON-NEXT:    strictconnect %b, %a
  // COMMON-NEXT:  }
  module @Test(in %a: !firrtl.vector<bundle<clock: clock, valid: uint<1>>, 2>, out %b: !firrtl.vector<bundle<clock: clock, valid: uint<1>>, 2>) {
     %0 = subindex %a[0] : !firrtl.vector<bundle<clock: clock, valid: uint<1>>, 2>
     %1 = subindex %b[0] : !firrtl.vector<bundle<clock: clock, valid: uint<1>>, 2>
     %2 = subfield %0[clock] : !firrtl.bundle<clock: clock, valid: uint<1>>
     %3 = subfield %1[clock] : !firrtl.bundle<clock: clock, valid: uint<1>>
     strictconnect %3, %2 : !firrtl.clock
     %4 = subfield %0[valid] : !firrtl.bundle<clock: clock, valid: uint<1>>
     %5 = subfield %1[valid] : !firrtl.bundle<clock: clock, valid: uint<1>>
     strictconnect %5, %4 : !firrtl.uint<1>
     %6 = subindex %a[1] : !firrtl.vector<bundle<clock: clock, valid: uint<1>>, 2>
     %7 = subindex %b[1] : !firrtl.vector<bundle<clock: clock, valid: uint<1>>, 2>
     %8 = subfield %6[clock] : !firrtl.bundle<clock: clock, valid: uint<1>>
     %9 = subfield %7[clock] : !firrtl.bundle<clock: clock, valid: uint<1>>
     strictconnect %9, %8 : !firrtl.clock
     %10 = subfield %6[valid] : !firrtl.bundle<clock: clock, valid: uint<1>>
     %11 = subfield %7[valid] : !firrtl.bundle<clock: clock, valid: uint<1>>
     strictconnect %11, %10 : !firrtl.uint<1>
  }

  // circuit Bar :
  //   module Bar :
  //     output a : {b: UInt<1>, c:UInt<1>}
  //     a.b <= UInt<1>(0)
  //     a.c <= UInt<1>(1)
  // COMMON-LABEL: module @Constant(
  // COMMON-NEXT:    %0 = aggregateconstant [0 : ui1, 1 : ui1]
  // COMMON-NEXT:    strictconnect %a, %0
  // COMMON-NEXT:  }
  module @Constant(out %a: !firrtl.bundle<b: uint<1>, c: uint<1>>) {
    %c0_ui1 = constant 0 : !firrtl.uint<1>
    %c1_ui1 = constant 1 : !firrtl.uint<1>
    %0 = subfield %a[b] : !firrtl.bundle<b: uint<1>, c: uint<1>>
    %1 = subfield %a[c] : !firrtl.bundle<b: uint<1>, c: uint<1>>
    strictconnect %0, %c0_ui1 : !firrtl.uint<1>
    strictconnect %1, %c1_ui1 : !firrtl.uint<1>
  }

  // AGGRESSIVE-LABEL:  module @ConcatToVector(
  // AGGRESSIVE-NEXT:     %0 = vectorcreate %s1, %s2 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.vector<uint<1>, 2>
  // AGGRESSIVE-NEXT:     strictconnect %sink, %0
  // AGGRESSIVE-NEXT:   }
  // CHECK-LABEL:       module @ConcatToVector(
  // CHECK-NEXT:          %0 = subindex %sink[1]
  // CHECK-NEXT:          %1 = subindex %sink[0]
  // CHECK-NEXT:          strictconnect %1, %s1
  // CHECK-NEXT:          strictconnect %0, %s2
  // CHECK-NEXT:        }

  module @ConcatToVector(in %s1: !firrtl.uint<1>, in %s2: !firrtl.uint<1>, out %sink: !firrtl.vector<uint<1>, 2>) {
    %0 = subindex %sink[1] : !firrtl.vector<uint<1>, 2>
    %1 = subindex %sink[0] : !firrtl.vector<uint<1>, 2>
    strictconnect %1, %s1 : !firrtl.uint<1>
    strictconnect %0, %s2 : !firrtl.uint<1>
  }

  // Check that we don't use %s1 as a source value.
  // AGGRESSIVE-LABEL:   module @FailedToUseAggregate(
  // AGGRESSIVE-NEXT:    %0 = subindex %s1[0]
  // AGGRESSIVE-NEXT:    %1 = vectorcreate %0, %s2
  // AGGRESSIVE-NEXT:    strictconnect %sink, %1
  // AGGRESSIVE-NEXT:   }
  // CHECK-LABEL:       module @FailedToUseAggregate(
  // CHECK-NEXT:         %0 = subindex %sink[1]
  // CHECK-NEXT:         %1 = subindex %s1[0]
  // CHECK-NEXT:         %2 = subindex %sink[0]
  // CHECK-NEXT:         strictconnect %2, %1
  // CHECK-NEXT:         strictconnect %0, %s2
  // CHECK-NEXT:        }
  module @FailedToUseAggregate(in %s1: !firrtl.vector<uint<1>, 2>, in %s2: !firrtl.uint<1>, out %sink: !firrtl.vector<uint<1>, 2>) {
    %0 = subindex %sink[1] : !firrtl.vector<uint<1>, 2>
    %1 = subindex %s1[0] : !firrtl.vector<uint<1>, 2>
    %2 = subindex %sink[0] : !firrtl.vector<uint<1>, 2>
    strictconnect %2, %1 : !firrtl.uint<1>
    strictconnect %0, %s2 : !firrtl.uint<1>
  }
}


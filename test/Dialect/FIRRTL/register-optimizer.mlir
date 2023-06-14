// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl.module(firrtl-register-optimizer)))' %s | FileCheck %s

firrtl.circuit "invalidReg"   {
  // CHECK-LABEL: @invalidReg
  module @invalidReg(in %clock: !firrtl.clock, out %a: !firrtl.uint<1>) {
    %foobar = reg %clock  : !firrtl.clock, !firrtl.uint<1>
    strictconnect %foobar, %foobar : !firrtl.uint<1>
    //CHECK-NOT: connect %foobar, %foobar
    //CHECK: %[[inv:.*]] = invalidvalue
    //CHECK: strictconnect %a, %[[inv]]
    strictconnect %a, %foobar : !firrtl.uint<1>
  }

  // CHECK-LABEL: @constantRegWrite
  module @constantRegWrite(in %clock: !firrtl.clock, out %a: !firrtl.uint<1>) {
    %c = constant 0 : !firrtl.uint<1>
    %foobar = reg %clock  : !firrtl.clock, !firrtl.uint<1>
    strictconnect %foobar, %c : !firrtl.uint<1>
    //CHECK-NOT: connect %foobar, %c
    //CHECK: %[[const:.*]] = constant
    //CHECK: strictconnect %a, %[[const]]
    strictconnect %a, %foobar : !firrtl.uint<1>
  }

  // CHECK-LABEL: @constantRegWriteDom
  module @constantRegWriteDom(in %clock: !firrtl.clock, out %a: !firrtl.uint<1>) {
    %foobar = reg %clock  : !firrtl.clock, !firrtl.uint<1>
    //CHECK-NOT: connect %foobar, %c
    //CHECK: %[[const:.*]] = constant
    //CHECK: strictconnect %a, %[[const]]
    strictconnect %a, %foobar : !firrtl.uint<1>
    %c = constant 0 : !firrtl.uint<1>
    strictconnect %foobar, %c : !firrtl.uint<1>
  }

  // CHECK-LABEL: @constantRegResetWrite
  module @constantRegResetWrite(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, out %a: !firrtl.uint<1>) {
    %c = constant 0 : !firrtl.uint<1>
    %foobar = regreset %clock, %reset, %c  : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>
    strictconnect %foobar, %c : !firrtl.uint<1>
    //CHECK-NOT: connect %foobar, %c
    //CHECK: %[[const:.*]] = constant
    //CHECK: strictconnect %a, %[[const]]
    strictconnect %a, %foobar : !firrtl.uint<1>
  }

  // CHECK-LABEL: @constantRegResetWriteSelf
  module @constantRegResetWriteSelf(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, out %a: !firrtl.uint<1>) {
    %c = constant 0 : !firrtl.uint<1>
    %foobar = regreset %clock, %reset, %c  : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>
    strictconnect %foobar, %foobar : !firrtl.uint<1>
    //CHECK-NOT: connect %foobar, %c
    //CHECK: %[[const:.*]] = constant
    //CHECK: strictconnect %a, %[[const]]
    strictconnect %a, %foobar : !firrtl.uint<1>
  }

  // CHECK-LABEL: @movedFromIMCP
  module @movedFromIMCP(
        in %clock: !firrtl.clock,
        in %reset: !firrtl.uint<1>,
        out %result6: !firrtl.uint<2>,
        out %result7: !firrtl.uint<4>) {
    %c0_ui1 = constant 0 : !firrtl.uint<1>
    %c0_ui2 = constant 0 : !firrtl.uint<2>
    %c0_ui4 = constant 0 : !firrtl.uint<4>
    %c1_ui1 = constant 1 : !firrtl.uint<1>

    // regreset
    %regreset = regreset %clock, %reset, %c0_ui2 : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<2>, !firrtl.uint<2>

    strictconnect %regreset, %c0_ui2 : !firrtl.uint<2>

    // CHECK: strictconnect %result6, %c0_ui2
    strictconnect %result6, %regreset: !firrtl.uint<2>

    // reg
    %reg = reg %clock  : !firrtl.clock, !firrtl.uint<4>
    strictconnect %reg, %c0_ui4 : !firrtl.uint<4>
    // CHECK: strictconnect %result7, %c0_ui4
    strictconnect %result7, %reg: !firrtl.uint<4>
  }

  // CHECK-LABEL: RegResetImplicitExtOrTrunc
  module @RegResetImplicitExtOrTrunc(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, out %out: !firrtl.uint<4>) {
    // CHECK: regreset
    %c0_ui3 = constant 0 : !firrtl.uint<3>
    %r = regreset %clock, %reset, %c0_ui3 : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<3>, !firrtl.uint<2>
    %0 = cat %r, %r : (!firrtl.uint<2>, !firrtl.uint<2>) -> !firrtl.uint<4>
    strictconnect %r, %r : !firrtl.uint<2>
    strictconnect %out, %0 : !firrtl.uint<4>
  }
}

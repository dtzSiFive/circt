// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl-imconstprop), canonicalize{top-down region-simplify}, circuit(firrtl.module(firrtl-register-optimizer)))'  %s | FileCheck %s
// github.com/chipsalliance/firrtl: test/scala/firrtlTests/ConstantPropagationTests.scala

//propagate constant inputs  
firrtl.circuit "ConstInput"   {
  module @ConstInput(in %x: !firrtl.uint<1>, out %z: !firrtl.uint<1>) {
    %c1_ui1 = constant 1 : !firrtl.uint<1>
    %c_in0, %c_in1, %c_out = instance c @Child(in in0: !firrtl.uint<1>, in in1: !firrtl.uint<1>, out out: !firrtl.uint<1>)
    connect %c_in0, %x : !firrtl.uint<1>, !firrtl.uint<1>
    connect %c_in1, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    connect %z, %c_out : !firrtl.uint<1>, !firrtl.uint<1>
  }
  // CHECK-LABEL: module private @Child
  module private @Child(in %in0: !firrtl.uint<1>, in %in1: !firrtl.uint<1>, out %out: !firrtl.uint<1>) {
    %0 = and %in0, %in1 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    // CHECK: strictconnect %out, %in0 :
    connect %out, %0 : !firrtl.uint<1>, !firrtl.uint<1>
  }
}

//propagate constant inputs ONLY if ALL instance inputs get the same value
firrtl.circuit "InstanceInput"   {
  // CHECK-LABEL: module private @Bottom1
  module private @Bottom1(in %in: !firrtl.uint<1>, out %out: !firrtl.uint<1>) {
      // CHECK: %c1_ui1 = constant 1
      // CHECK: strictconnect %out, %c1_ui1
    connect %out, %in : !firrtl.uint<1>, !firrtl.uint<1>
  }
  // CHECK-LABEL: module private @Child1
  module private @Child1(out %out: !firrtl.uint<1>) {
    %c1_ui = constant 1 : !firrtl.uint
    %b0_in, %b0_out = instance b0 @Bottom1(in in: !firrtl.uint<1>, out out: !firrtl.uint<1>)
    connect %b0_in, %c1_ui : !firrtl.uint<1>, !firrtl.uint
    // CHECK: %[[C1:.+]] = constant 1 :
    // CHECK: strictconnect %out, %[[C1]]
    connect %out, %b0_out : !firrtl.uint<1>, !firrtl.uint<1>
  }
  // CHECK-LABEL:  module @InstanceInput
  module @InstanceInput(in %x: !firrtl.uint<1>, out %z: !firrtl.uint<1>) {
    %c1_ui = constant 1 : !firrtl.uint
    %c_out = instance c @Child1(out out: !firrtl.uint<1>)
    %b0_in, %b0_out = instance b0  @Bottom1(in in: !firrtl.uint<1>, out out: !firrtl.uint<1>)
    connect %b0_in, %c1_ui : !firrtl.uint<1>, !firrtl.uint
    %b1_in, %b1_out = instance b1  @Bottom1(in in: !firrtl.uint<1>, out out: !firrtl.uint<1>)
    connect %b1_in, %c1_ui : !firrtl.uint<1>, !firrtl.uint
    %0 = and %b0_out, %b1_out : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    %1 = and %0, %c_out : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    // CHECK: %[[C0:.+]] = constant 1 : !firrtl.uint<1>
    // CHECK: strictconnect %z, %[[C0]] : !firrtl.uint<1>
    connect %z, %1 : !firrtl.uint<1>, !firrtl.uint<1>
  }
}

//propagate constant inputs ONLY if ALL instance inputs get the same value
firrtl.circuit "InstanceInput2"   {
  // CHECK-LABEL: module private @Bottom2
  module private @Bottom2(in %in: !firrtl.uint<1>, out %out: !firrtl.uint<1>) {
    // CHECK: strictconnect %out, %in 
    connect %out, %in : !firrtl.uint<1>, !firrtl.uint<1>
  }
 // CHECK-LABEL:  module private @Child2
  module private @Child2(out %out: !firrtl.uint<1>) {
    %c1_ui = constant 1 : !firrtl.uint
    %b0_in, %b0_out = instance b0 @Bottom2(in in: !firrtl.uint<1>, out out: !firrtl.uint<1>)
    connect %b0_in, %c1_ui : !firrtl.uint<1>, !firrtl.uint
    // CHECK: strictconnect %out, %b0_out
    connect %out, %b0_out : !firrtl.uint<1>, !firrtl.uint<1>
  }
 // CHECK-LABEL:  module @InstanceInput2
  module @InstanceInput2(in %x: !firrtl.uint<1>, out %z: !firrtl.uint<1>) {
    %c1_ui = constant 1 : !firrtl.uint
    %c_out = instance c @Child2(out out: !firrtl.uint<1>)
    %b0_in, %b0_out = instance b0 @Bottom2(in in: !firrtl.uint<1>, out out: !firrtl.uint<1>)
    connect %b0_in, %x : !firrtl.uint<1>, !firrtl.uint<1>
    %b1_in, %b1_out = instance b1 @Bottom2(in in: !firrtl.uint<1>, out out: !firrtl.uint<1>)
    connect %b1_in, %c1_ui : !firrtl.uint<1>, !firrtl.uint
    %0 = and %b0_out, %b1_out : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    %1 = and %0, %c_out : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
   // CHECK:  strictconnect %z, %1
    connect %z, %1 : !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// ConstProp should work across wires
firrtl.circuit "acrossWire"   {
  // CHECK-LABEL: module @acrossWire
  module @acrossWire(in %x: !firrtl.uint<1>, out %y: !firrtl.uint<1>) {
    %_z = wire droppable_name : !firrtl.uint<1>
    connect %y, %_z : !firrtl.uint<1>, !firrtl.uint<1>
    %c0_ui1 = constant 0 : !firrtl.uint<1>
    %0 = mux(%x, %c0_ui1, %c0_ui1) : (!firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    connect %_z, %0 : !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK: %[[C2:.+]] = constant 0 : !firrtl.uint<1>
    // CHECK-NEXT: strictconnect %y, %[[C2]] : !firrtl.uint<1>
  }
}

//"ConstProp" should "propagate constant outputs"
firrtl.circuit "constOutput"   {
  module private @constOutChild(out %out: !firrtl.uint<1>) {
    %c0_ui1 = constant 0 : !firrtl.uint<1>
    connect %out, %c0_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
  }
  module @constOutput(in %x: !firrtl.uint<1>, out %z: !firrtl.uint<1>) {
    %c_out = instance c @constOutChild(out out: !firrtl.uint<1>)
    %0 = and %x, %c_out : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    connect %z, %0 : !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK: %[[C3_0:.+]] = constant 0 : !firrtl.uint<1>
    // CHECK: %[[C3:.+]] = constant 0 : !firrtl.uint<1>
    // CHECK: strictconnect %z, %[[C3:.+]] : !firrtl.uint<1>
  }
}

// Optimizing this mux gives: z <= pad(UInt<2>(0), 4)
// Thus this checks that we then optimize that pad
//"ConstProp" should "optimize nested Expressions" in {
firrtl.circuit "optiMux"   {
  // CHECK-LABEL: module @optiMux
  module @optiMux(out %z: !firrtl.uint<4>) {
    %c1_ui = constant 1 : !firrtl.uint
    %c0_ui2 = constant 0 : !firrtl.uint<2>
    %c0_ui4 = constant 0 : !firrtl.uint<4>
    %0 = mux(%c1_ui, %c0_ui2, %c0_ui4) : (!firrtl.uint, !firrtl.uint<2>, !firrtl.uint<4>) -> !firrtl.uint<4>
    // CHECK: %[[C4:.+]] = constant 0 :
    // CHECK: strictconnect %z, %[[C4]]
    connect %z, %0 : !firrtl.uint<4>, !firrtl.uint<4>
  }
}

firrtl.circuit "divFold"   {
  // CHECK-LABEL: module @divFold
  module @divFold(in %a: !firrtl.uint<8>, out %b: !firrtl.uint<8>) {
    %0 = div %a, %a : (!firrtl.uint<8>, !firrtl.uint<8>) -> !firrtl.uint<8>
    connect %b, %0 : !firrtl.uint<8>, !firrtl.uint<8>
    // CHECK: %[[C5:.+]] = constant 1 : !firrtl.uint<8>
    // CHECK: strictconnect %b, %[[C5]] : !firrtl.uint<8>
  }
}

//"pad constant connections to wires when propagating"
firrtl.circuit "padConstWire"   {
  // CHECK-LABEL: module @padConstWire
  module @padConstWire(out %z: !firrtl.uint<16>) {
    %_w_a = wire droppable_name  : !firrtl.uint<8>
    %_w_b = wire droppable_name : !firrtl.uint<8>
    %c3_ui2 = constant 3 : !firrtl.uint<2>
    connect %_w_a, %c3_ui2 : !firrtl.uint<8>, !firrtl.uint<2>
    connect %_w_b, %c3_ui2 : !firrtl.uint<8>, !firrtl.uint<2>
    %0 = cat %_w_a, %_w_b : (!firrtl.uint<8>, !firrtl.uint<8>) -> !firrtl.uint<16>
    connect %z, %0 : !firrtl.uint<16>, !firrtl.uint<16>
    // CHECK: %[[C6:.+]] = constant 771 : !firrtl.uint<16>
    // CHECK-NEXT: strictconnect %z, %[[C6]] : !firrtl.uint<16>
  }
}

// "pad constant connections to registers when propagating"
firrtl.circuit "padConstReg"   {
  // CHECK-LABEL: module @padConstReg
  module @padConstReg(in %clock: !firrtl.clock, out %z: !firrtl.uint<16>) {
    %r_a = reg droppable_name %clock  :  !firrtl.clock, !firrtl.uint<8>
    %r_b = reg droppable_name %clock  :  !firrtl.clock, !firrtl.uint<8>
    %c3_ui2 = constant 3 : !firrtl.uint<2>
    connect %r_a, %c3_ui2 : !firrtl.uint<8>, !firrtl.uint<2>
    connect %r_b, %c3_ui2 : !firrtl.uint<8>, !firrtl.uint<2>
    %0 = cat %r_a, %r_b : (!firrtl.uint<8>, !firrtl.uint<8>) -> !firrtl.uint<16>
    connect %z, %0 : !firrtl.uint<16>, !firrtl.uint<16>
    // CHECK: %[[C6:.+]] = constant 771 : !firrtl.uint<16>
    // CHECK-NEXT: strictconnect %z, %[[C6]] : !firrtl.uint<16>
  }
}

// should "pad constant connections to outputs when propagating"
firrtl.circuit "padConstOut"   {
  module private @padConstOutChild(out %x: !firrtl.uint<8>) {
    %c3_ui2 = constant 3 : !firrtl.uint<2>
    connect %x, %c3_ui2 : !firrtl.uint<8>, !firrtl.uint<2>
  }
  // CHECK-LABEL: module @padConstOut
  module @padConstOut(out %z: !firrtl.uint<16>) {
    %c_x = instance c @padConstOutChild(out x: !firrtl.uint<8>)
    %c3_ui2 = constant 3 : !firrtl.uint<2>
    %0 = cat %c3_ui2, %c_x : (!firrtl.uint<2>, !firrtl.uint<8>) -> !firrtl.uint<10>
    // CHECK: %[[C8:.+]] = constant 771 : !firrtl.uint<16>
    // CHECK: strictconnect %z, %[[C8]] : !firrtl.uint<16>
    connect %z, %0 : !firrtl.uint<16>, !firrtl.uint<10>
  }
}

// "pad constant connections to submodule inputs when propagating"
firrtl.circuit "padConstIn"   {
  // CHECK-LABEL: module private @padConstInChild
  module private @padConstInChild(in %x: !firrtl.uint<8>, out %y: !firrtl.uint<16>) {
    %c3_ui2 = constant 3 : !firrtl.uint<2>
    %0 = cat %c3_ui2, %x : (!firrtl.uint<2>, !firrtl.uint<8>) -> !firrtl.uint<10>
    // CHECK: %[[C9:.+]] = constant 771 : !firrtl.uint<16>
    // CHECK: strictconnect %y, %[[C9]] : !firrtl.uint<16>
    connect %y, %0 : !firrtl.uint<16>, !firrtl.uint<10>
  }
  // CHECK-LABEL: module @padConstIn
  module @padConstIn(out %z: !firrtl.uint<16>) {
    %c_x, %c_y = instance c @padConstInChild(in x: !firrtl.uint<8>, out y: !firrtl.uint<16>)
    %c3_ui2 = constant 3 : !firrtl.uint<2>
    connect %c_x, %c3_ui2 : !firrtl.uint<8>, !firrtl.uint<2>
    connect %z, %c_y : !firrtl.uint<16>, !firrtl.uint<16>
    // CHECK: %[[C10:.+]] = constant 771 : !firrtl.uint<16>
    // CHECK: strictconnect %z, %[[C10]] : !firrtl.uint<16>
  }
}

//  "remove pads if the width is <= the width of the argument"
firrtl.circuit "removePad"   {
  // CHECK-LABEL: module @removePad
  module @removePad(in %x: !firrtl.uint<8>, out %z: !firrtl.uint<8>) {
    %0 = pad %x, 6 : (!firrtl.uint<8>) -> !firrtl.uint<8>
    // CHECK: strictconnect %z, %x : !firrtl.uint<8>
    connect %z, %0 : !firrtl.uint<8>, !firrtl.uint<8>
  }
}

//"Registers async reset and a constant connection" should "NOT be removed
firrtl.circuit "asyncReset"   {
  // CHECK-LABEL: module @asyncReset
  module @asyncReset(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset, in %en: !firrtl.uint<1>, out %z: !firrtl.uint<8>) {
    %c11_ui4 = constant 11 : !firrtl.uint<4>
    %r = regreset %clock, %reset, %c11_ui4  : !firrtl.clock, !firrtl.asyncreset, !firrtl.uint<4>, !firrtl.uint<8>
    %c0_ui4 = constant 0 : !firrtl.uint<4>
    %0 = mux(%en, %c0_ui4, %r) : (!firrtl.uint<1>, !firrtl.uint<4>, !firrtl.uint<8>) -> !firrtl.uint<8>
    connect %r, %0 : !firrtl.uint<8>, !firrtl.uint<8>
    connect %z, %r : !firrtl.uint<8>, !firrtl.uint<8>
    // CHECK: strictconnect %r, %0 : !firrtl.uint<8>
    // CHECK: strictconnect %z, %r : !firrtl.uint<8>
  }
}

//"Registers with ONLY constant connection" should "be replaced with that constant"
firrtl.circuit "constReg2"   {
  // CHECK-LABEL: module @constReg2
  module @constReg2(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, out %z: !firrtl.sint<8>) {
    %r = reg %clock  :  !firrtl.clock, !firrtl.sint<8>
    %c-5_si4 = constant -5 : !firrtl.sint<4>
    connect %r, %c-5_si4 : !firrtl.sint<8>, !firrtl.sint<4>
    connect %z, %r : !firrtl.sint<8>, !firrtl.sint<8>
    // CHECK: %[[C12:.+]] = constant -5 : !firrtl.sint<8>
    // CHECK: strictconnect %z, %[[C12]] : !firrtl.sint<8>
  }
}

//"propagation of signed expressions" should "have the correct signs"
firrtl.circuit "SignTester"   {
  // CHECK-LABEL: module @SignTester
  module @SignTester(out %ref: !firrtl.sint<3>) {
    %c0_ui1 = constant 0 : !firrtl.uint<1>
    %c0_si3 = constant 0 : !firrtl.sint<3>
    %c3_ui2 = constant 3 : !firrtl.uint<2>
    %0 = neg %c3_ui2 : (!firrtl.uint<2>) -> !firrtl.sint<3>
    %1 = mux(%c0_ui1, %c0_si3, %0) : (!firrtl.uint<1>, !firrtl.sint<3>, !firrtl.sint<3>) -> !firrtl.sint<3>
    connect %ref, %1 : !firrtl.sint<3>, !firrtl.sint<3>
    // CHECK:  %[[C14:.+]] = constant -3 : !firrtl.sint<3>
    // CHECK:  strictconnect %ref, %[[C14]] : !firrtl.sint<3>
  }
}

//"addition of negative literals" should "be propagated"
firrtl.circuit "AddTester"   {
  // CHECK-LABEL: module @AddTester
  module @AddTester(out %ref: !firrtl.sint<2>) {
    %c-1_si1 = constant -1 : !firrtl.sint<1>
    %0 = add %c-1_si1, %c-1_si1 : (!firrtl.sint<1>, !firrtl.sint<1>) -> !firrtl.sint<2>
    connect %ref, %0 : !firrtl.sint<2>, !firrtl.sint<2>
    // CHECK:  %[[C15:.+]] = constant -2 : !firrtl.sint<2>
    // CHECK:  strictconnect %ref, %[[C15]]
  }
}

//"reduction of literals" should "be propagated"
firrtl.circuit "ConstPropReductionTester"   {
  // CHECK-LABEL: module @ConstPropReductionTester
  module @ConstPropReductionTester(out %out1: !firrtl.uint<1>, out %out2: !firrtl.uint<1>, out %out3: !firrtl.uint<1>) {
    %c-1_si2 = constant -1 : !firrtl.sint<2>
    %0 = xorr %c-1_si2 : (!firrtl.sint<2>) -> !firrtl.uint<1>
    connect %out1, %0 : !firrtl.uint<1>, !firrtl.uint<1>
    %1 = andr %c-1_si2 : (!firrtl.sint<2>) -> !firrtl.uint<1>
    connect %out2, %1 : !firrtl.uint<1>, !firrtl.uint<1>
    %2 = orr %c-1_si2 : (!firrtl.sint<2>) -> !firrtl.uint<1>
    connect %out3, %2 : !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK:  %[[C16:.+]] = constant 0
    // CHECK:  %[[C17:.+]] = constant 1
    // CHECK:  strictconnect %out1, %[[C16]]
    // CHECK:  strictconnect %out2, %[[C17]]
    // CHECK:  strictconnect %out3, %[[C17]]
  }
}

firrtl.circuit "TailTester"   {
  // CHECK-LABEL: module @TailTester
  module @TailTester(out %out: !firrtl.uint<1>) {
    %c0_ui1 = constant 0 : !firrtl.uint<1>
    %c23_ui5 = constant 23 : !firrtl.uint<5>
    %0 = add %c0_ui1, %c23_ui5 : (!firrtl.uint<1>, !firrtl.uint<5>) -> !firrtl.uint<6>
    %_temp = node droppable_name %0  : !firrtl.uint<6>
    %1 = head %_temp, 3 : (!firrtl.uint<6>) -> !firrtl.uint<3>
    %_head_temp = node droppable_name %1  : !firrtl.uint<3>
    %2 = tail %_head_temp, 2 : (!firrtl.uint<3>) -> !firrtl.uint<1>
    connect %out, %2 : !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK:  %[[C18:.+]] = constant 0
    // CHECK:  strictconnect %out, %[[C18]]
  }
}

//"tail of constants" should "be propagated"
firrtl.circuit "TailTester2"   {
  // CHECK-LABEL: module @TailTester2
  module @TailTester2(out %out: !firrtl.uint<1>) {
    %c0_ui1 = constant 0 : !firrtl.uint<1>
    %c23_ui5 = constant 23 : !firrtl.uint<5>
    %0 = add %c0_ui1, %c23_ui5 : (!firrtl.uint<1>, !firrtl.uint<5>) -> !firrtl.uint<6>
    %_temp = node droppable_name %0  : !firrtl.uint<6>
    %1 = tail %_temp, 1 : (!firrtl.uint<6>) -> !firrtl.uint<5>
    %_tail_temp = node droppable_name %1  : !firrtl.uint<5>
    %2 = tail %_tail_temp, 4 : (!firrtl.uint<5>) -> !firrtl.uint<1>
    connect %out, %2 : !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK:  %[[C21:.+]] = constant 1
    // CHECK:  strictconnect %out, %[[C21]]
  }
}

//"addition by zero width wires" should "have the correct widths"
firrtl.circuit "ZeroWidthAdd"   {
  // CHECK-LABEL: module @ZeroWidthAdd
  module @ZeroWidthAdd(in %x: !firrtl.uint<0>, out %y: !firrtl.uint<7>) {
    %c0_ui9 = constant 0 : !firrtl.uint<9>
    %0 = add %x, %c0_ui9 : (!firrtl.uint<0>, !firrtl.uint<9>) -> !firrtl.uint<10>
    %_temp = node droppable_name %0  : !firrtl.uint<10>
    %1 = cat %_temp, %_temp : (!firrtl.uint<10>, !firrtl.uint<10>) -> !firrtl.uint<20>
    %2 = tail %1, 13 : (!firrtl.uint<20>) -> !firrtl.uint<7>
    connect %y, %2 : !firrtl.uint<7>, !firrtl.uint<7>
    // CHECK:  %[[C20:.+]] = constant 0
    // CHECK:  strictconnect %y, %[[C20]]
  }
}

//"Registers with constant reset and connection to the same constant" should "be replaced with that constant"
firrtl.circuit "regConstReset"   {
  // CHECK-LABEL: module @regConstReset
  module @regConstReset(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %cond: !firrtl.uint<1>, out %z: !firrtl.uint<8>) {
    %c11_ui8 = constant 11 : !firrtl.uint<8>
    %r = regreset %clock, %reset, %c11_ui8  : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<8>, !firrtl.uint<8>
    %0 = mux(%cond, %c11_ui8, %r) : (!firrtl.uint<1>, !firrtl.uint<8>, !firrtl.uint<8>) -> !firrtl.uint<8>
    connect %r, %0 : !firrtl.uint<8>, !firrtl.uint<8>
    connect %z, %r : !firrtl.uint<8>, !firrtl.uint<8>
    // CHECK: %[[C22:.+]] = constant 11 
    // CHECK: strictconnect %z, %[[C22]]
  }
}

//"Const prop of registers" should "do limited speculative expansion of optimized muxes to absorb bigger cones"
firrtl.circuit "constPropRegMux"   {
  // CHECK-LABEL: module @constPropRegMux
  module @constPropRegMux(in %clock: !firrtl.clock, in %en: !firrtl.uint<1>, out %out: !firrtl.uint<1>) {
  %r1 = reg %clock  : !firrtl.clock, !firrtl.uint<1>
  %r2 = reg %clock  : !firrtl.clock, !firrtl.uint<1>
  %c1_ui1 = constant 1 : !firrtl.uint<1>
  %0 = mux(%en, %c1_ui1, %r1) : (!firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
  connect %r1, %0 : !firrtl.uint<1>, !firrtl.uint<1>
  %c0_ui1 = constant 0 : !firrtl.uint<1>
  %1 = mux(%en, %r2, %c0_ui1) : (!firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
  connect %r2, %1 : !firrtl.uint<1>, !firrtl.uint<1>
  %2 = xor %r1, %r2 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
  connect %out, %2 : !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK: %[[C23:.+]] = constant 1
    // CHECK: strictconnect %out, %[[C23]]
  }
}

// Registers with no reset or connections" should "be replaced with constant zero
firrtl.circuit "uninitSelfReg"   {
  // CHECK-LABEL: module @uninitSelfReg
  module @uninitSelfReg(in %clock: !firrtl.clock, out %z: !firrtl.uint<8>) {
    %r = reg %clock  :  !firrtl.clock, !firrtl.uint<8>
    strictconnect %r, %r : !firrtl.uint<8>
    strictconnect %z, %r : !firrtl.uint<8>
    // CHECK: %invalid_ui8 = invalidvalue : !firrtl.uint<8>
    // CHECK: strictconnect %z, %invalid_ui8 : !firrtl.uint<8>
  }

//"Registers with ONLY constant reset" should "be replaced with that constant" in {
  // CHECK-LABEL: module @constResetReg(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, out %z: !firrtl.uint<8>) {
  module @constResetReg(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, out %z: !firrtl.uint<8>) {
    %c11_ui4 = constant 11 : !firrtl.uint<8>
    %r = regreset %clock, %reset, %c11_ui4  : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<8>, !firrtl.uint<8>
    strictconnect %r, %r : !firrtl.uint<8>
    strictconnect %z, %r : !firrtl.uint<8>
    // CHECK: %[[C11:.+]] = constant 11 : !firrtl.uint<8>
    // CHECK: strictconnect %z, %[[C11]] : !firrtl.uint<8>
  }

//"Registers with identical constant reset and connection" should "be replaced with that constant" in {
  // CHECK-LABEL: module @regSameConstReset
  module @regSameConstReset(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, out %z: !firrtl.uint<8>) {
    %c11_ui4 = constant 11 : !firrtl.uint<8>
    %r = regreset %clock, %reset, %c11_ui4  : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<8>, !firrtl.uint<8>
    strictconnect %r, %c11_ui4 : !firrtl.uint<8>
    strictconnect %z, %r : !firrtl.uint<8>
    // CHECK: %[[C13:.+]] = constant 11 : !firrtl.uint<8>
    // CHECK: strictconnect %z, %[[C13]] : !firrtl.uint<8>
  }
}

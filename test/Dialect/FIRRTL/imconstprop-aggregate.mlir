// RUN: circt-opt -pass-pipeline='firrtl.circuit(firrtl-imconstprop)' --split-input-file  %s | FileCheck %s

firrtl.circuit "VectorPropagation1" {
  // CHECK-LABEL: @VectorPropagation1
  firrtl.module @VectorPropagation1(in %clock: !firrtl.clock, out %b: !firrtl.uint<1>) {
    %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
    %tmp = firrtl.reg %clock  : !firrtl.vector<uint<1>, 2>
    %0 = firrtl.subindex %tmp[0] : !firrtl.vector<uint<1>, 2>
    firrtl.connect %0, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    %1 = firrtl.subindex %tmp[1] : !firrtl.vector<uint<1>, 2>
    firrtl.connect %1, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    %2 = firrtl.xor %0, %1 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    firrtl.connect %b, %2 : !firrtl.uint<1>, !firrtl.uint<1>
    // TODO: Register and connections are not currently erased.
    // CHECK: %tmp = firrtl.reg %clock

    // CHECK:      %[[c0_ui1:.+]] = firrtl.constant 0 : !firrtl.uint<1>
    // CHECK-NEXT: firrtl.connect %b, %[[c0_ui1]] : !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "VectorPropagation2" {
  // CHECK-LABEL: @VectorPropagation2
  firrtl.module @VectorPropagation2(in %clock: !firrtl.clock, out %b1: !firrtl.uint<6>, out %b2: !firrtl.uint<6>, out %b3: !firrtl.uint<6>) {

    // tmp1[0][0] <= 1
    // tmp1[0][1] <= 2
    // tmp1[1][0] <= 4
    // tmp1[1][1] <= 8
    // tmp1[2][0] <= 16
    // tmp1[2][1] <= 32

    // b1 <= tmp[0][0] xor tmp1[1][0] = 5
    // b2 <= tmp[2][1] xor tmp1[0][1] = 34
    // b3 <= tmp[1][1] xor tmp1[2][0] = 24

    %c32_ui6 = firrtl.constant 32 : !firrtl.uint<6>
    %c16_ui6 = firrtl.constant 16 : !firrtl.uint<6>
    %c8_ui6 = firrtl.constant 8 : !firrtl.uint<6>
    %c4_ui6 = firrtl.constant 4 : !firrtl.uint<6>
    %c2_ui6 = firrtl.constant 2 : !firrtl.uint<6>
    %c1_ui6 = firrtl.constant 1 : !firrtl.uint<6>
    %tmp = firrtl.reg %clock  : !firrtl.vector<vector<uint<6>, 2>, 3>
    %0 = firrtl.subindex %tmp[0] : !firrtl.vector<vector<uint<6>, 2>, 3>
    %1 = firrtl.subindex %0[0] : !firrtl.vector<uint<6>, 2>
    firrtl.connect %1, %c1_ui6 : !firrtl.uint<6>, !firrtl.uint<6>
    %2 = firrtl.subindex %0[1] : !firrtl.vector<uint<6>, 2>
    firrtl.connect %2, %c2_ui6 : !firrtl.uint<6>, !firrtl.uint<6>
    %3 = firrtl.subindex %tmp[1] : !firrtl.vector<vector<uint<6>, 2>, 3>
    %4 = firrtl.subindex %3[0] : !firrtl.vector<uint<6>, 2>
    firrtl.connect %4, %c4_ui6 : !firrtl.uint<6>, !firrtl.uint<6>
    %5 = firrtl.subindex %3[1] : !firrtl.vector<uint<6>, 2>
    firrtl.connect %5, %c8_ui6 : !firrtl.uint<6>, !firrtl.uint<6>
    %6 = firrtl.subindex %tmp[2] : !firrtl.vector<vector<uint<6>, 2>, 3>
    %7 = firrtl.subindex %6[0] : !firrtl.vector<uint<6>, 2>
    firrtl.connect %7, %c16_ui6 : !firrtl.uint<6>, !firrtl.uint<6>
    %8 = firrtl.subindex %6[1] : !firrtl.vector<uint<6>, 2>
    firrtl.connect %8, %c32_ui6 : !firrtl.uint<6>, !firrtl.uint<6>
    %9 = firrtl.xor %1, %4 : (!firrtl.uint<6>, !firrtl.uint<6>) -> !firrtl.uint<6>
    firrtl.connect %b1, %9 : !firrtl.uint<6>, !firrtl.uint<6>
    %10 = firrtl.xor %8, %2 : (!firrtl.uint<6>, !firrtl.uint<6>) -> !firrtl.uint<6>
    firrtl.connect %b2, %10 : !firrtl.uint<6>, !firrtl.uint<6>
    %11 = firrtl.xor %7, %5 : (!firrtl.uint<6>, !firrtl.uint<6>) -> !firrtl.uint<6>
    firrtl.connect %b3, %11 : !firrtl.uint<6>, !firrtl.uint<6>
    // TODO: Register and connections are not currently erased.
    // CHECK: %tmp = firrtl.reg %clock

    // CHECK:      %c5_ui6 = firrtl.constant 5 : !firrtl.uint<6>
    // CHECK-NEXT: firrtl.connect %b1, %c5_ui6 : !firrtl.uint<6>, !firrtl.uint<6>
    // CHECK-NEXT: %c34_ui6 = firrtl.constant 34 : !firrtl.uint<6>
    // CHECK-NEXT: firrtl.connect %b2, %c34_ui6 : !firrtl.uint<6>, !firrtl.uint<6>
    // CHECK-NEXT: %c24_ui6 = firrtl.constant 24 : !firrtl.uint<6>
    // CHECK-NEXT: firrtl.connect %b3, %c24_ui6 : !firrtl.uint<6>, !firrtl.uint<6>
  }
}

// -----

firrtl.circuit "BundlePropagation1"   {
  // CHECK-LABEL: @BundlePropagation1
  firrtl.module @BundlePropagation1(in %clock: !firrtl.clock, out %result: !firrtl.uint<3>) {
    %tmp = firrtl.reg %clock  : !firrtl.bundle<a: uint<3>, b: uint<3>, c: uint<3>>
    %c1_ui3 = firrtl.constant 1 : !firrtl.uint<3>
    %c2_ui3 = firrtl.constant 2 : !firrtl.uint<3>
    %c4_ui3 = firrtl.constant 4 : !firrtl.uint<3>
    %0 = firrtl.subfield %tmp(0) : (!firrtl.bundle<a: uint<3>, b: uint<3>, c: uint<3>>) -> !firrtl.uint<3>
    %1 = firrtl.subfield %tmp(1) : (!firrtl.bundle<a: uint<3>, b: uint<3>, c: uint<3>>) -> !firrtl.uint<3>
    %2 = firrtl.subfield %tmp(2) : (!firrtl.bundle<a: uint<3>, b: uint<3>, c: uint<3>>) -> !firrtl.uint<3>
    firrtl.connect %0, %c1_ui3 : !firrtl.uint<3>, !firrtl.uint<3>
    firrtl.connect %1, %c2_ui3 : !firrtl.uint<3>, !firrtl.uint<3>
    firrtl.connect %2, %c4_ui3 : !firrtl.uint<3>, !firrtl.uint<3>
    %3 = firrtl.xor %0, %1 : (!firrtl.uint<3>, !firrtl.uint<3>) -> !firrtl.uint<3>
    %4 = firrtl.xor %3, %2 : (!firrtl.uint<3>, !firrtl.uint<3>) -> !firrtl.uint<3>
    firrtl.connect %result, %4 : !firrtl.uint<3>, !firrtl.uint<3>
    // TODO: Register and connections are not currently erased.
    // CHECK: %tmp = firrtl.reg %clock

    // CHECK:      %c7_ui3 = firrtl.constant 7 : !firrtl.uint<3>
    // CHECK-NEXT: firrtl.connect %result, %c7_ui3 : !firrtl.uint<3>, !firrtl.uint<3>
  }
}

// -----

// circuit AggregateAsyncReset:
//   module AggregateAsyncReset:
//     input clock : Clock
//     input reset : UInt<1>
//     output res1: UInt<3>
//     output res2: UInt<3>
//
//     wire init : UInt<3>[2]
//     reg reg : UInt<3>[2] , clock with : ( reset => ( reset , init ) )
//     init[0] <= UInt<3>(0)
//     init[1] <= UInt<3>(2)
//     reg[0] <= UInt<3>(1)
//     reg[1] <= UInt<3>(2)
//     res1 <= reg[0]
//     res2 <= reg[1]

firrtl.circuit "AggregateAsyncReset" {
  // CHECK-LABEL: @AggregateAsyncReset
  firrtl.module @AggregateAsyncReset(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset, out %res1: !firrtl.uint<3>, out %res2: !firrtl.uint<3>) {
    %c0_ui3 = firrtl.constant 0 : !firrtl.uint<3>
    %c2_ui3 = firrtl.constant 2 : !firrtl.uint<3>
    %c1_ui3 = firrtl.constant 1 : !firrtl.uint<3>
    %init = firrtl.wire  : !firrtl.vector<uint<3>, 2>
    %reg = firrtl.regreset %clock, %reset, %init  : !firrtl.asyncreset, !firrtl.vector<uint<3>, 2>, !firrtl.vector<uint<3>, 2>
    %0 = firrtl.subindex %init[0] : !firrtl.vector<uint<3>, 2>
    firrtl.connect %0, %c0_ui3 : !firrtl.uint<3>, !firrtl.uint<3>
    %1 = firrtl.subindex %init[1] : !firrtl.vector<uint<3>, 2>
    firrtl.connect %1, %c2_ui3 : !firrtl.uint<3>, !firrtl.uint<3>
    %2 = firrtl.subindex %reg[0] : !firrtl.vector<uint<3>, 2>
    firrtl.connect %2, %c1_ui3 : !firrtl.uint<3>, !firrtl.uint<3>
    %3 = firrtl.subindex %reg[1] : !firrtl.vector<uint<3>, 2>
    firrtl.connect %3, %c2_ui3 : !firrtl.uint<3>, !firrtl.uint<3>
    firrtl.connect %res1, %2 : !firrtl.uint<3>, !firrtl.uint<3>
    firrtl.connect %res2, %3 : !firrtl.uint<3>, !firrtl.uint<3>
    // Check that %init is not deleted.
    // CHECK: %init = firrtl.wire

    // Check that %res1 is not folded because reg[0] is either 0 or 1.
    // CHECK:      firrtl.connect %res1, %2 : !firrtl.uint<3>, !firrtl.uint<3>
    // CHECK-NEXT: firrtl.connect %res2, %c2_ui3_2 : !firrtl.uint<3>, !firrtl.uint<3>
  }
}

// -----

firrtl.circuit "AggregateRegReset" {
  // CHECK-LABEL: @AggregateRegReset
  firrtl.module @AggregateRegReset(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, out %out: !firrtl.uint<1>) {
    %init = firrtl.wire : !firrtl.vector<uint<1>, 1>
    %0 = firrtl.subindex %init[0] : !firrtl.vector<uint<1>, 1>
    %true = firrtl.constant 1 : !firrtl.uint<1>
    firrtl.connect %0, %true : !firrtl.uint<1>, !firrtl.uint<1>
    %reg = firrtl.regreset %clock, %reset, %init  : !firrtl.uint<1>, !firrtl.vector<uint<1>, 1>, !firrtl.vector<uint<1>, 1>
    %1 = firrtl.subindex %reg[0] : !firrtl.vector<uint<1>, 1>
    firrtl.connect %1, %true : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %out, %1 : !firrtl.uint<1>, !firrtl.uint<1>
    // Check that %init is not deleted.
    // CHECK: %init = firrtl.wire
    // CHECK: firrtl.connect %out, %[[c1:c1_ui1.*]] : !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "DontTouchAggregate" {
  firrtl.module @DontTouchAggregate(in %clock: !firrtl.clock, out %out1: !firrtl.uint<1>, out %out2: !firrtl.uint<1>) {
    // fieldID 1 means the first element. Check that we don't propagate througth it.
    %init = firrtl.wire sym @dntSym: !firrtl.vector<uint<1>, 2>
    %0 = firrtl.subindex %init[0] : !firrtl.vector<uint<1>, 2>
    %1 = firrtl.subindex %init[1] : !firrtl.vector<uint<1>, 2>
    %true = firrtl.constant 1 : !firrtl.uint<1>
    firrtl.connect %0, %true : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %1, %true : !firrtl.uint<1>, !firrtl.uint<1>

    firrtl.connect %out1, %0 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %out2, %1 : !firrtl.uint<1>, !firrtl.uint<1>
    // Make sure that we don't propagate through %init[0].
    // CHECK:      firrtl.connect %0, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK-NEXT: firrtl.connect %1, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK-NEXT: firrtl.connect %out1, %0 : !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK-NEXT: firrtl.connect %out2, %1 : !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// -----
// Following tests are ported from normal imconstprop tests.

firrtl.circuit "OutPortTop" {
  // Check that we don't propagate througth it.
  firrtl.module @OutPortChild(out %out: !firrtl.vector<uint<1>, 2>) attributes {
    portAnnotations = [[]],
    portSyms = ["dntSym"]
  }
  {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %0 = firrtl.subindex %out[0] : !firrtl.vector<uint<1>, 2>
    %1 = firrtl.subindex %out[1] : !firrtl.vector<uint<1>, 2>
    firrtl.connect %0, %c0_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %1, %c0_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
  }
  // CHECK-LABEL: firrtl.module @OutPortTop
  firrtl.module @OutPortTop(out %out1: !firrtl.uint<1>, out %out2: !firrtl.uint<1>) {
    %c_out = firrtl.instance c @OutPortChild(out out: !firrtl.vector<uint<1>, 2>)
    %0 = firrtl.subindex %c_out[0] : !firrtl.vector<uint<1>, 2>
    %1 = firrtl.subindex %c_out[1] : !firrtl.vector<uint<1>, 2>
    firrtl.connect %out1, %0 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %out2, %1 : !firrtl.uint<1>, !firrtl.uint<1>
    // Make sure that we don't propagate through %c_out[0].
    // FIXME: Currently, we are not propagating through %c_out[1] too because
    // we don't look at symbols in the field sensitive way.

    // CHECK:      %0 = firrtl.subindex %c_out[0] : !firrtl.vector<uint<1>, 2>
    // CHECK-NEXT: %1 = firrtl.subindex %c_out[1] : !firrtl.vector<uint<1>, 2>
    // CHECK-NEXT: firrtl.connect %out1, %0 : !firrtl.uint<1>, !firrtl.uint<1>
    // CHECK-NEXT: firrtl.connect %out2, %1 : !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "InputPortTop"  {
  // CHECK-LABEL: firrtl.module @InputPortChild2
  firrtl.module @InputPortChild2(in %in0: !firrtl.bundle<v: uint<1>>, in %in1: !firrtl.bundle<v: uint<1>>, out %out: !firrtl.bundle<v: uint<1>>) {
    %0 = firrtl.subfield %in1(0) : (!firrtl.bundle<v: uint<1>>) -> !firrtl.uint<1>
    %1 = firrtl.subfield %in0(0) : (!firrtl.bundle<v: uint<1>>) -> !firrtl.uint<1>
    %2 = firrtl.subfield %out(0) : (!firrtl.bundle<v: uint<1>>) -> !firrtl.uint<1>
    // CHECK: = firrtl.constant 1
    %3 = firrtl.and %1, %0 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    firrtl.connect %2, %3 : !firrtl.uint<1>, !firrtl.uint<1>
  }
  // CHECK-LABEL: firrtl.module @InputPortChild
  firrtl.module @InputPortChild(in %in0: !firrtl.bundle<v: uint<1>>, in %in1: !firrtl.bundle<v: uint<1>>, out %out: !firrtl.bundle<v: uint<1>>) attributes {
    portAnnotations = [[], [], []], portSyms = ["", "dntSym", ""]
  }
  {
    %0 = firrtl.subfield %in1(0) : (!firrtl.bundle<v: uint<1>>) -> !firrtl.uint<1>
    %1 = firrtl.subfield %in0(0) : (!firrtl.bundle<v: uint<1>>) -> !firrtl.uint<1>
    %2 = firrtl.subfield %out(0) : (!firrtl.bundle<v: uint<1>>) -> !firrtl.uint<1>
    // CHECK-NOT: firrtl.constant
    %3 = firrtl.and %1, %0 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    firrtl.connect %2, %3 : !firrtl.uint<1>, !firrtl.uint<1>
  }

  // CHECK-LABEL: firrtl.module @InputPortTop
  firrtl.module @InputPortTop(in %x: !firrtl.bundle<v: uint<1>>, out %z: !firrtl.bundle<v: uint<1>>, out %z2: !firrtl.bundle<v: uint<1>>) {
    %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
    %0 = firrtl.subfield %z2(0) : (!firrtl.bundle<v: uint<1>>) -> !firrtl.uint<1>
    %1 = firrtl.subfield %x(0) : (!firrtl.bundle<v: uint<1>>) -> !firrtl.uint<1>
    %2 = firrtl.subfield %z(0) : (!firrtl.bundle<v: uint<1>>) -> !firrtl.uint<1>
    %c_in0, %c_in1, %c_out = firrtl.instance c  @InputPortChild(in in0: !firrtl.bundle<v: uint<1>>, in in1: !firrtl.bundle<v: uint<1>>, out out: !firrtl.bundle<v: uint<1>>)
    %3 = firrtl.subfield %c_in1(0) : (!firrtl.bundle<v: uint<1>>) -> !firrtl.uint<1>
    %4 = firrtl.subfield %c_in0(0) : (!firrtl.bundle<v: uint<1>>) -> !firrtl.uint<1>
    %5 = firrtl.subfield %c_out(0) : (!firrtl.bundle<v: uint<1>>) -> !firrtl.uint<1>
    %c2_in0, %c2_in1, %c2_out = firrtl.instance c2  @InputPortChild2(in in0: !firrtl.bundle<v: uint<1>>, in in1: !firrtl.bundle<v: uint<1>>, out out: !firrtl.bundle<v: uint<1>>)
    %6 = firrtl.subfield %c2_in1(0) : (!firrtl.bundle<v: uint<1>>) -> !firrtl.uint<1>
    %7 = firrtl.subfield %c2_in0(0) : (!firrtl.bundle<v: uint<1>>) -> !firrtl.uint<1>
    %8 = firrtl.subfield %c2_out(0) : (!firrtl.bundle<v: uint<1>>) -> !firrtl.uint<1>
    firrtl.connect %2, %5 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %4, %1 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %3, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %0, %8 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %7, %1 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %6, %c1_ui1 : !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// -----

// CHECK-LABK: firrtl.circuit "rhs_sink_output_used_as_wire"
firrtl.circuit "rhs_sink_output_used_as_wire" {
  // CHECK: firrtl.module @Bar
  firrtl.module @Bar(in %a: !firrtl.bundle<v: uint<1>>, in %b: !firrtl.bundle<v: uint<1>>, out %c: !firrtl.bundle<v: uint<1>>, out %d: !firrtl.bundle<v: uint<1>>) {
    %0 = firrtl.subfield %d(0) : (!firrtl.bundle<v: uint<1>>) -> !firrtl.uint<1>
    %1 = firrtl.subfield %a(0) : (!firrtl.bundle<v: uint<1>>) -> !firrtl.uint<1>
    %2 = firrtl.subfield %b(0) : (!firrtl.bundle<v: uint<1>>) -> !firrtl.uint<1>
    %3 = firrtl.subfield %c(0) : (!firrtl.bundle<v: uint<1>>) -> !firrtl.uint<1>
    firrtl.connect %3, %2 : !firrtl.uint<1>, !firrtl.uint<1>
    %_c = firrtl.wire  : !firrtl.bundle<v: uint<1>>
    %4 = firrtl.subfield %_c(0) : (!firrtl.bundle<v: uint<1>>) -> !firrtl.uint<1>
    %5 = firrtl.xor %1, %3 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    // CHECK: firrtl.xor
    firrtl.connect %4, %5 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %0, %4 : !firrtl.uint<1>, !firrtl.uint<1>
  }
  firrtl.module @rhs_sink_output_used_as_wire(in %a: !firrtl.bundle<v: uint<1>>, in %b: !firrtl.bundle<v: uint<1>>, out %c: !firrtl.bundle<v: uint<1>>, out %d: !firrtl.bundle<v: uint<1>>) {
    %bar_a, %bar_b, %bar_c, %bar_d = firrtl.instance bar  @Bar(in a: !firrtl.bundle<v: uint<1>>, in b: !firrtl.bundle<v: uint<1>>, out c: !firrtl.bundle<v: uint<1>>, out d: !firrtl.bundle<v: uint<1>>)
    %0 = firrtl.subfield %a(0) : (!firrtl.bundle<v: uint<1>>) -> !firrtl.uint<1>
    %1 = firrtl.subfield %bar_a(0) : (!firrtl.bundle<v: uint<1>>) -> !firrtl.uint<1>
    firrtl.connect %1, %0 : !firrtl.uint<1>, !firrtl.uint<1>
    %2 = firrtl.subfield %b(0) : (!firrtl.bundle<v: uint<1>>) -> !firrtl.uint<1>
    %3 = firrtl.subfield %bar_b(0) : (!firrtl.bundle<v: uint<1>>) -> !firrtl.uint<1>
    firrtl.connect %3, %2 : !firrtl.uint<1>, !firrtl.uint<1>
    %4 = firrtl.subfield %bar_c(0) : (!firrtl.bundle<v: uint<1>>) -> !firrtl.uint<1>
    %5 = firrtl.subfield %c(0) : (!firrtl.bundle<v: uint<1>>) -> !firrtl.uint<1>
    firrtl.connect %5, %4 : !firrtl.uint<1>, !firrtl.uint<1>
    %6 = firrtl.subfield %bar_d(0) : (!firrtl.bundle<v: uint<1>>) -> !firrtl.uint<1>
    %7 = firrtl.subfield %d(0) : (!firrtl.bundle<v: uint<1>>) -> !firrtl.uint<1>
    firrtl.connect %7, %6 : !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// -----
// CHECK-LABEL: "Oscillators"
firrtl.circuit "Oscillators"  {
  // CHECK: firrtl.module @Foo
  firrtl.module @Foo(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset, out %a: !firrtl.bundle<v1: uint<1>, v2: uint<1>>) {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %0 = firrtl.subfield %a(1) : (!firrtl.bundle<v1: uint<1>, v2: uint<1>>) -> !firrtl.uint<1>
    %1 = firrtl.subfield %a(0) : (!firrtl.bundle<v1: uint<1>, v2: uint<1>>) -> !firrtl.uint<1>
    // CHECK: firrtl.reg
    // CHECK-NEXT: firrtl.regreset
    %r = firrtl.reg %clock  : !firrtl.uint<1>
    %s = firrtl.regreset %clock, %reset, %c0_ui1  : !firrtl.asyncreset, !firrtl.uint<1>, !firrtl.uint<1>
    %2 = firrtl.not %r : (!firrtl.uint<1>) -> !firrtl.uint<1>
    firrtl.connect %r, %2 : !firrtl.uint<1>, !firrtl.uint<1>
    %3 = firrtl.not %s : (!firrtl.uint<1>) -> !firrtl.uint<1>
    firrtl.connect %s, %3 : !firrtl.uint<1>, !firrtl.uint<1>
    %4 = firrtl.or %r, %s : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    firrtl.connect %1, %4 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %0, %4 : !firrtl.uint<1>, !firrtl.uint<1>
  }

  // CHECK: firrtl.module @Bar
  firrtl.module @Bar(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset, out %a: !firrtl.bundle<v1: uint<1>, v2: uint<1>>) {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %c1_ui1 = firrtl.constant 1 : !firrtl.uint<1>
    %0 = firrtl.subfield %a(0) : (!firrtl.bundle<v1: uint<1>, v2: uint<1>>) -> !firrtl.uint<1>
    %1 = firrtl.subfield %a(1) : (!firrtl.bundle<v1: uint<1>, v2: uint<1>>) -> !firrtl.uint<1>
    // CHECK: firrtl.reg
    // CHECK-NEXT: firrtl.regreset
    %r = firrtl.reg %clock  : !firrtl.uint<1>
    %s = firrtl.regreset %clock, %reset, %c0_ui1  : !firrtl.asyncreset, !firrtl.uint<1>, !firrtl.uint<1>
    %2 = firrtl.xor %1, %c1_ui1 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    firrtl.connect %r, %2 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %s, %2 : !firrtl.uint<1>, !firrtl.uint<1>
    %3 = firrtl.or %r, %s : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    firrtl.connect %0, %3 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %1, %3 : !firrtl.uint<1>, !firrtl.uint<1>
  }

  // CHECK: firrtl.module @Baz
  firrtl.module @Baz(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset, out %a: !firrtl.bundle<v1: uint<1>, v2: uint<1>>) {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %0 = firrtl.subfield %a(0) : (!firrtl.bundle<v1: uint<1>, v2: uint<1>>) -> !firrtl.uint<1>
    %1 = firrtl.subfield %a(1) : (!firrtl.bundle<v1: uint<1>, v2: uint<1>>) -> !firrtl.uint<1>
    // CHECK: firrtl.reg
    // CHECK-NEXT: firrtl.regreset
    %r = firrtl.reg %clock  : !firrtl.uint<1>
    %s = firrtl.regreset %clock, %reset, %c0_ui1  : !firrtl.asyncreset, !firrtl.uint<1>, !firrtl.uint<1>
    %2 = firrtl.not %1 : (!firrtl.uint<1>) -> !firrtl.uint<1>
    firrtl.connect %r, %2 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %s, %2 : !firrtl.uint<1>, !firrtl.uint<1>
    %3 = firrtl.or %r, %s : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    firrtl.connect %0, %3 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %1, %3 : !firrtl.uint<1>, !firrtl.uint<1>
  }
  firrtl.extmodule @Ext(in a: !firrtl.bundle<v1: uint<1>, v2: uint<1>>)

  // CHECK: firrtl.module @Qux
  firrtl.module @Qux(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset, out %a: !firrtl.bundle<v1: uint<1>, v2: uint<1>>) {
    %c0_ui1 = firrtl.constant 0 : !firrtl.uint<1>
    %0 = firrtl.subfield %a(1) : (!firrtl.bundle<v1: uint<1>, v2: uint<1>>) -> !firrtl.uint<1>
    %1 = firrtl.subfield %a(0) : (!firrtl.bundle<v1: uint<1>, v2: uint<1>>) -> !firrtl.uint<1>
    %ext_a = firrtl.instance ext  @Ext(in a: !firrtl.bundle<v1: uint<1>, v2: uint<1>>)
    %2 = firrtl.subfield %ext_a(0) : (!firrtl.bundle<v1: uint<1>, v2: uint<1>>) -> !firrtl.uint<1>
    %3 = firrtl.subfield %ext_a(1) : (!firrtl.bundle<v1: uint<1>, v2: uint<1>>) -> !firrtl.uint<1>
    // CHECK: firrtl.reg
    // CHECK-NEXT: firrtl.regreset
    %r = firrtl.reg %clock  : !firrtl.uint<1>
    %s = firrtl.regreset %clock, %reset, %c0_ui1  : !firrtl.asyncreset, !firrtl.uint<1>, !firrtl.uint<1>
    %4 = firrtl.not %3 : (!firrtl.uint<1>) -> !firrtl.uint<1>
    firrtl.connect %r, %4 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %s, %4 : !firrtl.uint<1>, !firrtl.uint<1>
    %5 = firrtl.or %r, %s : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    firrtl.connect %2, %5 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %3, %5 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %1, %2 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %0, %3 : !firrtl.uint<1>, !firrtl.uint<1>
  }
  // CHECK: firrtl.module @Oscillators
  firrtl.module @Oscillators(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset, out %foo_a: !firrtl.bundle<v1: uint<1>, v2: uint<1>>, out %bar_a: !firrtl.bundle<v1: uint<1>, v2: uint<1>>, out %baz_a: !firrtl.bundle<v1: uint<1>, v2: uint<1>>, out %qux_a: !firrtl.bundle<v1: uint<1>, v2: uint<1>>) {
    %0 = firrtl.subfield %qux_a(1) : (!firrtl.bundle<v1: uint<1>, v2: uint<1>>) -> !firrtl.uint<1>
    %1 = firrtl.subfield %qux_a(0) : (!firrtl.bundle<v1: uint<1>, v2: uint<1>>) -> !firrtl.uint<1>
    %2 = firrtl.subfield %baz_a(1) : (!firrtl.bundle<v1: uint<1>, v2: uint<1>>) -> !firrtl.uint<1>
    %3 = firrtl.subfield %baz_a(0) : (!firrtl.bundle<v1: uint<1>, v2: uint<1>>) -> !firrtl.uint<1>
    %4 = firrtl.subfield %bar_a(1) : (!firrtl.bundle<v1: uint<1>, v2: uint<1>>) -> !firrtl.uint<1>
    %5 = firrtl.subfield %bar_a(0) : (!firrtl.bundle<v1: uint<1>, v2: uint<1>>) -> !firrtl.uint<1>
    %6 = firrtl.subfield %foo_a(1) : (!firrtl.bundle<v1: uint<1>, v2: uint<1>>) -> !firrtl.uint<1>
    %7 = firrtl.subfield %foo_a(0) : (!firrtl.bundle<v1: uint<1>, v2: uint<1>>) -> !firrtl.uint<1>
    %foo_clock, %foo_reset, %foo_a_0 = firrtl.instance foo  @Foo(in clock: !firrtl.clock, in reset: !firrtl.asyncreset, out a: !firrtl.bundle<v1: uint<1>, v2: uint<1>>)
    %8 = firrtl.subfield %foo_a_0(1) : (!firrtl.bundle<v1: uint<1>, v2: uint<1>>) -> !firrtl.uint<1>
    %9 = firrtl.subfield %foo_a_0(0) : (!firrtl.bundle<v1: uint<1>, v2: uint<1>>) -> !firrtl.uint<1>
    firrtl.connect %foo_clock, %clock : !firrtl.clock, !firrtl.clock
    firrtl.connect %foo_reset, %reset : !firrtl.asyncreset, !firrtl.asyncreset
    firrtl.connect %7, %9 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %6, %8 : !firrtl.uint<1>, !firrtl.uint<1>
    %bar_clock, %bar_reset, %bar_a_1 = firrtl.instance bar  @Bar(in clock: !firrtl.clock, in reset: !firrtl.asyncreset, out a: !firrtl.bundle<v1: uint<1>, v2: uint<1>>)
    %10 = firrtl.subfield %bar_a_1(1) : (!firrtl.bundle<v1: uint<1>, v2: uint<1>>) -> !firrtl.uint<1>
    %11 = firrtl.subfield %bar_a_1(0) : (!firrtl.bundle<v1: uint<1>, v2: uint<1>>) -> !firrtl.uint<1>
    firrtl.connect %bar_clock, %clock : !firrtl.clock, !firrtl.clock
    firrtl.connect %bar_reset, %reset : !firrtl.asyncreset, !firrtl.asyncreset
    firrtl.connect %5, %11 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %4, %10 : !firrtl.uint<1>, !firrtl.uint<1>
    %baz_clock, %baz_reset, %baz_a_2 = firrtl.instance baz  @Baz(in clock: !firrtl.clock, in reset: !firrtl.asyncreset, out a: !firrtl.bundle<v1: uint<1>, v2: uint<1>>)
    %12 = firrtl.subfield %baz_a_2(1) : (!firrtl.bundle<v1: uint<1>, v2: uint<1>>) -> !firrtl.uint<1>
    %13 = firrtl.subfield %baz_a_2(0) : (!firrtl.bundle<v1: uint<1>, v2: uint<1>>) -> !firrtl.uint<1>
    firrtl.connect %baz_clock, %clock : !firrtl.clock, !firrtl.clock
    firrtl.connect %baz_reset, %reset : !firrtl.asyncreset, !firrtl.asyncreset
    firrtl.connect %3, %13 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %2, %12 : !firrtl.uint<1>, !firrtl.uint<1>
    %qux_clock, %qux_reset, %qux_a_3 = firrtl.instance qux  @Qux(in clock: !firrtl.clock, in reset: !firrtl.asyncreset, out a: !firrtl.bundle<v1: uint<1>, v2: uint<1>>)
    %14 = firrtl.subfield %qux_a_3(1) : (!firrtl.bundle<v1: uint<1>, v2: uint<1>>) -> !firrtl.uint<1>
    %15 = firrtl.subfield %qux_a_3(0) : (!firrtl.bundle<v1: uint<1>, v2: uint<1>>) -> !firrtl.uint<1>
    firrtl.connect %qux_clock, %clock : !firrtl.clock, !firrtl.clock
    firrtl.connect %qux_reset, %reset : !firrtl.asyncreset, !firrtl.asyncreset
    firrtl.connect %1, %15 : !firrtl.uint<1>, !firrtl.uint<1>
    firrtl.connect %0, %14 : !firrtl.uint<1>, !firrtl.uint<1>
  }
}

// -----
firrtl.circuit "dntOutput"  {
  // CHECK-LABEL: firrtl.module @dntOutput
  // CHECK: %[[SUB_B:.+]] = firrtl.subfield %b(0)
  // CHECK: %[[SUB:.+]] = firrtl.subfield %int_b(0)
  // CHECK: %[[MUX:.+]] = firrtl.mux(%c, %[[SUB]], %c2_ui3)
  // CHECK-NEXT: firrtl.connect %[[SUB_B]], %[[MUX]]
  firrtl.module @dntOutput(out %b: !firrtl.bundle<v: uint<3>>, in %c: !firrtl.uint<1>) {
    %c2_ui3 = firrtl.constant 2 : !firrtl.uint<3>
    %0 = firrtl.subfield %b(0) : (!firrtl.bundle<v: uint<3>>) -> !firrtl.uint<3>
    %int_b = firrtl.instance int  @foo(out b: !firrtl.bundle<v: uint<3>>)
    %1 = firrtl.subfield %int_b(0) : (!firrtl.bundle<v: uint<3>>) -> !firrtl.uint<3>
    %2 = firrtl.mux(%c, %1, %c2_ui3) : (!firrtl.uint<1>, !firrtl.uint<3>, !firrtl.uint<3>) -> !firrtl.uint<3>
    firrtl.connect %0, %2 : !firrtl.uint<3>, !firrtl.uint<3>
  }
  firrtl.module @foo(out %b: !firrtl.bundle<v: uint<3>>) attributes {portSyms = ["dntSym1"] }{
    %c1_ui3 = firrtl.constant 1 : !firrtl.uint<3>
    %0 = firrtl.subfield %b(0) : (!firrtl.bundle<v: uint<3>>) -> !firrtl.uint<3>
    firrtl.connect %0, %c1_ui3 : !firrtl.uint<3>, !firrtl.uint<3>
  }
}
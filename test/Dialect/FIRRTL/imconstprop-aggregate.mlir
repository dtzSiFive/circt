// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl-imconstprop))' --split-input-file  %s | FileCheck %s

// This contains a lot of tests which should be caught by IMCP.
// For now, we are checking that the aggregates don't cause the pass to error out.

firrtl.circuit "VectorPropagation1" {
  // CHECK-LABEL: @VectorPropagation1
  module @VectorPropagation1(out %b: !firrtl.uint<1>) {
    %c1_ui1 = constant 1 : !firrtl.uint<1>
    %tmp = wire : !firrtl.vector<uint<1>, 2>
    %0 = subindex %tmp[0] : !firrtl.vector<uint<1>, 2>
    %1 = subindex %tmp[1] : !firrtl.vector<uint<1>, 2>
    %2 = xor %0, %1 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    strictconnect %0, %c1_ui1 : !firrtl.uint<1>
    strictconnect %1, %c1_ui1 : !firrtl.uint<1>
    // CHECK: strictconnect %b, %c0_ui1 : !firrtl.uint<1>
    strictconnect %b, %2 : !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "VectorPropagation2" {
  // CHECK-LABEL: @VectorPropagation2
  module @VectorPropagation2(out %b1: !firrtl.uint<6>, out %b2: !firrtl.uint<6>, out %b3: !firrtl.uint<6>) {

    // tmp1[0][0] <= 1
    // tmp1[0][1] <= 2
    // tmp1[1][0] <= 4
    // tmp1[1][1] <= 8
    // tmp1[2][0] <= 16
    // tmp1[2][1] <= 32

    // b1 <= tmp[0][0] xor tmp1[1][0] = 5
    // b2 <= tmp[2][1] xor tmp1[0][1] = 34
    // b3 <= tmp[1][1] xor tmp1[2][0] = 24

    %c32_ui6 = constant 32 : !firrtl.uint<6>
    %c16_ui6 = constant 16 : !firrtl.uint<6>
    %c8_ui6 = constant 8 : !firrtl.uint<6>
    %c4_ui6 = constant 4 : !firrtl.uint<6>
    %c2_ui6 = constant 2 : !firrtl.uint<6>
    %c1_ui6 = constant 1 : !firrtl.uint<6>
    %tmp = wire  : !firrtl.vector<vector<uint<6>, 2>, 3>
    %0 = subindex %tmp[0] : !firrtl.vector<vector<uint<6>, 2>, 3>
    %1 = subindex %0[0] : !firrtl.vector<uint<6>, 2>
    strictconnect %1, %c1_ui6 : !firrtl.uint<6>
    %2 = subindex %0[1] : !firrtl.vector<uint<6>, 2>
    strictconnect %2, %c2_ui6 : !firrtl.uint<6>
    %3 = subindex %tmp[1] : !firrtl.vector<vector<uint<6>, 2>, 3>
    %4 = subindex %3[0] : !firrtl.vector<uint<6>, 2>
    strictconnect %4, %c4_ui6 : !firrtl.uint<6>
    %5 = subindex %3[1] : !firrtl.vector<uint<6>, 2>
    strictconnect %5, %c8_ui6 : !firrtl.uint<6>
    %6 = subindex %tmp[2] : !firrtl.vector<vector<uint<6>, 2>, 3>
    %7 = subindex %6[0] : !firrtl.vector<uint<6>, 2>
    strictconnect %7, %c16_ui6 : !firrtl.uint<6>
    %8 = subindex %6[1] : !firrtl.vector<uint<6>, 2>
    strictconnect %8, %c32_ui6 : !firrtl.uint<6>
    %9 = xor %1, %4 : (!firrtl.uint<6>, !firrtl.uint<6>) -> !firrtl.uint<6>
    strictconnect %b1, %9 : !firrtl.uint<6>
    %10 = xor %8, %2 : (!firrtl.uint<6>, !firrtl.uint<6>) -> !firrtl.uint<6>
    strictconnect %b2, %10 : !firrtl.uint<6>
    %11 = xor %7, %5 : (!firrtl.uint<6>, !firrtl.uint<6>) -> !firrtl.uint<6>
    strictconnect %b3, %11 : !firrtl.uint<6>
    // CHECK:      strictconnect %b1, %c5_ui6 : !firrtl.uint<6>
    // CHECK-NEXT: strictconnect %b2, %c34_ui6 : !firrtl.uint<6>
    // CHECK-NEXT: strictconnect %b3, %c24_ui6 : !firrtl.uint<6>
  }
}

// -----

firrtl.circuit "BundlePropagation1"   {
  // CHECK-LABEL: @BundlePropagation1
  module @BundlePropagation1(out %result: !firrtl.uint<3>) {
    %tmp = wire  : !firrtl.bundle<a: uint<3>, b: uint<3>, c: uint<3>>
    %c1_ui3 = constant 1 : !firrtl.uint<3>
    %c2_ui3 = constant 2 : !firrtl.uint<3>
    %c4_ui3 = constant 4 : !firrtl.uint<3>
    %0 = subfield %tmp[a] : !firrtl.bundle<a: uint<3>, b: uint<3>, c: uint<3>>
    %1 = subfield %tmp[b] : !firrtl.bundle<a: uint<3>, b: uint<3>, c: uint<3>>
    %2 = subfield %tmp[c] : !firrtl.bundle<a: uint<3>, b: uint<3>, c: uint<3>>
    strictconnect %0, %c1_ui3 : !firrtl.uint<3>
    strictconnect %1, %c2_ui3 : !firrtl.uint<3>
    strictconnect %2, %c4_ui3 : !firrtl.uint<3>
    %3 = xor %0, %1 : (!firrtl.uint<3>, !firrtl.uint<3>) -> !firrtl.uint<3>
    %4 = xor %3, %2 : (!firrtl.uint<3>, !firrtl.uint<3>) -> !firrtl.uint<3>
    strictconnect %result, %4 : !firrtl.uint<3>
    // CHECK:  strictconnect %result, %c7_ui3 : !firrtl.uint<3>
  }
}

// -----

firrtl.circuit "DontTouchAggregate" {
  module @DontTouchAggregate(in %clock: !firrtl.clock, out %out1: !firrtl.uint<1>, out %out2: !firrtl.uint<1>) {
    %init = wire sym @dntSym: !firrtl.vector<uint<1>, 2>
    %0 = subindex %init[0] : !firrtl.vector<uint<1>, 2>
    %1 = subindex %init[1] : !firrtl.vector<uint<1>, 2>
    %true = constant 1 : !firrtl.uint<1>
    strictconnect %0, %true : !firrtl.uint<1>
    strictconnect %1, %true : !firrtl.uint<1>

    // CHECK:      strictconnect %out1, %0 : !firrtl.uint<1>
    // CHECK-NEXT: strictconnect %out2, %1 : !firrtl.uint<1>
    strictconnect %out1, %0 : !firrtl.uint<1>
    strictconnect %out2, %1 : !firrtl.uint<1>
  }
}

// -----
// Following tests are ported from normal imconstprop tests.

firrtl.circuit "OutPortTop" {
  // Check that we don't propagate througth it.
  module @OutPortChild(out %out: !firrtl.vector<uint<1>, 2> sym @dntSym)
  {
    %c0_ui1 = constant 0 : !firrtl.uint<1>
    %0 = subindex %out[0] : !firrtl.vector<uint<1>, 2>
    %1 = subindex %out[1] : !firrtl.vector<uint<1>, 2>
    strictconnect %0, %c0_ui1 : !firrtl.uint<1>
    strictconnect %1, %c0_ui1 : !firrtl.uint<1>
  }
  // CHECK-LABEL: module @OutPortTop
  module @OutPortTop(out %out1: !firrtl.uint<1>, out %out2: !firrtl.uint<1>) {
    %c_out = instance c @OutPortChild(out out: !firrtl.vector<uint<1>, 2>)
    %0 = subindex %c_out[0] : !firrtl.vector<uint<1>, 2>
    %1 = subindex %c_out[1] : !firrtl.vector<uint<1>, 2>
    strictconnect %out1, %0 : !firrtl.uint<1>
    strictconnect %out2, %1 : !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "InputPortTop"  {
  // CHECK-LABEL: module private @InputPortChild2
  module private @InputPortChild2(in %in0: !firrtl.bundle<v: uint<1>>, in %in1: !firrtl.bundle<v: uint<1>>, out %out: !firrtl.bundle<v: uint<1>>) {
    // CHECK: and %0, %c1_ui1
    %0 = subfield %in1[v] : !firrtl.bundle<v: uint<1>>
    %1 = subfield %in0[v] : !firrtl.bundle<v: uint<1>>
    %2 = subfield %out[v] : !firrtl.bundle<v: uint<1>>
    %3 = and %1, %0 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    strictconnect %2, %3 : !firrtl.uint<1>
  }
  // CHECK-LABEL: module private @InputPortChild
  module private @InputPortChild(in %in0: !firrtl.bundle<v: uint<1>>,
    in %in1: !firrtl.bundle<v: uint<1>> sym @dntSym,
    out %out: !firrtl.bundle<v: uint<1>>)
  {
    // CHECK: and %1, %0
    %0 = subfield %in1[v] : !firrtl.bundle<v: uint<1>>
    %1 = subfield %in0[v] : !firrtl.bundle<v: uint<1>>
    %2 = subfield %out[v] : !firrtl.bundle<v: uint<1>>
    %3 = and %1, %0 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    strictconnect %2, %3 : !firrtl.uint<1>
  }

  // CHECK-LABEL: module @InputPortTop
  module @InputPortTop(in %x: !firrtl.bundle<v: uint<1>>, out %z: !firrtl.bundle<v: uint<1>>, out %z2: !firrtl.bundle<v: uint<1>>) {
    %c1_ui1 = constant 1 : !firrtl.uint<1>
    %0 = subfield %z2[v] : !firrtl.bundle<v: uint<1>>
    %1 = subfield %x[v] : !firrtl.bundle<v: uint<1>>
    %2 = subfield %z[v] : !firrtl.bundle<v: uint<1>>
    %c_in0, %c_in1, %c_out = instance c  @InputPortChild(in in0: !firrtl.bundle<v: uint<1>>, in in1: !firrtl.bundle<v: uint<1>>, out out: !firrtl.bundle<v: uint<1>>)
    %3 = subfield %c_in1[v] : !firrtl.bundle<v: uint<1>>
    %4 = subfield %c_in0[v] : !firrtl.bundle<v: uint<1>>
    %5 = subfield %c_out[v] : !firrtl.bundle<v: uint<1>>
    %c2_in0, %c2_in1, %c2_out = instance c2  @InputPortChild2(in in0: !firrtl.bundle<v: uint<1>>, in in1: !firrtl.bundle<v: uint<1>>, out out: !firrtl.bundle<v: uint<1>>)
    %6 = subfield %c2_in1[v] : !firrtl.bundle<v: uint<1>>
    %7 = subfield %c2_in0[v] : !firrtl.bundle<v: uint<1>>
    %8 = subfield %c2_out[v] : !firrtl.bundle<v: uint<1>>
    strictconnect %2, %5 : !firrtl.uint<1>
    strictconnect %4, %1 : !firrtl.uint<1>
    strictconnect %3, %c1_ui1 : !firrtl.uint<1>
    strictconnect %0, %8 : !firrtl.uint<1>
    strictconnect %7, %1 : !firrtl.uint<1>
    strictconnect %6, %c1_ui1 : !firrtl.uint<1>
  }
}

// -----

// CHECK-LABK: circuit "rhs_sink_output_used_as_wire"
// This test checks that an output port sink, used as a RHS of a connect, is not
// optimized away.  This is similar to the oscillator tests above, but more
// reduced. See:
//   - https://github.com/llvm/circt/issues/1488
//
firrtl.circuit "rhs_sink_output_used_as_wire" {
  // CHECK-LABEL: module private @Bar
  module private @Bar(in %a: !firrtl.bundle<v: uint<1>>, in %b: !firrtl.bundle<v: uint<1>>, out %c: !firrtl.bundle<v: uint<1>>, out %d: !firrtl.bundle<v: uint<1>>) {
    %0 = subfield %d[v] : !firrtl.bundle<v: uint<1>>
    %1 = subfield %a[v] : !firrtl.bundle<v: uint<1>>
    %2 = subfield %b[v] : !firrtl.bundle<v: uint<1>>
    %3 = subfield %c[v] : !firrtl.bundle<v: uint<1>>
    strictconnect %3, %2 : !firrtl.uint<1>
    %_c = wire  : !firrtl.bundle<v: uint<1>>
    %4 = subfield %_c[v] : !firrtl.bundle<v: uint<1>>
    %5 = xor %1, %3 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    strictconnect %4, %5 : !firrtl.uint<1>
    strictconnect %0, %4 : !firrtl.uint<1>
  }

  // CHECK-LABEL: module @rhs_sink_output_used_as_wire
  module @rhs_sink_output_used_as_wire(in %a: !firrtl.bundle<v: uint<1>>, in %b: !firrtl.bundle<v: uint<1>>, out %c: !firrtl.bundle<v: uint<1>>, out %d: !firrtl.bundle<v: uint<1>>) {
    %bar_a, %bar_b, %bar_c, %bar_d = instance bar  @Bar(in a: !firrtl.bundle<v: uint<1>>, in b: !firrtl.bundle<v: uint<1>>, out c: !firrtl.bundle<v: uint<1>>, out d: !firrtl.bundle<v: uint<1>>)
    %0 = subfield %a[v] : !firrtl.bundle<v: uint<1>>
    %1 = subfield %bar_a[v] : !firrtl.bundle<v: uint<1>>
    strictconnect %1, %0 : !firrtl.uint<1>
    %2 = subfield %b[v] : !firrtl.bundle<v: uint<1>>
    %3 = subfield %bar_b[v] : !firrtl.bundle<v: uint<1>>
    strictconnect %3, %2 : !firrtl.uint<1>
    %4 = subfield %bar_c[v] : !firrtl.bundle<v: uint<1>>
    %5 = subfield %c[v] : !firrtl.bundle<v: uint<1>>
    strictconnect %5, %4 : !firrtl.uint<1>
    %6 = subfield %bar_d[v] : !firrtl.bundle<v: uint<1>>
    %7 = subfield %d[v] : !firrtl.bundle<v: uint<1>>
    strictconnect %7, %6 : !firrtl.uint<1>
  }
}

// -----
firrtl.circuit "dntOutput"  {
  // CHECK-LABEL: module @dntOutput
  // CHECK:      %[[INT_B_V:.+]] = subfield %int_b[v] : !firrtl.bundle<v: uint<3>>
  // CHECK-NEXT: %[[MUX:.+]] = mux(%c, %[[INT_B_V]], %c2_ui3)
  module @dntOutput(out %b: !firrtl.bundle<v: uint<3>>, in %c: !firrtl.uint<1>) {
    %c2_ui3 = constant 2 : !firrtl.uint<3>
    %0 = subfield %b[v] : !firrtl.bundle<v: uint<3>>
    %int_b = instance int  @foo(out b: !firrtl.bundle<v: uint<3>>)
    %1 = subfield %int_b[v] : !firrtl.bundle<v: uint<3>>
    %2 = mux(%c, %1, %c2_ui3) : (!firrtl.uint<1>, !firrtl.uint<3>, !firrtl.uint<3>) -> !firrtl.uint<3>
    strictconnect %0, %2 : !firrtl.uint<3>
  }
  module private @foo(out %b: !firrtl.bundle<v: uint<3>> sym @dntSym1){
    %c1_ui3 = constant 1 : !firrtl.uint<3>
    %0 = subfield %b[v] : !firrtl.bundle<v: uint<3>>
    strictconnect %0, %c1_ui3 : !firrtl.uint<3>
  }
}

// -----

firrtl.circuit "Issue4369"  {
  // CHECK-LABEL: module private @Bar
  module private @Bar(in %in: !firrtl.vector<uint<1>, 1>, out %out: !firrtl.uint<1>) {
    %0 = subindex %in[0] : !firrtl.vector<uint<1>, 1>
    %a = wire   : !firrtl.uint<1>
    // CHECK: strictconnect %a, %0
    // CHECK-NEXT: strictconnect %out, %a
    strictconnect %a, %0 : !firrtl.uint<1>
    strictconnect %out, %a : !firrtl.uint<1>
  }
  module @Issue4369(in %a_0: !firrtl.uint<1>, out %b: !firrtl.uint<1>) {
    %bar_in, %bar_out = instance bar  @Bar(in in: !firrtl.vector<uint<1>, 1>, out out: !firrtl.uint<1>)
    %0 = subindex %bar_in[0] : !firrtl.vector<uint<1>, 1>
    strictconnect %0, %a_0 : !firrtl.uint<1>
    strictconnect %b, %bar_out : !firrtl.uint<1>
  }
}

// -----
firrtl.circuit "AggregateConstant"  {
  // CHECK-LABEL: AggregateConstant
  module @AggregateConstant(out %out: !firrtl.uint<1>) {
    %0 = aggregateconstant [0 : ui1, 1 : ui1] : !firrtl.vector<uint<1>, 2>
    %w = wire : !firrtl.vector<uint<1>, 2>
    %1 = subindex %w[1] : !firrtl.vector<uint<1>, 2>
    strictconnect %out, %1 : !firrtl.uint<1>
    // CHECK: strictconnect %out, %c1_ui1
    strictconnect %w, %0 : !firrtl.vector<uint<1>, 2>
  }
}
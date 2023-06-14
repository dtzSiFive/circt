// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl-imconstprop))' --split-input-file  %s | FileCheck %s

firrtl.circuit "Test" {

  // CHECK-LABEL: @PassThrough
  // CHECK: (in %source: !firrtl.uint<1>, out %dest: !firrtl.uint<1>)
  module private @PassThrough(in %source: !firrtl.uint<1>, out %dest: !firrtl.uint<1>) {
    // CHECK-NEXT: %c0_ui1 = constant 0 : !firrtl.uint<1>

    %dontTouchWire = wire sym @a1 : !firrtl.uint<1>
    // CHECK-NEXT: %dontTouchWire = wire
    strictconnect %dontTouchWire, %source : !firrtl.uint<1>
    // CHECK-NEXT: strictconnect %dontTouchWire, %c0_ui1

    // CHECK-NEXT: strictconnect %dest, %dontTouchWire
    strictconnect %dest, %dontTouchWire : !firrtl.uint<1>
    // CHECK-NEXT: }
  }

  // CHECK-LABEL: @Test
  module @Test(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>,
                      out %result1: !firrtl.uint<1>,
                      out %result2: !firrtl.clock,
                      out %result3: !firrtl.uint<1>,
                      out %result4: !firrtl.uint<1>,
                      out %result5: !firrtl.uint<2>,
                      out %result6: !firrtl.uint<2>,
                      out %result7: !firrtl.uint<4>,
                      out %result8: !firrtl.uint<4>,
                      out %result9: !firrtl.uint<2>) {
    %c0_ui1 = constant 0 : !firrtl.uint<1>
    %c0_ui2 = constant 0 : !firrtl.uint<2>
    %c0_ui4 = constant 0 : !firrtl.uint<4>
    %c1_ui1 = constant 1 : !firrtl.uint<1>

    // Trivial wire constant propagation.
    %someWire = wire interesting_name : !firrtl.uint<1>
    strictconnect %someWire, %c0_ui1 : !firrtl.uint<1>

    // CHECK: %someWire = wire
    // CHECK: strictconnect %someWire, %c0_ui1
    // CHECK: strictconnect %result1, %c0_ui1
    strictconnect %result1, %someWire : !firrtl.uint<1>

    // Trivial wire special constant propagation.
    %c0_clock = specialconstant 0 : !firrtl.clock
    %clockWire = wire interesting_name : !firrtl.clock
    strictconnect %clockWire, %c0_clock : !firrtl.clock

    // CHECK: %clockWire = wire
    // CHECK: strictconnect %clockWire, %c0_clock
    // CHECK: strictconnect %result2, %c0_clock
    strictconnect %result2, %clockWire : !firrtl.clock

    // Not a constant.
    %nonconstWire = wire : !firrtl.uint<1>
    strictconnect %nonconstWire, %c0_ui1 : !firrtl.uint<1>
    strictconnect %nonconstWire, %c1_ui1 : !firrtl.uint<1>

    // CHECK: strictconnect %result3, %nonconstWire
    strictconnect %result3, %nonconstWire : !firrtl.uint<1>

    // Constant propagation through instance.
    %source, %dest = instance "" sym @dm21 @PassThrough(in source: !firrtl.uint<1>, out dest: !firrtl.uint<1>)

    // CHECK: strictconnect %inst_source, %c0_ui1
    strictconnect %source, %c0_ui1 : !firrtl.uint<1>

    // CHECK: strictconnect %result4, %inst_dest
    strictconnect %result4, %dest : !firrtl.uint<1>

    // Check connect extensions.
    %extWire = wire : !firrtl.uint<2>
    strictconnect %extWire, %c0_ui2 : !firrtl.uint<2>

    // Connects of invalid values should hurt.
    %invalid = invalidvalue : !firrtl.uint<2>
    strictconnect %extWire, %invalid : !firrtl.uint<2>

    // CHECK-NOT: strictconnect %result5, %c0_ui2
    strictconnect %result5, %extWire: !firrtl.uint<2>

    // Constant propagation through instance.
    instance ReadMem @ReadMem()
  }

  // Unused modules should NOT be completely dropped.
  // https://github.com/llvm/circt/issues/1236

  // CHECK-LABEL: @UnusedModule(in %source: !firrtl.uint<1>, out %dest: !firrtl.uint<1>)
  module private @UnusedModule(in %source: !firrtl.uint<1>, out %dest: !firrtl.uint<1>) {
    // CHECK-NEXT: strictconnect %dest, %source
    strictconnect %dest, %source : !firrtl.uint<1>
    // CHECK-NEXT: }
  }


  // CHECK-LABEL: ReadMem
  module private @ReadMem() {
    %c0_ui1 = constant 0 : !firrtl.uint<4>
    %c1_ui1 = constant 1 : !firrtl.uint<1>

    %0 = mem Undefined {depth = 16 : i64, name = "ReadMemory", portNames = ["read0"], readLatency = 1 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: sint<8>>

    %1 = subfield %0[data] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: sint<8>>
    %2 = subfield %0[addr] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: sint<8>>
    strictconnect %2, %c0_ui1 : !firrtl.uint<4>
    %3 = subfield %0[en] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: sint<8>>
    strictconnect %3, %c1_ui1 : !firrtl.uint<1>
    %4 = subfield %0[clk] : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: sint<8>>
  }
}

// -----

// CHECK-LABEL: module @Issue1188
// https://github.com/llvm/circt/issues/1188
// Make sure that we handle recursion through muxes correctly.
firrtl.circuit "Issue1188"  {
  module @Issue1188(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, out %io_out: !firrtl.uint<6>, out %io_out3: !firrtl.uint<3>) {
    %c1_ui6 = constant 1 : !firrtl.uint<6>
    %D0123456 = reg %clock  : !firrtl.clock, !firrtl.uint<6>
    %0 = bits %D0123456 4 to 0 : (!firrtl.uint<6>) -> !firrtl.uint<5>
    %1 = bits %D0123456 5 to 5 : (!firrtl.uint<6>) -> !firrtl.uint<1>
    %2 = cat %0, %1 : (!firrtl.uint<5>, !firrtl.uint<1>) -> !firrtl.uint<6>
    %3 = bits %D0123456 4 to 4 : (!firrtl.uint<6>) -> !firrtl.uint<1>
    %4 = xor %2, %3 : (!firrtl.uint<6>, !firrtl.uint<1>) -> !firrtl.uint<6>
    %5 = bits %D0123456 1 to 1 : (!firrtl.uint<6>) -> !firrtl.uint<1>
    %6 = bits %D0123456 3 to 3 : (!firrtl.uint<6>) -> !firrtl.uint<1>
    %7 = cat %5, %6 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<2>
    %8 = cat %7, %1 : (!firrtl.uint<2>, !firrtl.uint<1>) -> !firrtl.uint<3>
    strictconnect %io_out, %D0123456 : !firrtl.uint<6>
    strictconnect %io_out3, %8 : !firrtl.uint<3>
    // CHECK: mux(%reset, %c1_ui6, %4)
    %9 = mux(%reset, %c1_ui6, %4) : (!firrtl.uint<1>, !firrtl.uint<6>, !firrtl.uint<6>) -> !firrtl.uint<6>
    strictconnect %D0123456, %9 : !firrtl.uint<6>
  }
}

// -----

// DontTouch annotation should block constant propagation.
firrtl.circuit "testDontTouch"  {
  // CHECK-LABEL: module private @blockProp
  module private @blockProp1(in %clock: !firrtl.clock,
    in %a: !firrtl.uint<1> sym @dntSym, out %b: !firrtl.uint<1>){
    //CHECK: %c = reg
    %c = reg %clock : !firrtl.clock, !firrtl.uint<1>
    strictconnect %c, %a : !firrtl.uint<1>
    strictconnect %b, %c : !firrtl.uint<1>
  }
  // CHECK-LABEL: module private @allowProp
  module private @allowProp(in %clock: !firrtl.clock, in %a: !firrtl.uint<1>, out %b: !firrtl.uint<1>) {
    // CHECK: [[CONST:%.+]] = constant 1 : !firrtl.uint<1>
    %c = wire  : !firrtl.uint<1>
    strictconnect %c, %a : !firrtl.uint<1>
    // CHECK: strictconnect %b, [[CONST]]
    strictconnect %b, %c : !firrtl.uint<1>
  }
  // CHECK-LABEL: module private @blockProp3
  module private @blockProp3(in %clock: !firrtl.clock, in %a: !firrtl.uint<1> , out %b: !firrtl.uint<1>) {
    //CHECK: %c = reg
    %c = reg sym @s2 %clock : !firrtl.clock, !firrtl.uint<1>
    strictconnect %c, %a : !firrtl.uint<1>
    strictconnect %b, %c : !firrtl.uint<1>
  }
  // CHECK-LABEL: module @testDontTouch
  module @testDontTouch(in %clock: !firrtl.clock, out %a: !firrtl.uint<1>, out %a1: !firrtl.uint<1>, out %a2: !firrtl.uint<1>) {
    %c1_ui1 = constant 1 : !firrtl.uint<1>
    %blockProp1_clock, %blockProp1_a, %blockProp1_b = instance blockProp1 sym @a1 @blockProp1(in clock: !firrtl.clock, in a: !firrtl.uint<1>, out b: !firrtl.uint<1>)
    %allowProp_clock, %allowProp_a, %allowProp_b = instance allowProp sym @a2 @allowProp(in clock: !firrtl.clock, in a: !firrtl.uint<1>, out b: !firrtl.uint<1>)
    %blockProp3_clock, %blockProp3_a, %blockProp3_b = instance blockProp3  sym @a3 @blockProp3(in clock: !firrtl.clock, in a: !firrtl.uint<1>, out b: !firrtl.uint<1>)
    strictconnect %blockProp1_clock, %clock : !firrtl.clock
    strictconnect %allowProp_clock, %clock : !firrtl.clock
    strictconnect %blockProp3_clock, %clock : !firrtl.clock
    strictconnect %blockProp1_a, %c1_ui1 : !firrtl.uint<1>
    strictconnect %allowProp_a, %c1_ui1 : !firrtl.uint<1>
    strictconnect %blockProp3_a, %c1_ui1 : !firrtl.uint<1>
    // CHECK: strictconnect %a, %blockProp1_b
    strictconnect %a, %blockProp1_b : !firrtl.uint<1>
    // CHECK: strictconnect %a1, %c
    strictconnect %a1, %allowProp_b : !firrtl.uint<1>
    // CHECK: strictconnect %a2, %blockProp3_b
    strictconnect %a2, %blockProp3_b : !firrtl.uint<1>
  }
  // CHECK-LABEL: module @CheckNode
  module @CheckNode(out %x: !firrtl.uint<1>, out %y: !firrtl.uint<1>, out %z: !firrtl.uint<1>) {
    %c1_ui1 = constant 1 : !firrtl.uint<1>
    // CHECK-NOT: %d1 = node
    %d1 = node droppable_name %c1_ui1 : !firrtl.uint<1>
    // CHECK: %d2 = node
    %d2 = node interesting_name %c1_ui1 : !firrtl.uint<1>
    // CHECK: %d3 = node
    %d3 = node   sym @s2 %c1_ui1: !firrtl.uint<1>
    // CHECK: strictconnect %x, %c1_ui1
    strictconnect %x, %d1 : !firrtl.uint<1>
    // CHECK: strictconnect %y, %c1_ui1
    strictconnect %y, %d2 : !firrtl.uint<1>
    // CHECK: strictconnect %z, %d3
    strictconnect %z, %d3 : !firrtl.uint<1>
  }

}

// -----

firrtl.circuit "OutPortTop" {
    module private @OutPortChild1(out %out: !firrtl.uint<1> sym @dntSym1) {
      %c0_ui1 = constant 0 : !firrtl.uint<1>
      strictconnect %out, %c0_ui1 : !firrtl.uint<1>
    }
    module private @OutPortChild2(out %out: !firrtl.uint<1>) {
      %c0_ui1 = constant 0 : !firrtl.uint<1>
      strictconnect %out, %c0_ui1 : !firrtl.uint<1>
    }
  // CHECK-LABEL: module @OutPortTop
    module @OutPortTop(in %x: !firrtl.uint<1>, out %zc: !firrtl.uint<1>, out %zn: !firrtl.uint<1>) {
      // CHECK: %c0_ui1 = constant 0
      %c_out = instance c  sym @a2 @OutPortChild1(out out: !firrtl.uint<1>)
      %c_out_0 = instance c  sym @a1 @OutPortChild2(out out: !firrtl.uint<1>)
      // CHECK: %0 = and %x, %c_out
      %0 = and %x, %c_out : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
      %1 = and %x, %c_out_0 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
      // CHECK: strictconnect %zn, %0
      strictconnect %zn, %0 : !firrtl.uint<1>
      // CHECK: strictconnect %zc, %c0_ui1
      strictconnect %zc, %1 : !firrtl.uint<1>
    }
}


// -----

firrtl.circuit "InputPortTop"   {
  // CHECK-LABEL: module private @InputPortChild2
  module private @InputPortChild2(in %in0: !firrtl.uint<1>, in %in1: !firrtl.uint<1>, out %out: !firrtl.uint<1>) {
    // CHECK: = constant 1
    %0 = and %in0, %in1 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    strictconnect %out, %0 : !firrtl.uint<1>
  }
  // CHECK-LABEL: module private @InputPortChild
  module private @InputPortChild(in %in0: !firrtl.uint<1>,
    in %in1 : !firrtl.uint<1> sym @dntSym1, out %out: !firrtl.uint<1>) {
    // CHECK: %0 = and %in0, %in1
    %0 = and %in0, %in1 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    strictconnect %out, %0 : !firrtl.uint<1>
  }
  module @InputPortTop(in %x: !firrtl.uint<1>, out %z: !firrtl.uint<1>, out %z2: !firrtl.uint<1>) {
    %c1_ui1 = constant 1 : !firrtl.uint<1>
    %c_in0, %c_in1, %c_out = instance c @InputPortChild(in in0: !firrtl.uint<1>, in in1: !firrtl.uint<1>, out out: !firrtl.uint<1>)
    %c2_in0, %c2_in1, %c2_out = instance c2 @InputPortChild2(in in0: !firrtl.uint<1>, in in1: !firrtl.uint<1>, out out: !firrtl.uint<1>)
    strictconnect %z, %c_out : !firrtl.uint<1>
    strictconnect %c_in0, %x : !firrtl.uint<1>
    strictconnect %c_in1, %c1_ui1 : !firrtl.uint<1>
    strictconnect %z2, %c2_out : !firrtl.uint<1>
    strictconnect %c2_in0, %x : !firrtl.uint<1>
    strictconnect %c2_in1, %c1_ui1 : !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "InstanceOut"   {
  extmodule private @Ext(in a: !firrtl.uint<1>)

  // CHECK-LABEL: module @InstanceOut
  module @InstanceOut(in %a: !firrtl.uint<1>, out %b: !firrtl.uint<1>) {
    %ext_a = instance ext @Ext(in a: !firrtl.uint<1>)
    strictconnect %ext_a, %a : !firrtl.uint<1>
    %w = wire  : !firrtl.uint<1>
    // CHECK: strictconnect %w, %ext_a : !firrtl.uint<1>
    strictconnect %w, %ext_a : !firrtl.uint<1>
    // CHECK: strictconnect %b, %w : !firrtl.uint<1>
    strictconnect %b, %w : !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "InstanceOut2"   {
  module private @Ext(in %a: !firrtl.uint<1>) {
  }

  // CHECK-LABEL: module @InstanceOut2
  module @InstanceOut2(in %a: !firrtl.uint<1>, out %b: !firrtl.uint<1>) {
    %ext_a = instance ext @Ext(in a: !firrtl.uint<1>)
    strictconnect %ext_a, %a : !firrtl.uint<1>
    %w = wire  : !firrtl.uint<1>
    // CHECK: strictconnect %w, %ext_a : !firrtl.uint<1>
    strictconnect %w, %ext_a : !firrtl.uint<1>
    // CHECK: strictconnect %b, %w : !firrtl.uint<1>
    strictconnect %b, %w : !firrtl.uint<1>
  }
}

// -----

firrtl.circuit "invalidReg1"   {
  // CHECK-LABEL: @invalidReg1
  module @invalidReg1(in %clock: !firrtl.clock, out %a: !firrtl.uint<1>) {
    %foobar = reg %clock  : !firrtl.clock, !firrtl.uint<1>
      //CHECK: %0 = not %foobar : (!firrtl.uint<1>) -> !firrtl.uint<1>
      %0 = not %foobar : (!firrtl.uint<1>) -> !firrtl.uint<1>
      //CHECK: strictconnect %foobar, %0 : !firrtl.uint<1>
      strictconnect %foobar, %0 : !firrtl.uint<1>
      //CHECK: strictconnect %a, %foobar : !firrtl.uint<1>
      strictconnect %a, %foobar : !firrtl.uint<1>
  }
}

// -----

// This test is checking the behavior of a RegOp, "r", and a RegResetOp, "s",
// that are combinationally connected to themselves through simple and weird
// formulations.  In all cases it should NOT be optimized away.  For more discussion, see:
//   - https://github.com/llvm/circt/issues/1465
//   - https://github.com/llvm/circt/issues/1466
//   - https://github.com/llvm/circt/issues/1478
//
// CHECK-LABEL: "Oscillators"
firrtl.circuit "Oscillators"   {
  // CHECK: module private @Foo
  module private @Foo(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset, out %a: !firrtl.uint<1>) {
    // CHECK: reg
    %r = reg %clock : !firrtl.clock, !firrtl.uint<1>
    %c0_ui1 = constant 0 : !firrtl.uint<1>
    // CHECK: regreset
    %s = regreset %clock, %reset, %c0_ui1 : !firrtl.clock, !firrtl.asyncreset, !firrtl.uint<1>, !firrtl.uint<1>
    %0 = not %r : (!firrtl.uint<1>) -> !firrtl.uint<1>
    strictconnect %r, %0 : !firrtl.uint<1>
    %1 = not %s : (!firrtl.uint<1>) -> !firrtl.uint<1>
    strictconnect %s, %1 : !firrtl.uint<1>
    %2 = or %r, %s : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    strictconnect %a, %2 : !firrtl.uint<1>
  }
  // CHECK: module private @Bar
  module private @Bar(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset, out %a: !firrtl.uint<1>) {
    // CHECK: %r = reg
    %r = reg %clock  : !firrtl.clock, !firrtl.uint<1>
    %c0_ui1 = constant 0 : !firrtl.uint<1>
    // CHECK: regreset
    %s = regreset %clock, %reset, %c0_ui1 : !firrtl.clock, !firrtl.asyncreset, !firrtl.uint<1>, !firrtl.uint<1>
    %c1_ui1 = constant 1 : !firrtl.uint<1>
    %0 = xor %a, %c1_ui1 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    strictconnect %r, %0 : !firrtl.uint<1>
    %1 = xor %a, %c1_ui1 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    strictconnect %s, %1 : !firrtl.uint<1>
    %2 = or %r, %s : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    strictconnect %a, %2 : !firrtl.uint<1>
  }
  // CHECK: module private @Baz
  module private @Baz(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset, out %a: !firrtl.uint<1>) {
    // CHECK: reg
    %r = reg %clock  : !firrtl.clock, !firrtl.uint<1>
    %c0_ui1 = constant 0 : !firrtl.uint<1>
    // CHECK: regreset
    %s = regreset %clock, %reset, %c0_ui1 : !firrtl.clock, !firrtl.asyncreset, !firrtl.uint<1>, !firrtl.uint<1>
    %0 = not %a : (!firrtl.uint<1>) -> !firrtl.uint<1>
    strictconnect %r, %0 : !firrtl.uint<1>
    %1 = not %a : (!firrtl.uint<1>) -> !firrtl.uint<1>
    strictconnect %s, %1 : !firrtl.uint<1>
    %2 = or %r, %s : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    strictconnect %a, %2 : !firrtl.uint<1>
  }
  extmodule @Ext(in a: !firrtl.uint<1>)
  // CHECK: module private @Qux
  module private @Qux(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset, out %a: !firrtl.uint<1>) {
    %ext_a = instance ext @Ext(in a: !firrtl.uint<1>)
    // CHECK: reg
    %r = reg %clock  : !firrtl.clock, !firrtl.uint<1>
    %c0_ui1 = constant 0 : !firrtl.uint<1>
    // CHECK: regreset
    %s = regreset %clock, %reset, %c0_ui1 : !firrtl.clock, !firrtl.asyncreset, !firrtl.uint<1>, !firrtl.uint<1>
    %0 = not %ext_a : (!firrtl.uint<1>) -> !firrtl.uint<1>
    strictconnect %r, %0 : !firrtl.uint<1>
    %1 = not %ext_a : (!firrtl.uint<1>) -> !firrtl.uint<1>
    strictconnect %s, %1 : !firrtl.uint<1>
    %2 = or %r, %s : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    strictconnect %ext_a, %2 : !firrtl.uint<1>
    strictconnect %a, %ext_a : !firrtl.uint<1>
  }
  module @Oscillators(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset, out %foo_a: !firrtl.uint<1>, out %bar_a: !firrtl.uint<1>, out %baz_a: !firrtl.uint<1>, out %qux_a: !firrtl.uint<1>) {
    %foo_clock, %foo_reset, %foo_a_0 = instance foo @Foo(in clock: !firrtl.clock, in reset: !firrtl.asyncreset, out a: !firrtl.uint<1>)
    strictconnect %foo_clock, %clock : !firrtl.clock
    strictconnect %foo_reset, %reset : !firrtl.asyncreset
    strictconnect %foo_a, %foo_a_0 : !firrtl.uint<1>
    %bar_clock, %bar_reset, %bar_a_1 = instance bar @Bar (in clock: !firrtl.clock, in reset: !firrtl.asyncreset, out a: !firrtl.uint<1>)
    strictconnect %bar_clock, %clock : !firrtl.clock
    strictconnect %bar_reset, %reset : !firrtl.asyncreset
    strictconnect %bar_a, %bar_a_1 : !firrtl.uint<1>
    %baz_clock, %baz_reset, %baz_a_2 = instance baz @Baz(in clock: !firrtl.clock, in reset: !firrtl.asyncreset, out a: !firrtl.uint<1>)
    strictconnect %baz_clock, %clock : !firrtl.clock
    strictconnect %baz_reset, %reset : !firrtl.asyncreset
    strictconnect %baz_a, %baz_a_2 : !firrtl.uint<1>
    %qux_clock, %qux_reset, %qux_a_3 = instance qux @Qux(in clock: !firrtl.clock, in reset: !firrtl.asyncreset, out a: !firrtl.uint<1>)
    strictconnect %qux_clock, %clock : !firrtl.clock
    strictconnect %qux_reset, %reset : !firrtl.asyncreset
    strictconnect %qux_a, %qux_a_3 : !firrtl.uint<1>
  }
}

// -----

// This test checks that an output port sink, used as a RHS of a connect, is not
// optimized away.  This is similar to the oscillator tests above, but more
// reduced. See:
//   - https://github.com/llvm/circt/issues/1488
//
// CHECK-LABEL: circuit "rhs_sink_output_used_as_wire"
firrtl.circuit "rhs_sink_output_used_as_wire" {
  // CHECK: module private @Bar
  module private @Bar(in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>, out %d: !firrtl.uint<1>) {
    strictconnect %c, %b : !firrtl.uint<1>
    %_c = wire  : !firrtl.uint<1>
    // CHECK: xor %a, %c
    %0 = xor %a, %c : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    strictconnect %_c, %0 : !firrtl.uint<1>
    strictconnect %d, %_c : !firrtl.uint<1>
  }
  module @rhs_sink_output_used_as_wire(in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>, out %c: !firrtl.uint<1>, out %d: !firrtl.uint<1>) {
    %bar_a, %bar_b, %bar_c, %bar_d = instance bar @Bar(in a: !firrtl.uint<1>, in b: !firrtl.uint<1>, out c: !firrtl.uint<1>, out d: !firrtl.uint<1>)
    strictconnect %bar_a, %a : !firrtl.uint<1>
    strictconnect %bar_b, %b : !firrtl.uint<1>
    strictconnect %c, %bar_c : !firrtl.uint<1>
    strictconnect %d, %bar_d : !firrtl.uint<1>
  }
}

// -----

// issue 1793
// Ensure don't touch on output port is seen by instances
firrtl.circuit "dntOutput" {
  // CHECK-LABEL: module @dntOutput
  // CHECK: %0 = mux(%c, %int_b, %c2_ui3)
  // CHECK-NEXT: strictconnect %b, %0
  module @dntOutput(out %b : !firrtl.uint<3>, in %c : !firrtl.uint<1>) {
    %const = constant 2 : !firrtl.uint<3>
    %int_b = instance int @foo(out b: !firrtl.uint<3>)
    %m = mux(%c, %int_b, %const) : (!firrtl.uint<1>, !firrtl.uint<3>, !firrtl.uint<3>) -> !firrtl.uint<3>
    strictconnect %b, %m : !firrtl.uint<3>
  }
  module private @foo(out %b: !firrtl.uint<3>  sym @dntSym1) {
    %const = constant 1 : !firrtl.uint<3>
    strictconnect %b, %const : !firrtl.uint<3>
  }
}

// -----

// An annotation should block removal of a wire, but should not block constant
// folding.
//
// CHECK-LABEL: "AnnotationsBlockRemoval"
firrtl.circuit "AnnotationsBlockRemoval"  {
  module @AnnotationsBlockRemoval(out %b: !firrtl.uint<1>) {
    %c1_ui1 = constant 1 : !firrtl.uint<1>
    // CHECK: %w = wire
    %w = wire droppable_name {annotations = [{class = "foo"}]} : !firrtl.uint<1>
    strictconnect %w, %c1_ui1 : !firrtl.uint<1>
    // CHECK: strictconnect %b, %c1_ui1
    strictconnect %b, %w : !firrtl.uint<1>
  }
}

// -----

// CHECK-LABEL: "Issue3372"
firrtl.circuit "Issue3372"  {
  module @Issue3372(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, out %value: !firrtl.uint<1>) {
    %c0_ui1 = constant 0 : !firrtl.uint<1>
    %c1_ui1 = constant 1 : !firrtl.uint<1>
    %other_zero = instance other interesting_name  @Other(out zero: !firrtl.uint<1>)
    %shared = regreset interesting_name %clock, %other_zero, %c1_ui1  : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>
    strictconnect %shared, %shared : !firrtl.uint<1>
    %test = wire interesting_name  : !firrtl.uint<1>
    strictconnect %test, %shared : !firrtl.uint<1>
    strictconnect %value, %test : !firrtl.uint<1>
  }
// CHECK:  %other_zero = instance other interesting_name @Other(out zero: !firrtl.uint<1>)
// CHECK:  %test = wire interesting_name : !firrtl.uint<1>
// CHECK:  strictconnect %value, %test : !firrtl.uint<1>

  module private @Other(out %zero: !firrtl.uint<1>) {
    %c0_ui1 = constant 0 : !firrtl.uint<1>
    strictconnect %zero, %c0_ui1 : !firrtl.uint<1>
  }
}

// -----

// CHECK-LABEL: "SendThroughRef"
firrtl.circuit "SendThroughRef" {
  module private @Bar(out %_a: !firrtl.probe<uint<1>>) {
    %zero = constant 0 : !firrtl.uint<1>
    %ref_zero = ref.send %zero : !firrtl.uint<1>
    ref.define %_a, %ref_zero : !firrtl.probe<uint<1>>
  }
  // CHECK:  strictconnect %a, %c0_ui1 : !firrtl.uint<1>
  module @SendThroughRef(out %a: !firrtl.uint<1>) {
    %bar_a = instance bar @Bar(out _a: !firrtl.probe<uint<1>>)
    %0 = ref.resolve %bar_a : !firrtl.probe<uint<1>>
    strictconnect %a, %0 : !firrtl.uint<1>
  }
}

// -----

// CHECK-LABEL: "ForwardRef"
firrtl.circuit "ForwardRef" {
  module private @RefForward2(out %_a: !firrtl.probe<uint<1>>) {
    %zero = constant 0 : !firrtl.uint<1>
    %ref_zero = ref.send %zero : !firrtl.uint<1>
    ref.define %_a, %ref_zero : !firrtl.probe<uint<1>>
  }
  module private @RefForward(out %_a: !firrtl.probe<uint<1>>) {
    %fwd_2 = instance fwd_2 @RefForward2(out _a: !firrtl.probe<uint<1>>)
    ref.define %_a, %fwd_2 : !firrtl.probe<uint<1>>
  }
  // CHECK:  strictconnect %a, %c0_ui1 : !firrtl.uint<1>
  module @ForwardRef(out %a: !firrtl.uint<1>) {
    %fwd_a = instance fwd @RefForward(out _a: !firrtl.probe<uint<1>>)
    %0 = ref.resolve %fwd_a : !firrtl.probe<uint<1>>
    strictconnect %a, %0 : !firrtl.uint<1>
  }
}

// -----

// Don't prop through a rwprobe ref.

// CHECK-LABEL: "SendThroughRWProbe"
firrtl.circuit "SendThroughRWProbe" {
  // CHECK-LABEL: module private @Bar
  module private @Bar(out %rw: !firrtl.rwprobe<uint<1>>, out %out : !firrtl.uint<1>) {
    %zero = constant 0 : !firrtl.uint<1>
    // CHECK: %[[N:.+]], %{{.+}} = node
    // CHECK-SAME: forceable
    %n, %n_ref = node %zero forceable : !firrtl.uint<1>
    // CHECK: node %[[N]]
    %user = node %n : !firrtl.uint<1>
    strictconnect %out, %user : !firrtl.uint<1>
    ref.define %rw, %n_ref : !firrtl.rwprobe<uint<1>>
  }
  // CHECK:  strictconnect %a, %0 : !firrtl.uint<1>
  module @SendThroughRWProbe(out %a: !firrtl.uint<1>, out %out: !firrtl.uint<1>) {
    %bar_rw, %bar_out = instance bar @Bar(out rw: !firrtl.rwprobe<uint<1>>, out out: !firrtl.uint<1>)
    strictconnect %out, %bar_out : !firrtl.uint<1>
    %0 = ref.resolve %bar_rw : !firrtl.rwprobe<uint<1>>
    strictconnect %a, %0 : !firrtl.uint<1>
  }
}

// -----

// Verbatim expressions should not be optimized away.
firrtl.circuit "Verbatim"  {
  module @Verbatim() {
    // CHECK: %[[v0:.+]] = verbatim.expr
    %0 = verbatim.expr "random.something" : () -> !firrtl.uint<1>
    // CHECK: %tap = wire   : !firrtl.uint<1>
    %tap = wire   : !firrtl.uint<1>
    %fizz = wire   {annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]} : !firrtl.uint<1>
    strictconnect %fizz, %tap : !firrtl.uint<1>
    // CHECK: strictconnect %tap, %[[v0]] : !firrtl.uint<1>
    strictconnect %tap, %0 : !firrtl.uint<1>
    // CHECK: verbatim.wire "randomBar.b"
    %1 = verbatim.wire "randomBar.b" : () -> !firrtl.uint<1> {symbols = []}
    // CHECK: %tap2 = wire   : !firrtl.uint<1>
    %tap2 = wire   : !firrtl.uint<1>
    strictconnect %tap2, %1 : !firrtl.uint<1>
  }
}

// -----

// This test is only checking that IMCP doesn't generate invalid IR.  IMCP needs
// to delete the strictconnect instead of replacing its destination with an
// invalid value that will replace the register.  For more information, see:
//   - https://github.com/llvm/circt/issues/4498
//
// CHECK-LABEL: "Issue4498"
firrtl.circuit "Issue4498"  {
  module @Issue4498(in %clock: !firrtl.clock) {
    %a = wire : !firrtl.uint<1>
    %r = reg interesting_name %clock : !firrtl.clock, !firrtl.uint<1>
    strictconnect %r, %a : !firrtl.uint<1>
  }
}

// -----

// An ordering dependnecy crept in with unwritten.  Check that it's gone
// CHECK-LABEL: "Ordering"
firrtl.circuit "Ordering" {
  module public @Ordering(out %b: !firrtl.uint<1>) {
    %0 = wire : !firrtl.uint<1>
    %c1_ui1 = constant 1 : !firrtl.uint<1>
    strictconnect %0, %c1_ui1 : !firrtl.uint<1>
    %1 = xor %0, %c1_ui1 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    strictconnect %b, %1 : !firrtl.uint<1>
  }
  // CHECK: strictconnect %b, %c0_ui1
}

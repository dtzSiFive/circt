// RUN: circt-opt -canonicalize='top-down=true region-simplify=true' %s | FileCheck %s

firrtl.circuit "Casts" {

// CHECK-LABEL: module @Casts
firrtl.module @Casts(in %ui1 : !firrtl.uint<1>, in %si1 : !firrtl.sint<1>,
    in %clock : !firrtl.clock, in %asyncreset : !firrtl.asyncreset,
    in %inreset : !firrtl.reset, out %outreset : !firrtl.reset,
    out %out_ui1 : !firrtl.uint<1>, out %out_si1 : !firrtl.sint<1>,
    out %out_clock : !firrtl.clock, out %out_asyncreset : !firrtl.asyncreset) {

  %c1_ui1 = constant 1 : !firrtl.uint<1>
  %c1_si1 = constant 1 : !firrtl.sint<1>
  %invalid_ui1 = invalidvalue : !firrtl.uint<1>
  %invalid_si1 = invalidvalue : !firrtl.sint<1>
  %invalid_clock = invalidvalue : !firrtl.clock
  %invalid_asyncreset = invalidvalue : !firrtl.asyncreset

  // No effect
  // CHECK: strictconnect %out_ui1, %ui1 : !firrtl.uint<1>
  %0 = asUInt %ui1 : (!firrtl.uint<1>) -> !firrtl.uint<1>
  connect %out_ui1, %0 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK: strictconnect %out_si1, %si1 : !firrtl.sint<1>
  %1 = asSInt %si1 : (!firrtl.sint<1>) -> !firrtl.sint<1>
  connect %out_si1, %1 : !firrtl.sint<1>, !firrtl.sint<1>
  // CHECK: strictconnect %out_clock, %clock : !firrtl.clock
  %2 = asClock %clock : (!firrtl.clock) -> !firrtl.clock
  connect %out_clock, %2 : !firrtl.clock, !firrtl.clock
  // CHECK: strictconnect %out_asyncreset, %asyncreset : !firrtl.asyncreset
  %3 = asAsyncReset %asyncreset : (!firrtl.asyncreset) -> !firrtl.asyncreset
  connect %out_asyncreset, %3 : !firrtl.asyncreset, !firrtl.asyncreset

  // Constant fold.
  // CHECK: strictconnect %out_ui1, %c1_ui1 : !firrtl.uint<1>
  %4 = asUInt %c1_si1 : (!firrtl.sint<1>) -> !firrtl.uint<1>
  connect %out_ui1, %4 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK: strictconnect %out_si1, %c-1_si1 : !firrtl.sint<1>
  %5 = asSInt %c1_ui1 : (!firrtl.uint<1>) -> !firrtl.sint<1>
  connect %out_si1, %5 : !firrtl.sint<1>, !firrtl.sint<1>
  // CHECK: strictconnect %out_clock, %c1_clock : !firrtl.clock
  %6 = asClock %c1_ui1 : (!firrtl.uint<1>) -> !firrtl.clock
  connect %out_clock, %6 : !firrtl.clock, !firrtl.clock
  // CHECK: strictconnect %out_asyncreset, %c1_asyncreset : !firrtl.asyncreset
  %7 = asAsyncReset %c1_ui1 : (!firrtl.uint<1>) -> !firrtl.asyncreset
  connect %out_asyncreset, %7 : !firrtl.asyncreset, !firrtl.asyncreset
  // CHECK: strictconnect %outreset, %inreset : !firrtl.reset
  %8 = resetCast %inreset : (!firrtl.reset) -> !firrtl.reset
  strictconnect %outreset, %8 : !firrtl.reset
}

// CHECK-LABEL: module @Div
firrtl.module @Div(in %a: !firrtl.uint<4>,
                   out %b: !firrtl.uint<4>,
                   in %c: !firrtl.sint<4>,
                   out %d: !firrtl.sint<5>,
                   in %e: !firrtl.uint,
                   out %f: !firrtl.uint,
                   in %g: !firrtl.sint,
                   out %h: !firrtl.sint,
                   out %i: !firrtl.uint<4>) {

  // CHECK-DAG: [[ONE_i4:%.+]] = constant 1 : !firrtl.uint<4>
  // CHECK-DAG: [[ONE_s5:%.+]] = constant 1 : !firrtl.sint<5>
  // CHECK-DAG: [[ONE_i2:%.+]] = constant 1 : !firrtl.uint
  // CHECK-DAG: [[ONE_s2:%.+]] = constant 1 : !firrtl.sint

  // Check that 'div(a, a) -> 1' works for known UInt widths.
  // CHECK: strictconnect %b, [[ONE_i4]]
  %0 = div %a, %a : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  connect %b, %0 : !firrtl.uint<4>, !firrtl.uint<4>

  // Check that 'div(c, c) -> 1' works for known SInt widths.
  // CHECK: strictconnect %d, [[ONE_s5]] : !firrtl.sint<5>
  %1 = div %c, %c : (!firrtl.sint<4>, !firrtl.sint<4>) -> !firrtl.sint<5>
  connect %d, %1 : !firrtl.sint<5>, !firrtl.sint<5>

  // Check that 'div(e, e) -> 1' works for unknown UInt widths.
  // CHECK: connect %f, [[ONE_i2]]
  %2 = div %e, %e : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
  connect %f, %2 : !firrtl.uint, !firrtl.uint

  // Check that 'div(g, g) -> 1' works for unknown SInt widths.
  // CHECK: connect %h, [[ONE_s2]]
  %3 = div %g, %g : (!firrtl.sint, !firrtl.sint) -> !firrtl.sint
  connect %h, %3 : !firrtl.sint, !firrtl.sint

  // Check that 'div(a, 1) -> a' for known UInt widths.
  // CHECK: strictconnect %b, %a
  %c1_ui2 = constant 1 : !firrtl.uint<2>
  %4 = div %a, %c1_ui2 : (!firrtl.uint<4>, !firrtl.uint<2>) -> !firrtl.uint<4>
  connect %b, %4 : !firrtl.uint<4>, !firrtl.uint<4>

  // CHECK: strictconnect %i, %c5_ui4
  %c1_ui4 = constant 15 : !firrtl.uint<4>
  %c3_ui4 = constant 3 : !firrtl.uint<4>
  %5 = div %c1_ui4, %c3_ui4 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  connect %i, %5 : !firrtl.uint<4>, !firrtl.uint<4>
}

// CHECK-LABEL: module @And
firrtl.module @And(in %in: !firrtl.uint<4>,
                   in %in6: !firrtl.uint<6>,
                   in %sin: !firrtl.sint<4>,
                   in %zin1: !firrtl.uint<0>,
                   in %zin2: !firrtl.uint<0>,
                   out %out: !firrtl.uint<4>,
                   out %out6: !firrtl.uint<6>,
                   out %outz: !firrtl.uint<0>) {
  // CHECK: strictconnect %out, %c1_ui4
  %c1_ui4 = constant 1 : !firrtl.uint<4>
  %c3_ui4 = constant 3 : !firrtl.uint<4>
  %0 = and %c3_ui4, %c1_ui4 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  connect %out, %0 : !firrtl.uint<4>, !firrtl.uint<4>

  // CHECK: strictconnect %out, %in
  %c15_ui4 = constant 15 : !firrtl.uint<4>
  %1 = and %in, %c15_ui4 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  connect %out, %1 : !firrtl.uint<4>, !firrtl.uint<4>

  // CHECK: strictconnect %out, %c0_ui4
  %c1_ui0 = constant 0 : !firrtl.uint<4>
  %2 = and %in, %c1_ui0 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  connect %out, %2 : !firrtl.uint<4>, !firrtl.uint<4>

  // CHECK: strictconnect %out, %c0_ui4
  %inv_2 = and %c1_ui0, %in : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  connect %out, %inv_2 : !firrtl.uint<4>, !firrtl.uint<4>

  // CHECK: strictconnect %out, %in
  %3 = and %in, %in : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  connect %out, %3 : !firrtl.uint<4>, !firrtl.uint<4>

  // CHECK: strictconnect %out, %c0_ui4
  // CHECK: strictconnect %outz, %c0_ui0
  %zw = and %zin1, %zin2 : (!firrtl.uint<0>, !firrtl.uint<0>) -> !firrtl.uint<0>
  connect %out, %zw : !firrtl.uint<4>, !firrtl.uint<0>
  strictconnect %outz, %zw : !firrtl.uint<0>

  // Mixed type inputs - the constant is zero extended, not sign extended, so it
  // cannot be folded!

  // Narrows, then folds away
  // CHECK: %0 = bits %in 1 to 0 : (!firrtl.uint<4>) -> !firrtl.uint<2>
  // CHECK-NEXT: %1 = pad %0, 4 : (!firrtl.uint<2>) -> !firrtl.uint<4>
  // CHECK-NEXT: strictconnect %out, %1
  %c3_ui2 = constant 3 : !firrtl.uint<2>
  %4 = and %in, %c3_ui2 : (!firrtl.uint<4>, !firrtl.uint<2>) -> !firrtl.uint<4>
  connect %out, %4 : !firrtl.uint<4>, !firrtl.uint<4>

  // Mixed type input and outputs.

  // CHECK: strictconnect %out, %c1_ui4
  %c1_si4 = constant 1 : !firrtl.sint<4>
  %5 = and %c1_si4, %c1_si4 : (!firrtl.sint<4>, !firrtl.sint<4>) -> !firrtl.uint<4>
  connect %out, %5 : !firrtl.uint<4>, !firrtl.uint<4>

  // CHECK: %[[AND:.+]] = asUInt %sin
  // CHECK-NEXT: strictconnect %out, %[[AND]]
  %6 = and %sin, %sin : (!firrtl.sint<4>, !firrtl.sint<4>) -> !firrtl.uint<4>
  connect %out, %6 : !firrtl.uint<4>, !firrtl.uint<4>

  // CHECK: strictconnect %out, %c0_ui4
  %c0_si2 = constant 0 : !firrtl.sint<2>
  %7 = and %sin, %c0_si2 : (!firrtl.sint<4>, !firrtl.sint<2>) -> !firrtl.uint<4>
  connect %out, %7 : !firrtl.uint<4>, !firrtl.uint<4>

  // CHECK: %[[trunc:.*]] = bits %in6
  // CHECK: %[[ANDPAD:.*]] = and %[[trunc]], %in
  // CHECK: %[[POST:.*]] = pad %[[ANDPAD]]
  // CHECK: strictconnect %out6, %[[POST]]
  %8 = pad %in, 6 : (!firrtl.uint<4>) -> !firrtl.uint<6>
  %9 = and %in6, %8  : (!firrtl.uint<6>, !firrtl.uint<6>) -> !firrtl.uint<6>
  connect %out6, %9 : !firrtl.uint<6>, !firrtl.uint<6>
}

// CHECK-LABEL: module @Or
firrtl.module @Or(in %in: !firrtl.uint<4>,
                  in %in6: !firrtl.uint<6>,
                  in %sin: !firrtl.sint<4>,
                  in %zin1: !firrtl.uint<0>,
                  in %zin2: !firrtl.uint<0>,
                  out %out: !firrtl.uint<4>,
                  out %out6: !firrtl.uint<6>,
                  out %outz: !firrtl.uint<0>) {
  // CHECK: strictconnect %out, %c7_ui4
  %c4_ui4 = constant 4 : !firrtl.uint<4>
  %c3_ui4 = constant 3 : !firrtl.uint<4>
  %0 = or %c3_ui4, %c4_ui4 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  connect %out, %0 : !firrtl.uint<4>, !firrtl.uint<4>

  // CHECK: strictconnect %out, %c15_ui4
  %c1_ui15 = constant 15 : !firrtl.uint<4>
  %1 = or %in, %c1_ui15 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  connect %out, %1 : !firrtl.uint<4>, !firrtl.uint<4>

  // CHECK: strictconnect %out, %in
  %c1_ui0 = constant 0 : !firrtl.uint<4>
  %2 = or %in, %c1_ui0 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  connect %out, %2 : !firrtl.uint<4>, !firrtl.uint<4>

  // CHECK: strictconnect %out, %in
  %inv_2 = or %c1_ui0, %in : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  connect %out, %inv_2 : !firrtl.uint<4>, !firrtl.uint<4>

  // CHECK: strictconnect %out, %in
  %3 = or %in, %in : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  connect %out, %3 : !firrtl.uint<4>, !firrtl.uint<4>

  // CHECK: strictconnect %out, %c0_ui4
  // CHECK: strictconnect %outz, %c0_ui0
  %zw = or %zin1, %zin2 : (!firrtl.uint<0>, !firrtl.uint<0>) -> !firrtl.uint<0>
  connect %out, %zw : !firrtl.uint<4>, !firrtl.uint<0>
  strictconnect %outz, %zw : !firrtl.uint<0>

  // Mixed type input and outputs.

  // CHECK: strictconnect %out, %c1_ui4
  %c1_si4 = constant 1 : !firrtl.sint<4>
  %5 = or %c1_si4, %c1_si4 : (!firrtl.sint<4>, !firrtl.sint<4>) -> !firrtl.uint<4>
  connect %out, %5 : !firrtl.uint<4>, !firrtl.uint<4>

  // CHECK: [[OR:%.+]] = asUInt %sin
  // CHECK-NEXT: strictconnect %out, [[OR]]
  %6 = or %sin, %sin : (!firrtl.sint<4>, !firrtl.sint<4>) -> !firrtl.uint<4>
  connect %out, %6 : !firrtl.uint<4>, !firrtl.uint<4>

  // CHECK: strictconnect %out, %c15_ui4
  %c0_si2 = constant -1 : !firrtl.sint<2>
  %7 = or %sin, %c0_si2 : (!firrtl.sint<4>, !firrtl.sint<2>) -> !firrtl.uint<4>
  connect %out, %7 : !firrtl.uint<4>, !firrtl.uint<4>

  // CHECK: %[[trunc:.*]] = bits %in6
  // CHECK: %[[trunc2:.*]] = bits %in6
  // CHECK: %[[OR:.*]] = or %[[trunc2]], %in
  // CHECK: %[[CAT:.*]] = cat %[[trunc]], %[[OR]]
  // CHECK: strictconnect %out6, %[[CAT]]
  %8 = pad %in, 6 : (!firrtl.uint<4>) -> !firrtl.uint<6>
  %9 = or %in6, %8  : (!firrtl.uint<6>, !firrtl.uint<6>) -> !firrtl.uint<6>
  connect %out6, %9 : !firrtl.uint<6>, !firrtl.uint<6>

}

// CHECK-LABEL: module @Xor
firrtl.module @Xor(in %in: !firrtl.uint<4>,
                   in %in6: !firrtl.uint<6>,
                   in %sin: !firrtl.sint<4>,
                   in %zin1: !firrtl.uint<0>,
                   in %zin2: !firrtl.uint<0>,
                   out %out: !firrtl.uint<4>,
                   out %out6: !firrtl.uint<6>,
                   out %outz: !firrtl.uint<0>) {
  // CHECK: strictconnect %out, %c2_ui4
  %c1_ui4 = constant 1 : !firrtl.uint<4>
  %c3_ui4 = constant 3 : !firrtl.uint<4>
  %0 = xor %c3_ui4, %c1_ui4 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  connect %out, %0 : !firrtl.uint<4>, !firrtl.uint<4>

  // CHECK: strictconnect %out, %in
  %c1_ui0 = constant 0 : !firrtl.uint<4>
  %2 = xor %in, %c1_ui0 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  connect %out, %2 : !firrtl.uint<4>, !firrtl.uint<4>

  // CHECK: strictconnect %out, %c0_ui4
  %3 = xor %in, %in : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  connect %out, %3 : !firrtl.uint<4>, !firrtl.uint<4>

  // CHECK: strictconnect %out, %c0_ui4
  // CHECK: strictconnect %outz, %c0_ui0
  %zw = xor %zin1, %zin2 : (!firrtl.uint<0>, !firrtl.uint<0>) -> !firrtl.uint<0>
  connect %out, %zw : !firrtl.uint<4>, !firrtl.uint<0>
  strictconnect %outz, %zw : !firrtl.uint<0>

  // Mixed type input and outputs.

  // CHECK: strictconnect %out, %c0_ui4
  %6 = xor %sin, %sin : (!firrtl.sint<4>, !firrtl.sint<4>) -> !firrtl.uint<4>
  connect %out, %6 : !firrtl.uint<4>, !firrtl.uint<4>

  // CHECK: %[[aui:.*]] = asUInt %sin
  // CHECK: strictconnect %out, %[[aui]]
  %c0_si2 = constant 0 : !firrtl.sint<2>
  %7 = xor %sin, %c0_si2 : (!firrtl.sint<4>, !firrtl.sint<2>) -> !firrtl.uint<4>
  connect %out, %7 : !firrtl.uint<4>, !firrtl.uint<4>

  // CHECK: %[[trunc:.*]] = bits %in6
  // CHECK: %[[trunc2:.*]] = bits %in6
  // CHECK: %[[XOR:.*]] = xor %[[trunc2]], %in
  // CHECK: %[[CAT:.*]] = cat %[[trunc]], %[[XOR]]
  // CHECK: strictconnect %out6, %[[CAT]]
  %8 = pad %in, 6 : (!firrtl.uint<4>) -> !firrtl.uint<6>
  %9 = xor %in6, %8  : (!firrtl.uint<6>, !firrtl.uint<6>) -> !firrtl.uint<6>
  connect %out6, %9 : !firrtl.uint<6>, !firrtl.uint<6>

}

// CHECK-LABEL: module @Not
firrtl.module @Not(in %in: !firrtl.uint<4>,
                   in %sin: !firrtl.sint<4>,
                   out %outu: !firrtl.uint<4>,
                   out %outs: !firrtl.uint<4>) {
  %0 = not %in : (!firrtl.uint<4>) -> !firrtl.uint<4>
  %1 = not %0 : (!firrtl.uint<4>) -> !firrtl.uint<4>
  connect %outu, %1 : !firrtl.uint<4>, !firrtl.uint<4>
  %2 = not %sin : (!firrtl.sint<4>) -> !firrtl.uint<4>
  %3 = not %2 : (!firrtl.uint<4>) -> !firrtl.uint<4>
  connect %outs, %3 : !firrtl.uint<4>, !firrtl.uint<4>
  // CHECK: strictconnect %outu, %in
  // CHECK: %[[cast:.*]] = asUInt %sin
  // CHECK: strictconnect %outs, %[[cast]]
}

// CHECK-LABEL: module @EQ
firrtl.module @EQ(in %in1: !firrtl.uint<1>,
                  in %in4: !firrtl.uint<4>,
                  out %out: !firrtl.uint<1>) {
  // CHECK: strictconnect %out, %in1
  %c1_ui1 = constant 1 : !firrtl.uint<1>
  %0 = eq %in1, %c1_ui1 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
  connect %out, %0 : !firrtl.uint<1>, !firrtl.uint<1>

  // Issue #368: https://github.com/llvm/circt/issues/368
  %c3_ui2 = constant 3 : !firrtl.uint<2>
  %1 = eq %in1, %c3_ui2 : (!firrtl.uint<1>, !firrtl.uint<2>) -> !firrtl.uint<1>
  connect %out, %1 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK: eq %in1, %c3_ui2
  // CHECK-NEXT: strictconnect

  %c0_ui1 = constant 0 : !firrtl.uint<1>
  %2 = eq %in1, %c0_ui1 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
  connect %out, %2 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK-NEXT: not %in1
  // CHECK-NEXT: strictconnect

  %c15_ui4 = constant 15 : !firrtl.uint<4>
  %3 = eq %in4, %c15_ui4 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<1>
  connect %out, %3 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK: andr %in4
  // CHECK-NEXT: strictconnect

  %4 = eq %in4, %c0_ui1 : (!firrtl.uint<4>, !firrtl.uint<1>) -> !firrtl.uint<1>
  connect %out, %4 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK: [[ORR:%.+]] = orr %in4
  // CHECK-NEXT: not [[ORR]]
  // CHECK-NEXT: strictconnect
}

// CHECK-LABEL: module @NEQ
firrtl.module @NEQ(in %in1: !firrtl.uint<1>,
                   in %in4: !firrtl.uint<4>,
                   out %out: !firrtl.uint<1>) {
  // CHECK: strictconnect %out, %in
  %c0_ui1 = constant 0 : !firrtl.uint<1>
  %0 = neq %in1, %c0_ui1 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
  connect %out, %0 : !firrtl.uint<1>, !firrtl.uint<1>

  %c1_ui1 = constant 1 : !firrtl.uint<1>
  %1 = neq %in1, %c1_ui1 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
  connect %out, %1 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK-NEXT: not %in1
  // CHECK-NEXT: strictconnect

  %2 = neq %in4, %c0_ui1 : (!firrtl.uint<4>, !firrtl.uint<1>) -> !firrtl.uint<1>
  connect %out, %2 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK: orr %in4
  // CHECK-NEXT: strictconnect

  %c15_ui4 = constant 15 : !firrtl.uint<4>
  %4 = neq %in4, %c15_ui4 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<1>
  connect %out, %4 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK: [[ANDR:%.+]] = andr %in4
  // CHECK-NEXT: not [[ANDR]]
  // CHECK-NEXT: strictconnect
}

// CHECK-LABEL: module @Cat
firrtl.module @Cat(in %in4: !firrtl.uint<4>,
                   in %sin4: !firrtl.sint<4>,
                   out %out4: !firrtl.uint<4>,
                   out %outcst: !firrtl.uint<8>,
                   out %outcst2: !firrtl.uint<8>,
                   in %in0 : !firrtl.uint<0>,
                   out %outpt1: !firrtl.uint<4>,
                   out %outpt2 : !firrtl.uint<4>) {
  %c0_ui2 = constant 0 : !firrtl.uint<2>
  %c0_si2 = constant 0 : !firrtl.sint<2>

  // CHECK: strictconnect %out4, %in4
  %0 = bits %in4 3 to 2 : (!firrtl.uint<4>) -> !firrtl.uint<2>
  %1 = bits %in4 1 to 0 : (!firrtl.uint<4>) -> !firrtl.uint<2>
  %2 = cat %0, %1 : (!firrtl.uint<2>, !firrtl.uint<2>) -> !firrtl.uint<4>
  connect %out4, %2 : !firrtl.uint<4>, !firrtl.uint<4>

  // CHECK: strictconnect %outcst, %c243_ui8
  %c15_ui4 = constant 15 : !firrtl.uint<4>
  %c3_ui4 = constant 3 : !firrtl.uint<4>
  %3 = cat %c15_ui4, %c3_ui4 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<8>
  connect %outcst, %3 : !firrtl.uint<8>, !firrtl.uint<8>

  // CHECK: strictconnect %outpt1, %in4
  %5 = cat %in0, %in4 : (!firrtl.uint<0>, !firrtl.uint<4>) -> !firrtl.uint<4>
  connect %outpt1, %5 : !firrtl.uint<4>, !firrtl.uint<4>
  // CHECK: strictconnect %outpt2, %in4
  %6 = cat %in4, %in0 : (!firrtl.uint<4>, !firrtl.uint<0>) -> !firrtl.uint<4>
  connect %outpt2, %6 : !firrtl.uint<4>, !firrtl.uint<4>

  // CHECK: cat %c0_ui4, %in4
  %7 = cat %c0_ui2, %in4 : (!firrtl.uint<2>, !firrtl.uint<4>) -> !firrtl.uint<6>
  %8 = cat %c0_ui2, %7 : (!firrtl.uint<2>, !firrtl.uint<6>) -> !firrtl.uint<8>
  connect %outcst, %8 : !firrtl.uint<8>, !firrtl.uint<8>

  // CHECK: asUInt %sin4
  // CHECK-NEXT: cat %c0_ui4
  %9  = cat %c0_si2, %sin4 : (!firrtl.sint<2>, !firrtl.sint<4>) -> !firrtl.uint<6>
  %10 = cat %c0_ui2, %9 : (!firrtl.uint<2>, !firrtl.uint<6>) -> !firrtl.uint<8>
  connect %outcst, %10 : !firrtl.uint<8>, !firrtl.uint<8>
}

// CHECK-LABEL: module @Bits
firrtl.module @Bits(in %in1: !firrtl.uint<1>,
                    in %in4: !firrtl.uint<4>,
                    out %out1: !firrtl.uint<1>,
                    out %out2: !firrtl.uint<2>,
                    out %out4: !firrtl.uint<4>) {
  // CHECK: strictconnect %out1, %in1
  %0 = bits %in1 0 to 0 : (!firrtl.uint<1>) -> !firrtl.uint<1>
  connect %out1, %0 : !firrtl.uint<1>, !firrtl.uint<1>

  // CHECK: strictconnect %out4, %in4
  %1 = bits %in4 3 to 0 : (!firrtl.uint<4>) -> !firrtl.uint<4>
  connect %out4, %1 : !firrtl.uint<4>, !firrtl.uint<4>

  // CHECK: strictconnect %out2, %c1_ui2
  %c10_ui4 = constant 10 : !firrtl.uint<4>
  %2 = bits %c10_ui4 2 to 1 : (!firrtl.uint<4>) -> !firrtl.uint<2>
  connect %out2, %2 : !firrtl.uint<2>, !firrtl.uint<2>


  // CHECK: bits %in4 2 to 2 : (!firrtl.uint<4>) -> !firrtl.uint<1>
  // CHECK-NEXT: strictconnect %out1, %
  %3 = bits %in4 3 to 1 : (!firrtl.uint<4>) -> !firrtl.uint<3>
  %4 = bits %3 1 to 1 : (!firrtl.uint<3>) -> !firrtl.uint<1>
  connect %out1, %4 : !firrtl.uint<1>, !firrtl.uint<1>

  // CHECK: strictconnect %out1, %in1
  %5 = bits %in1 0 to 0 : (!firrtl.uint<1>) -> !firrtl.uint<1>
  connect %out1, %5 : !firrtl.uint<1>, !firrtl.uint<1>
}

// CHECK-LABEL: module @Head
firrtl.module @Head(in %in4u: !firrtl.uint<4>,
                    out %out1u: !firrtl.uint<1>,
                    out %out3u: !firrtl.uint<3>) {
  // CHECK: [[BITS:%.+]] = bits %in4u 3 to 3
  // CHECK-NEXT: strictconnect %out1u, [[BITS]]
  %0 = head %in4u, 1 : (!firrtl.uint<4>) -> !firrtl.uint<1>
  connect %out1u, %0 : !firrtl.uint<1>, !firrtl.uint<1>

  // CHECK: [[BITS:%.+]] = bits %in4u 3 to 1
  // CHECK-NEXT: strictconnect %out3u, [[BITS]]
  %1 = head %in4u, 3 : (!firrtl.uint<4>) -> !firrtl.uint<3>
  connect %out3u, %1 : !firrtl.uint<3>, !firrtl.uint<3>

  // CHECK: strictconnect %out3u, %c5_ui3
  %c10_ui4 = constant 10 : !firrtl.uint<4>
  %2 = head %c10_ui4, 3 : (!firrtl.uint<4>) -> !firrtl.uint<3>
  connect %out3u, %2 : !firrtl.uint<3>, !firrtl.uint<3>
}

// CHECK-LABEL: module @Mux
firrtl.module @Mux(in %in: !firrtl.uint<4>,
                   in %cond: !firrtl.uint<1>,
                   in %val1: !firrtl.uint<1>,
                   in %val2: !firrtl.uint<1>,
                   in %val0: !firrtl.uint<0>,
                   out %out: !firrtl.uint<4>,
                   out %out1: !firrtl.uint<1>,
                   out %out2: !firrtl.uint<0>,
                   out %out3: !firrtl.uint<1>,
                   out %out4: !firrtl.uint<4>) {
  // CHECK: strictconnect %out, %in
  %0 = mux (%cond, %in, %in) : (!firrtl.uint<1>, !firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  connect %out, %0 : !firrtl.uint<4>, !firrtl.uint<4>

  // CHECK: strictconnect %out, %c7_ui4
  %c7_ui4 = constant 7 : !firrtl.uint<4>
  %c0_ui1 = constant 0 : !firrtl.uint<1>
  %2 = mux (%c0_ui1, %in, %c7_ui4) : (!firrtl.uint<1>, !firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  connect %out, %2 : !firrtl.uint<4>, !firrtl.uint<4>

  // CHECK: strictconnect %out1, %cond
  %c1_ui1 = constant 1 : !firrtl.uint<1>
  %3 = mux (%cond, %c1_ui1, %c0_ui1) : (!firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
  connect %out1, %3 : !firrtl.uint<1>, !firrtl.uint<1>

  // CHECK: strictconnect %out, %invalid_ui4
  %invalid_ui4 = invalidvalue : !firrtl.uint<4>
  %7 = mux (%cond, %invalid_ui4, %invalid_ui4) : (!firrtl.uint<1>, !firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  connect %out, %7 : !firrtl.uint<4>, !firrtl.uint<4>

  %9 = multibit_mux %c1_ui1, %c0_ui1, %cond : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK-NEXT: strictconnect %out1, %c0_ui1
  connect %out1, %9 : !firrtl.uint<1>, !firrtl.uint<1>

  %10 = multibit_mux %cond, %val1, %val2 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK-NEXT: %[[MUX:.+]] = mux(%cond, %val1, %val2)
  // CHECK-NEXT: strictconnect %out1, %[[MUX]]
  connect %out1, %10 : !firrtl.uint<1>, !firrtl.uint<1>

  %11 = multibit_mux %cond, %val1, %val1, %val1 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK-NEXT: strictconnect %out1, %val1
  connect %out1, %11 : !firrtl.uint<1>, !firrtl.uint<1>

  %c0_ui0 = constant 0 : !firrtl.uint<0>
  %12 = multibit_mux %c0_ui0, %val1, %val1 :!firrtl.uint<0>, !firrtl.uint<1>
  // CHECK-NEXT: strictconnect %out1, %val1
  connect %out1, %12 : !firrtl.uint<1>, !firrtl.uint<1>

  %13 = mux (%cond, %val0, %val0) : (!firrtl.uint<1>, !firrtl.uint<0>, !firrtl.uint<0>) -> !firrtl.uint<0>
  // CHECK-NEXT: strictconnect %out2, %c0_ui0
  strictconnect %out2, %13 : !firrtl.uint<0>

  %14 = mux (%cond, %c0_ui1, %c1_ui1) : (!firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
  // CHECK-NEXT: [[V1:%.+]] = not %cond
  // CHECK-NEXT: strictconnect %out3, [[V1]]
  connect %out3, %14 : !firrtl.uint<1>, !firrtl.uint<1>

  %c0_ui4 = constant 0 : !firrtl.uint<4>
  %c1_ui4 = constant 1 : !firrtl.uint<4>
  %15 = mux (%cond, %c0_ui4, %c1_ui4) : (!firrtl.uint<1>, !firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  // CHECK-NEXT: [[V2:%.+]] = mux(%cond
  // CHECK-NEXT: strictconnect %out4, [[V2]]
  connect %out4, %15 : !firrtl.uint<4>, !firrtl.uint<4>
}

// CHECK-LABEL: module @Pad
firrtl.module @Pad(in %in1u: !firrtl.uint<1>,
                   out %out1u: !firrtl.uint<1>,
                   out %outu: !firrtl.uint<4>,
                   out %outs: !firrtl.sint<4>) {
  // CHECK: strictconnect %out1u, %in1u
  %0 = pad %in1u, 1 : (!firrtl.uint<1>) -> !firrtl.uint<1>
  connect %out1u, %0 : !firrtl.uint<1>, !firrtl.uint<1>

  // CHECK: strictconnect %outu, %c1_ui4
  %c1_ui1 = constant 1 : !firrtl.uint<1>
  %1 = pad %c1_ui1, 4 : (!firrtl.uint<1>) -> !firrtl.uint<4>
  connect %outu, %1 : !firrtl.uint<4>, !firrtl.uint<4>

  // CHECK: strictconnect %outs, %c-1_si4
  %c1_si1 = constant -1 : !firrtl.sint<1>
  %2 = pad %c1_si1, 4 : (!firrtl.sint<1>) -> !firrtl.sint<4>
  connect %outs, %2 : !firrtl.sint<4>, !firrtl.sint<4>
}

// CHECK-LABEL: module @Shl
firrtl.module @Shl(in %in1u: !firrtl.uint<1>,
                   out %out1u: !firrtl.uint<1>,
                   out %outu: !firrtl.uint<4>) {
  // CHECK: strictconnect %out1u, %in1u
  %0 = shl %in1u, 0 : (!firrtl.uint<1>) -> !firrtl.uint<1>
  connect %out1u, %0 : !firrtl.uint<1>, !firrtl.uint<1>

  // CHECK: strictconnect %outu, %c8_ui4
  %c1_ui1 = constant 1 : !firrtl.uint<1>
  %1 = shl %c1_ui1, 3 : (!firrtl.uint<1>) -> !firrtl.uint<4>
  connect %outu, %1 : !firrtl.uint<4>, !firrtl.uint<4>
}

// CHECK-LABEL: module @Shr
firrtl.module @Shr(in %in1u: !firrtl.uint<1>,
                   in %in4u: !firrtl.uint<4>,
                   in %in1s: !firrtl.sint<1>,
                   in %in4s: !firrtl.sint<4>,
                   in %in0u: !firrtl.uint<0>,
                   out %out1s: !firrtl.sint<1>,
                   out %out1u: !firrtl.uint<1>,
                   out %outu: !firrtl.uint<4>) {
  // CHECK: strictconnect %out1u, %in1u
  %0 = shr %in1u, 0 : (!firrtl.uint<1>) -> !firrtl.uint<1>
  connect %out1u, %0 : !firrtl.uint<1>, !firrtl.uint<1>

  // CHECK: strictconnect %out1u, %c0_ui1
  %1 = shr %in4u, 4 : (!firrtl.uint<4>) -> !firrtl.uint<1>
  connect %out1u, %1 : !firrtl.uint<1>, !firrtl.uint<1>

  // CHECK: strictconnect %out1u, %c0_ui1
  %2 = shr %in4u, 5 : (!firrtl.uint<4>) -> !firrtl.uint<1>
  connect %out1u, %2 : !firrtl.uint<1>, !firrtl.uint<1>

  // CHECK: [[BITS:%.+]] = bits %in4s 3 to 3
  // CHECK-NEXT: [[CAST:%.+]] = asSInt [[BITS]]
  // CHECK-NEXT: strictconnect %out1s, [[CAST]]
  %3 = shr %in4s, 3 : (!firrtl.sint<4>) -> !firrtl.sint<1>
  connect %out1s, %3 : !firrtl.sint<1>, !firrtl.sint<1>

  // CHECK: [[BITS:%.+]] = bits %in4s 3 to 3
  // CHECK-NEXT: [[CAST:%.+]] = asSInt [[BITS]]
  // CHECK-NEXT: strictconnect %out1s, [[CAST]]
  %4 = shr %in4s, 4 : (!firrtl.sint<4>) -> !firrtl.sint<1>
  connect %out1s, %4 : !firrtl.sint<1>, !firrtl.sint<1>

  // CHECK: [[BITS:%.+]] = bits %in4s 3 to 3
  // CHECK-NEXT: [[CAST:%.+]] = asSInt [[BITS]]
  // CHECK-NEXT: strictconnect %out1s, [[CAST]]
  %5 = shr %in4s, 5 : (!firrtl.sint<4>) -> !firrtl.sint<1>
  connect %out1s, %5 : !firrtl.sint<1>, !firrtl.sint<1>

  // CHECK: strictconnect %out1u, %c1_ui1
  %c12_ui4 = constant 12 : !firrtl.uint<4>
  %6 = shr %c12_ui4, 3 : (!firrtl.uint<4>) -> !firrtl.uint<1>
  connect %out1u, %6 : !firrtl.uint<1>, !firrtl.uint<1>

  // CHECK: [[BITS:%.+]] = bits %in4u 3 to 3
  // CHECK-NEXT: strictconnect %out1u, [[BITS]]
  %7 = shr %in4u, 3 : (!firrtl.uint<4>) -> !firrtl.uint<1>
  connect %out1u, %7 : !firrtl.uint<1>, !firrtl.uint<1>

  // Issue #313: https://github.com/llvm/circt/issues/313
  // CHECK: strictconnect %out1s, %in1s : !firrtl.sint<1>
  %8 = shr %in1s, 42 : (!firrtl.sint<1>) -> !firrtl.sint<1>
  connect %out1s, %8 : !firrtl.sint<1>, !firrtl.sint<1>

  // Issue #1064: https://github.com/llvm/circt/issues/1064
  // CHECK: strictconnect %out1u, %c0_ui1
  %c1_ui1 = constant 1 : !firrtl.uint<1>
  %9 = dshr %in0u, %c1_ui1 : (!firrtl.uint<0>, !firrtl.uint<1>) -> !firrtl.uint<0>
  connect %out1u, %9 : !firrtl.uint<1>, !firrtl.uint<0>
}

// CHECK-LABEL: module @Tail
firrtl.module @Tail(in %in4u: !firrtl.uint<4>,
                    out %out1u: !firrtl.uint<1>,
                    out %out3u: !firrtl.uint<3>) {
  // CHECK: [[BITS:%.+]] = bits %in4u 0 to 0
  // CHECK-NEXT: strictconnect %out1u, [[BITS]]
  %0 = tail %in4u, 3 : (!firrtl.uint<4>) -> !firrtl.uint<1>
  connect %out1u, %0 : !firrtl.uint<1>, !firrtl.uint<1>

  // CHECK: [[BITS:%.+]] = bits %in4u 2 to 0
  // CHECK-NEXT: strictconnect %out3u, [[BITS]]
  %1 = tail %in4u, 1 : (!firrtl.uint<4>) -> !firrtl.uint<3>
  connect %out3u, %1 : !firrtl.uint<3>, !firrtl.uint<3>

  // CHECK: strictconnect %out3u, %c2_ui3
  %c10_ui4 = constant 10 : !firrtl.uint<4>
  %2 = tail %c10_ui4, 1 : (!firrtl.uint<4>) -> !firrtl.uint<3>
  connect %out3u, %2 : !firrtl.uint<3>, !firrtl.uint<3>
}

// CHECK-LABEL: module @Andr
firrtl.module @Andr(in %in0 : !firrtl.uint<0>,
                    out %a: !firrtl.uint<1>, out %b: !firrtl.uint<1>,
                    out %c: !firrtl.uint<1>, out %d: !firrtl.uint<1>,
                    out %e: !firrtl.uint<1>, out %f: !firrtl.uint<1>) {
  %invalid_ui2 = invalidvalue : !firrtl.uint<2>
  %c2_ui2 = constant 2 : !firrtl.uint<2>
  %c3_ui2 = constant 3 : !firrtl.uint<2>
  %cn2_si2 = constant -2 : !firrtl.sint<2>
  %cn1_si2 = constant -1 : !firrtl.sint<2>
  %0 = andr %c2_ui2 : (!firrtl.uint<2>) -> !firrtl.uint<1>
  %1 = andr %c3_ui2 : (!firrtl.uint<2>) -> !firrtl.uint<1>
  %2 = andr %cn2_si2 : (!firrtl.sint<2>) -> !firrtl.uint<1>
  %3 = andr %cn1_si2 : (!firrtl.sint<2>) -> !firrtl.uint<1>
  %4 = andr %in0 : (!firrtl.uint<0>) -> !firrtl.uint<1>
  // CHECK: %[[ONE:.+]] = constant 1 : !firrtl.uint<1>
  // CHECK: %[[ZERO:.+]] = constant 0 : !firrtl.uint<1>
  // CHECK: strictconnect %a, %[[ZERO]]
  connect %a, %0 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK: strictconnect %b, %[[ONE]]
  connect %b, %1 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK: strictconnect %c, %[[ZERO]]
  connect %c, %2 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK: strictconnect %d, %[[ONE]]
  connect %d, %3 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK: strictconnect %e, %[[ONE]]
  connect %e, %4 : !firrtl.uint<1>, !firrtl.uint<1>
}

// CHECK-LABEL: module @Orr
firrtl.module @Orr(in %in0 : !firrtl.uint<0>,
                   out %a: !firrtl.uint<1>, out %b: !firrtl.uint<1>,
                   out %c: !firrtl.uint<1>, out %d: !firrtl.uint<1>,
                   out %e: !firrtl.uint<1>, out %f: !firrtl.uint<1>) {
  %invalid_ui2 = invalidvalue : !firrtl.uint<2>
  %c0_ui2 = constant 0 : !firrtl.uint<2>
  %c2_ui2 = constant 2 : !firrtl.uint<2>
  %cn0_si2 = constant 0 : !firrtl.sint<2>
  %cn2_si2 = constant -2 : !firrtl.sint<2>
  %0 = orr %c0_ui2 : (!firrtl.uint<2>) -> !firrtl.uint<1>
  %1 = orr %c2_ui2 : (!firrtl.uint<2>) -> !firrtl.uint<1>
  %2 = orr %cn0_si2 : (!firrtl.sint<2>) -> !firrtl.uint<1>
  %3 = orr %cn2_si2 : (!firrtl.sint<2>) -> !firrtl.uint<1>
  %4 = orr %in0 : (!firrtl.uint<0>) -> !firrtl.uint<1>
  // CHECK-DAG: %[[ZERO:.+]] = constant 0 : !firrtl.uint<1>
  // CHECK-DAG: %[[ONE:.+]] = constant 1 : !firrtl.uint<1>
  // CHECK: strictconnect %a, %[[ZERO]]
  connect %a, %0 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK: strictconnect %b, %[[ONE]]
  connect %b, %1 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK: strictconnect %c, %[[ZERO]]
  connect %c, %2 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK: strictconnect %d, %[[ONE]]
  connect %d, %3 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK: strictconnect %e, %[[ZERO]]
  connect %e, %4 : !firrtl.uint<1>, !firrtl.uint<1>
}

// CHECK-LABEL: module @Xorr
firrtl.module @Xorr(in %in0 : !firrtl.uint<0>,
                    out %a: !firrtl.uint<1>, out %b: !firrtl.uint<1>,
                    out %c: !firrtl.uint<1>, out %d: !firrtl.uint<1>,
                    out %e: !firrtl.uint<1>, out %f: !firrtl.uint<1>) {
  %invalid_ui2 = invalidvalue : !firrtl.uint<2>
  %c3_ui2 = constant 3 : !firrtl.uint<2>
  %c2_ui2 = constant 2 : !firrtl.uint<2>
  %cn1_si2 = constant -1 : !firrtl.sint<2>
  %cn2_si2 = constant -2 : !firrtl.sint<2>
  %0 = xorr %c3_ui2 : (!firrtl.uint<2>) -> !firrtl.uint<1>
  %1 = xorr %c2_ui2 : (!firrtl.uint<2>) -> !firrtl.uint<1>
  %2 = xorr %cn1_si2 : (!firrtl.sint<2>) -> !firrtl.uint<1>
  %3 = xorr %cn2_si2 : (!firrtl.sint<2>) -> !firrtl.uint<1>
  %4 = xorr %in0 : (!firrtl.uint<0>) -> !firrtl.uint<1>
  // CHECK-DAG: %[[ZERO:.+]] = constant 0 : !firrtl.uint<1>
  // CHECK-DAG: %[[ONE:.+]] = constant 1 : !firrtl.uint<1>
  // CHECK: strictconnect %a, %[[ZERO]]
  connect %a, %0 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK: strictconnect %b, %[[ONE]]
  connect %b, %1 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK: strictconnect %c, %[[ZERO]]
  connect %c, %2 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK: strictconnect %d, %[[ONE]]
  connect %d, %3 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK: strictconnect %e, %[[ZERO]]
  connect %e, %4 : !firrtl.uint<1>, !firrtl.uint<1>
}

// CHECK-LABEL: module @Reduce
firrtl.module @Reduce(in %a: !firrtl.uint<1>, out %b: !firrtl.uint<1>,
                      out %c: !firrtl.uint<1>, out %d: !firrtl.uint<1>) {
  %0 = andr %a : (!firrtl.uint<1>) -> !firrtl.uint<1>
  %1 = orr %a : (!firrtl.uint<1>) -> !firrtl.uint<1>
  %2 = xorr %a : (!firrtl.uint<1>) -> !firrtl.uint<1>
  connect %b, %0 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK: strictconnect %b, %a
  connect %c, %1 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK: strictconnect %c, %a
  connect %d, %2 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK: strictconnect %d, %a
}


// CHECK-LABEL: module @subaccess
firrtl.module @subaccess(out %result: !firrtl.uint<8>, in %vec0: !firrtl.vector<uint<8>, 16>) {
  // CHECK: [[TMP:%.+]] = subindex %vec0[11]
  // CHECK-NEXT: strictconnect %result, [[TMP]]
  %c11_ui8 = constant 11 : !firrtl.uint<8>
  %0 = subaccess %vec0[%c11_ui8] : !firrtl.vector<uint<8>, 16>, !firrtl.uint<8>
  connect %result, %0 :!firrtl.uint<8>, !firrtl.uint<8>
}

// CHECK-LABEL: module @subindex
firrtl.module @subindex(out %out : !firrtl.uint<8>) {
  // CHECK: %c8_ui8 = constant 8 : !firrtl.uint<8>
  // CHECK: strictconnect %out, %c8_ui8 : !firrtl.uint<8>
  %0 = aggregateconstant [8 : ui8] : !firrtl.vector<uint<8>, 1>
  %1 = subindex %0[0] : !firrtl.vector<uint<8>, 1>
  strictconnect %out, %1 : !firrtl.uint<8>
}

// CHECK-LABEL: module @subindex_agg
firrtl.module @subindex_agg(out %out : !firrtl.bundle<a: uint<8>>) {
  // CHECK: %0 = aggregateconstant [8 : ui8] : !firrtl.bundle<a: uint<8>>
  // CHECK: strictconnect %out, %0 : !firrtl.bundle<a: uint<8>>
  %0 = aggregateconstant [[8 : ui8]] : !firrtl.vector<bundle<a: uint<8>>, 1>
  %1 = subindex %0[0] : !firrtl.vector<bundle<a: uint<8>>, 1>
  strictconnect %out, %1 : !firrtl.bundle<a: uint<8>>
}

// CHECK-LABEL: module @subfield
firrtl.module @subfield(out %out : !firrtl.uint<8>) {
  // CHECK: %c8_ui8 = constant 8 : !firrtl.uint<8>
  // CHECK: strictconnect %out, %c8_ui8 : !firrtl.uint<8>
  %0 = aggregateconstant [8 : ui8] : !firrtl.bundle<a: uint<8>>
  %1 = subfield %0[a] : !firrtl.bundle<a: uint<8>>
  strictconnect %out, %1 : !firrtl.uint<8>
}

// CHECK-LABEL: module @subfield_agg
firrtl.module @subfield_agg(out %out : !firrtl.vector<uint<8>, 1>) {
  // CHECK: %0 = aggregateconstant [8 : ui8] : !firrtl.vector<uint<8>, 1>
  // CHECK: strictconnect %out, %0 : !firrtl.vector<uint<8>, 1>
  %0 = aggregateconstant [[8 : ui8]] : !firrtl.bundle<a: vector<uint<8>, 1>>
  %1 = subfield %0[a] : !firrtl.bundle<a: vector<uint<8>, 1>>
  strictconnect %out, %1 : !firrtl.vector<uint<8>, 1>
}

// CHECK-LABEL: module @issue326
firrtl.module @issue326(out %tmp57: !firrtl.sint<1>) {
  %c29_si7 = constant 29 : !firrtl.sint<7>
  %0 = shr %c29_si7, 47 : (!firrtl.sint<7>) -> !firrtl.sint<1>
   // CHECK: c0_si1 = constant 0 : !firrtl.sint<1>
   connect %tmp57, %0 : !firrtl.sint<1>, !firrtl.sint<1>
}

// CHECK-LABEL: module @issue331
firrtl.module @issue331(out %tmp81: !firrtl.sint<1>) {
  // CHECK: %c-1_si1 = constant -1 : !firrtl.sint<1>
  %c-1_si1 = constant -1 : !firrtl.sint<1>
  %0 = shr %c-1_si1, 3 : (!firrtl.sint<1>) -> !firrtl.sint<1>
  connect %tmp81, %0 : !firrtl.sint<1>, !firrtl.sint<1>
}

// CHECK-LABEL: module @issue432
firrtl.module @issue432(out %tmp8: !firrtl.uint<10>) {
  %c130_si10 = constant 130 : !firrtl.sint<10>
  %0 = tail %c130_si10, 0 : (!firrtl.sint<10>) -> !firrtl.uint<10>
  connect %tmp8, %0 : !firrtl.uint<10>, !firrtl.uint<10>
  // CHECK-NEXT: %c130_ui10 = constant 130 : !firrtl.uint<10>
  // CHECK-NEXT: strictconnect %tmp8, %c130_ui10
}

// CHECK-LABEL: module @issue437
firrtl.module @issue437(out %tmp19: !firrtl.uint<1>) {
  // CHECK-NEXT: %c1_ui1 = constant 1 : !firrtl.uint<1>
  %c-1_si1 = constant -1 : !firrtl.sint<1>
  %0 = bits %c-1_si1 0 to 0 : (!firrtl.sint<1>) -> !firrtl.uint<1>
  connect %tmp19, %0 : !firrtl.uint<1>, !firrtl.uint<1>
}

// CHECK-LABEL: module @issue446
// CHECK-NEXT: [[TMP:%.+]] = constant 0 : !firrtl.uint<1>
// CHECK-NEXT: strictconnect %tmp10, [[TMP]] : !firrtl.uint<1>
firrtl.module @issue446(in %inp_1: !firrtl.sint<0>, out %tmp10: !firrtl.uint<1>) {
  %0 = xor %inp_1, %inp_1 : (!firrtl.sint<0>, !firrtl.sint<0>) -> !firrtl.uint<0>
  connect %tmp10, %0 : !firrtl.uint<1>, !firrtl.uint<0>
}

// CHECK-LABEL: module @xorUnsized
// CHECK-NEXT: %c0_ui = constant 0 : !firrtl.uint
firrtl.module @xorUnsized(in %inp_1: !firrtl.sint, out %tmp10: !firrtl.uint) {
  %0 = xor %inp_1, %inp_1 : (!firrtl.sint, !firrtl.sint) -> !firrtl.uint
  connect %tmp10, %0 : !firrtl.uint, !firrtl.uint
}

// https://github.com/llvm/circt/issues/516
// CHECK-LABEL: @issue516
// CHECK-NEXT: [[TMP:%.+]] = constant 0 : !firrtl.uint<0>
// CHECK-NEXT: strictconnect %tmp3, [[TMP]] : !firrtl.uint<0>
firrtl.module @issue516(in %inp_0: !firrtl.uint<0>, out %tmp3: !firrtl.uint<0>) {
  %0 = div %inp_0, %inp_0 : (!firrtl.uint<0>, !firrtl.uint<0>) -> !firrtl.uint<0>
  connect %tmp3, %0 : !firrtl.uint<0>, !firrtl.uint<0>
}

// https://github.com/llvm/circt/issues/591
// CHECK-LABEL: @reg_cst_prop1
// CHECK-NEXT:   %c5_ui8 = constant 5 : !firrtl.uint<8>
// CHECK-NEXT:   strictconnect %out_b, %c5_ui8 : !firrtl.uint<8>
// CHECK-NEXT:  }
firrtl.module @reg_cst_prop1(in %clock: !firrtl.clock, out %out_b: !firrtl.uint<8>) {
  %c5_ui8 = constant 5 : !firrtl.uint<8>
  %_tmp_a = reg droppable_name %clock {name = "_tmp_a"} : !firrtl.clock, !firrtl.uint<8>
  %tmp_b = reg droppable_name %clock {name = "_tmp_b"} : !firrtl.clock, !firrtl.uint<8>
  connect %_tmp_a, %c5_ui8 : !firrtl.uint<8>, !firrtl.uint<8>
  connect %tmp_b, %_tmp_a : !firrtl.uint<8>, !firrtl.uint<8>
  connect %out_b, %tmp_b : !firrtl.uint<8>, !firrtl.uint<8>
}

// Check for DontTouch annotation
// CHECK-LABEL: @reg_cst_prop1_DontTouch
// CHECK-NEXT:      %c5_ui8 = constant 5 : !firrtl.uint<8>
// CHECK-NEXT:      %tmp_a = reg sym @reg1 %clock : !firrtl.clock, !firrtl.uint<8>
// CHECK-NEXT:      %tmp_b = reg %clock  : !firrtl.clock, !firrtl.uint<8>
// CHECK-NEXT:      strictconnect %tmp_a, %c5_ui8 : !firrtl.uint<8>
// CHECK-NEXT:      strictconnect %tmp_b, %tmp_a : !firrtl.uint<8>
// CHECK-NEXT:      strictconnect %out_b, %tmp_b : !firrtl.uint<8>

firrtl.module @reg_cst_prop1_DontTouch(in %clock: !firrtl.clock, out %out_b: !firrtl.uint<8>) {
  %c5_ui8 = constant 5 : !firrtl.uint<8>
  %_tmp_a = reg  sym @reg1 %clock {name = "tmp_a"} : !firrtl.clock, !firrtl.uint<8>
  %_tmp_b = reg %clock {name = "tmp_b"} : !firrtl.clock, !firrtl.uint<8>
  connect %_tmp_a, %c5_ui8 : !firrtl.uint<8>, !firrtl.uint<8>
  connect %_tmp_b, %_tmp_a : !firrtl.uint<8>, !firrtl.uint<8>
  connect %out_b, %_tmp_b : !firrtl.uint<8>, !firrtl.uint<8>
}
// CHECK-LABEL: @reg_cst_prop2
// CHECK-NEXT:   %c5_ui8 = constant 5 : !firrtl.uint<8>
// CHECK-NEXT:   strictconnect %out_b, %c5_ui8 : !firrtl.uint<8>
// CHECK-NEXT:  }
firrtl.module @reg_cst_prop2(in %clock: !firrtl.clock, out %out_b: !firrtl.uint<8>) {
  %_tmp_b = reg droppable_name %clock {name = "_tmp_b"} : !firrtl.clock, !firrtl.uint<8>
  connect %out_b, %_tmp_b : !firrtl.uint<8>, !firrtl.uint<8>

  %_tmp_a = reg droppable_name %clock {name = "_tmp_a"} : !firrtl.clock, !firrtl.uint<8>
  %c5_ui8 = constant 5 : !firrtl.uint<8>
  connect %_tmp_a, %c5_ui8 : !firrtl.uint<8>, !firrtl.uint<8>
  connect %_tmp_b, %_tmp_a : !firrtl.uint<8>, !firrtl.uint<8>
}

// CHECK-LABEL: @reg_cst_prop3
// CHECK-NEXT:   %c0_ui8 = constant 0 : !firrtl.uint<8>
// CHECK-NEXT:   strictconnect %out_b, %c0_ui8 : !firrtl.uint<8>
// CHECK-NEXT:  }
firrtl.module @reg_cst_prop3(in %clock: !firrtl.clock, out %out_b: !firrtl.uint<8>) {
  %_tmp_a = reg droppable_name %clock {name = "_tmp_a"} : !firrtl.clock, !firrtl.uint<8>
  %c5_ui8 = constant 5 : !firrtl.uint<8>
  connect %_tmp_a, %c5_ui8 : !firrtl.uint<8>, !firrtl.uint<8>

  %xor = xor %_tmp_a, %c5_ui8 : (!firrtl.uint<8>, !firrtl.uint<8>) -> !firrtl.uint<8>
  connect %out_b, %xor : !firrtl.uint<8>, !firrtl.uint<8>
}

// https://github.com/llvm/circt/issues/788

// CHECK-LABEL: @AttachMerge
firrtl.module @AttachMerge(in %a: !firrtl.analog<1>, in %b: !firrtl.analog<1>,
                           in %c: !firrtl.analog<1>) {
  // CHECK-NEXT: attach %c, %b, %a :
  // CHECK-NEXT: }
  attach %b, %a : !firrtl.analog<1>, !firrtl.analog<1>
  attach %c, %b : !firrtl.analog<1>, !firrtl.analog<1>
}

// CHECK-LABEL: @AttachDeadWire
firrtl.module @AttachDeadWire(in %a: !firrtl.analog<1>, in %b: !firrtl.analog<1>) {
  // CHECK-NEXT: attach %a, %b :
  // CHECK-NEXT: }
  %c = wire  : !firrtl.analog<1>
  attach %a, %b, %c : !firrtl.analog<1>, !firrtl.analog<1>, !firrtl.analog<1>
}

// CHECK-LABEL: @AttachOpts
firrtl.module @AttachOpts(in %a: !firrtl.analog<1>) {
  // CHECK-NEXT: }
  %b = wire  : !firrtl.analog<1>
  attach %b, %a : !firrtl.analog<1>, !firrtl.analog<1>
}

// CHECK-LABEL: @AttachDeadWireDontTouch
firrtl.module @AttachDeadWireDontTouch(in %a: !firrtl.analog<1>, in %b: !firrtl.analog<1>) {
  // CHECK-NEXT: %c = wire
  // CHECK-NEXT: attach %a, %b, %c :
  // CHECK-NEXT: }
  %c = wire sym @s1 : !firrtl.analog<1>
  attach %a, %b, %c : !firrtl.analog<1>, !firrtl.analog<1>, !firrtl.analog<1>
}

// CHECK-LABEL: @wire_cst_prop1
// CHECK-NEXT:   %c10_ui9 = constant 10 : !firrtl.uint<9>
// CHECK-NEXT:   strictconnect %out_b, %c10_ui9 : !firrtl.uint<9>
// CHECK-NEXT:  }
firrtl.module @wire_cst_prop1(out %out_b: !firrtl.uint<9>) {
  %_tmp_a = wire droppable_name : !firrtl.uint<8>
  %c5_ui8 = constant 5 : !firrtl.uint<8>
  connect %_tmp_a, %c5_ui8 : !firrtl.uint<8>, !firrtl.uint<8>

  %xor = add %_tmp_a, %c5_ui8 : (!firrtl.uint<8>, !firrtl.uint<8>) -> !firrtl.uint<9>
  connect %out_b, %xor : !firrtl.uint<9>, !firrtl.uint<9>
}

// CHECK-LABEL: @wire_port_prop1
// CHECK-NEXT:   strictconnect %out_b, %in_a : !firrtl.uint<9>
// CHECK-NEXT:  }
firrtl.module @wire_port_prop1(in %in_a: !firrtl.uint<9>, out %out_b: !firrtl.uint<9>) {
  %_tmp = wire droppable_name : !firrtl.uint<9>
  connect %_tmp, %in_a : !firrtl.uint<9>, !firrtl.uint<9>

  connect %out_b, %_tmp : !firrtl.uint<9>, !firrtl.uint<9>
}

// CHECK-LABEL: @LEQWithConstLHS
// CHECK-NEXT: %c42_ui = constant
// CHECK-NEXT: %e = geq %a, %c42_ui
firrtl.module @LEQWithConstLHS(in %a: !firrtl.uint, out %b: !firrtl.uint<1>) {
  %0 = constant 42 : !firrtl.uint
  %1 = leq %0, %a {name = "e"} : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint<1>
  connect %b, %1 : !firrtl.uint<1>, !firrtl.uint<1>
}

// CHECK-LABEL: @LTWithConstLHS
// CHECK-NEXT: %c42_ui = constant
// CHECK-NEXT: %0 = gt %a, %c42_ui
firrtl.module @LTWithConstLHS(in %a: !firrtl.uint, out %b: !firrtl.uint<1>) {
  %0 = constant 42 : !firrtl.uint
  %1 = lt %0, %a : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint<1>
  connect %b, %1 : !firrtl.uint<1>, !firrtl.uint<1>
}

// CHECK-LABEL: @GEQWithConstLHS
// CHECK-NEXT: %c42_ui = constant
// CHECK-NEXT: %0 = leq %a, %c42_ui
firrtl.module @GEQWithConstLHS(in %a: !firrtl.uint, out %b: !firrtl.uint<1>) {
  %0 = constant 42 : !firrtl.uint
  %1 = geq %0, %a : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint<1>
  connect %b, %1 : !firrtl.uint<1>, !firrtl.uint<1>
}

// CHECK-LABEL: @GTWithConstLHS
// CHECK-NEXT: %c42_ui = constant
// CHECK-NEXT: %0 = lt %a, %c42_ui
firrtl.module @GTWithConstLHS(in %a: !firrtl.uint, out %b: !firrtl.uint<1>) {
  %0 = constant 42 : !firrtl.uint
  %1 = gt %0, %a : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint<1>
  connect %b, %1 : !firrtl.uint<1>, !firrtl.uint<1>
}

// CHECK-LABEL: @CompareWithSelf
firrtl.module @CompareWithSelf(
  in %a: !firrtl.uint,
  out %y0: !firrtl.uint<1>,
  out %y1: !firrtl.uint<1>,
  out %y2: !firrtl.uint<1>,
  out %y3: !firrtl.uint<1>,
  out %y4: !firrtl.uint<1>,
  out %y5: !firrtl.uint<1>
) {
  // CHECK-NEXT: [[_:.+]] = constant
  // CHECK-NEXT: [[_:.+]] = constant

  %0 = leq %a, %a : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint<1>
  connect %y0, %0 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK-NEXT: strictconnect %y0, %c1_ui1

  %1 = lt %a, %a : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint<1>
  connect %y1, %1 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK-NEXT: strictconnect %y1, %c0_ui1

  %2 = geq %a, %a : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint<1>
  connect %y2, %2 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK-NEXT: strictconnect %y2, %c1_ui1

  %3 = gt %a, %a : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint<1>
  connect %y3, %3 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK-NEXT: strictconnect %y3, %c0_ui1

  %4 = eq %a, %a : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint<1>
  connect %y4, %4 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK-NEXT: strictconnect %y4, %c1_ui1

  %5 = neq %a, %a : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint<1>
  connect %y5, %5 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK-NEXT: strictconnect %y5, %c0_ui1
}

// CHECK-LABEL: @LEQOutsideBounds
firrtl.module @LEQOutsideBounds(
  in %a: !firrtl.uint<3>,
  in %b: !firrtl.sint<3>,
  out %y0: !firrtl.uint<1>,
  out %y1: !firrtl.uint<1>,
  out %y2: !firrtl.uint<1>,
  out %y3: !firrtl.uint<1>,
  out %y4: !firrtl.uint<1>,
  out %y5: !firrtl.uint<1>
) {
  // CHECK-NEXT: [[_:.+]] = constant
  // CHECK-NEXT: [[_:.+]] = constant
  %cm5_si = constant -5 : !firrtl.sint
  %cm6_si = constant -6 : !firrtl.sint
  %c3_si = constant 3 : !firrtl.sint
  %c4_si = constant 4 : !firrtl.sint
  %c7_ui = constant 7 : !firrtl.uint
  %c8_ui = constant 8 : !firrtl.uint

  // a <= 7 -> 1
  // a <= 8 -> 1
  %0 = leq %a, %c7_ui : (!firrtl.uint<3>, !firrtl.uint) -> !firrtl.uint<1>
  %1 = leq %a, %c8_ui : (!firrtl.uint<3>, !firrtl.uint) -> !firrtl.uint<1>
  connect %y0, %0 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %y1, %1 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK-NEXT: strictconnect %y0, %c1_ui1
  // CHECK-NEXT: strictconnect %y1, %c1_ui1

  // b <= 3 -> 1
  // b <= 4 -> 1
  %2 = leq %b, %c3_si : (!firrtl.sint<3>, !firrtl.sint) -> !firrtl.uint<1>
  %3 = leq %b, %c4_si : (!firrtl.sint<3>, !firrtl.sint) -> !firrtl.uint<1>
  connect %y2, %2 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %y3, %3 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK-NEXT: strictconnect %y2, %c1_ui1
  // CHECK-NEXT: strictconnect %y3, %c1_ui1

  // b <= -5 -> 0
  // b <= -6 -> 0
  %4 = leq %b, %cm5_si : (!firrtl.sint<3>, !firrtl.sint) -> !firrtl.uint<1>
  %5 = leq %b, %cm6_si : (!firrtl.sint<3>, !firrtl.sint) -> !firrtl.uint<1>
  connect %y4, %4 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %y5, %5 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK-NEXT: strictconnect %y4, %c0_ui1
  // CHECK-NEXT: strictconnect %y5, %c0_ui1
}

// CHECK-LABEL: @LTOutsideBounds
firrtl.module @LTOutsideBounds(
  in %a: !firrtl.uint<3>,
  in %b: !firrtl.sint<3>,
  out %y0: !firrtl.uint<1>,
  out %y1: !firrtl.uint<1>,
  out %y2: !firrtl.uint<1>,
  out %y3: !firrtl.uint<1>,
  out %y4: !firrtl.uint<1>,
  out %y5: !firrtl.uint<1>
) {
  // CHECK-NEXT: [[_:.+]] = constant
  // CHECK-NEXT: [[_:.+]] = constant
  %cm4_si = constant -4 : !firrtl.sint
  %cm5_si = constant -5 : !firrtl.sint
  %c4_si = constant 4 : !firrtl.sint
  %c5_si = constant 5 : !firrtl.sint
  %c8_ui = constant 8 : !firrtl.uint
  %c9_ui = constant 9 : !firrtl.uint

  // a < 8 -> 1
  // a < 9 -> 1
  %0 = lt %a, %c8_ui : (!firrtl.uint<3>, !firrtl.uint) -> !firrtl.uint<1>
  %1 = lt %a, %c9_ui : (!firrtl.uint<3>, !firrtl.uint) -> !firrtl.uint<1>
  connect %y0, %0 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %y1, %1 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK-NEXT: strictconnect %y0, %c1_ui1
  // CHECK-NEXT: strictconnect %y1, %c1_ui1

  // b < 4 -> 1
  // b < 5 -> 1
  %2 = lt %b, %c4_si : (!firrtl.sint<3>, !firrtl.sint) -> !firrtl.uint<1>
  %3 = lt %b, %c5_si : (!firrtl.sint<3>, !firrtl.sint) -> !firrtl.uint<1>
  connect %y2, %2 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %y3, %3 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK-NEXT: strictconnect %y2, %c1_ui1
  // CHECK-NEXT: strictconnect %y3, %c1_ui1

  // b < -4 -> 0
  // b < -5 -> 0
  %4 = lt %b, %cm4_si : (!firrtl.sint<3>, !firrtl.sint) -> !firrtl.uint<1>
  %5 = lt %b, %cm5_si : (!firrtl.sint<3>, !firrtl.sint) -> !firrtl.uint<1>
  connect %y4, %4 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %y5, %5 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK-NEXT: strictconnect %y4, %c0_ui1
  // CHECK-NEXT: strictconnect %y5, %c0_ui1
}

// CHECK-LABEL: @GEQOutsideBounds
firrtl.module @GEQOutsideBounds(
  in %a: !firrtl.uint<3>,
  in %b: !firrtl.sint<3>,
  out %y0: !firrtl.uint<1>,
  out %y1: !firrtl.uint<1>,
  out %y2: !firrtl.uint<1>,
  out %y3: !firrtl.uint<1>,
  out %y4: !firrtl.uint<1>,
  out %y5: !firrtl.uint<1>
) {
  // CHECK-NEXT: [[_:.+]] = constant
  // CHECK-NEXT: [[_:.+]] = constant
  %cm4_si = constant -4 : !firrtl.sint
  %cm5_si = constant -5 : !firrtl.sint
  %c4_si = constant 4 : !firrtl.sint
  %c5_si = constant 5 : !firrtl.sint
  %c8_ui = constant 8 : !firrtl.uint
  %c9_ui = constant 9 : !firrtl.uint

  // a >= 8 -> 0
  // a >= 9 -> 0
  %0 = geq %a, %c8_ui : (!firrtl.uint<3>, !firrtl.uint) -> !firrtl.uint<1>
  %1 = geq %a, %c9_ui : (!firrtl.uint<3>, !firrtl.uint) -> !firrtl.uint<1>
  connect %y0, %0 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %y1, %1 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK-NEXT: strictconnect %y0, %c0_ui1
  // CHECK-NEXT: strictconnect %y1, %c0_ui1

  // b >= 4 -> 0
  // b >= 5 -> 0
  %2 = geq %b, %c4_si : (!firrtl.sint<3>, !firrtl.sint) -> !firrtl.uint<1>
  %3 = geq %b, %c5_si : (!firrtl.sint<3>, !firrtl.sint) -> !firrtl.uint<1>
  connect %y2, %2 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %y3, %3 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK-NEXT: strictconnect %y2, %c0_ui1
  // CHECK-NEXT: strictconnect %y3, %c0_ui1

  // b >= -4 -> 1
  // b >= -5 -> 1
  %4 = geq %b, %cm4_si : (!firrtl.sint<3>, !firrtl.sint) -> !firrtl.uint<1>
  %5 = geq %b, %cm5_si : (!firrtl.sint<3>, !firrtl.sint) -> !firrtl.uint<1>
  connect %y4, %4 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %y5, %5 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK-NEXT: strictconnect %y4, %c1_ui1
  // CHECK-NEXT: strictconnect %y5, %c1_ui1
}

// CHECK-LABEL: @GTOutsideBounds
firrtl.module @GTOutsideBounds(
  in %a: !firrtl.uint<3>,
  in %b: !firrtl.sint<3>,
  out %y0: !firrtl.uint<1>,
  out %y1: !firrtl.uint<1>,
  out %y2: !firrtl.uint<1>,
  out %y3: !firrtl.uint<1>,
  out %y4: !firrtl.uint<1>,
  out %y5: !firrtl.uint<1>
) {
  // CHECK-NEXT: [[_:.+]] = constant
  // CHECK-NEXT: [[_:.+]] = constant
  %cm5_si = constant -5 : !firrtl.sint
  %cm6_si = constant -6 : !firrtl.sint
  %c3_si = constant 3 : !firrtl.sint
  %c4_si = constant 4 : !firrtl.sint
  %c7_ui = constant 7 : !firrtl.uint
  %c8_ui = constant 8 : !firrtl.uint

  // a > 7 -> 0
  // a > 8 -> 0
  %0 = gt %a, %c7_ui : (!firrtl.uint<3>, !firrtl.uint) -> !firrtl.uint<1>
  %1 = gt %a, %c8_ui : (!firrtl.uint<3>, !firrtl.uint) -> !firrtl.uint<1>
  connect %y0, %0 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %y1, %1 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK-NEXT: strictconnect %y0, %c0_ui1
  // CHECK-NEXT: strictconnect %y1, %c0_ui1

  // b > 3 -> 0
  // b > 4 -> 0
  %2 = gt %b, %c3_si : (!firrtl.sint<3>, !firrtl.sint) -> !firrtl.uint<1>
  %3 = gt %b, %c4_si : (!firrtl.sint<3>, !firrtl.sint) -> !firrtl.uint<1>
  connect %y2, %2 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %y3, %3 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK-NEXT: strictconnect %y2, %c0_ui1
  // CHECK-NEXT: strictconnect %y3, %c0_ui1

  // b > -5 -> 1
  // b > -6 -> 1
  %4 = gt %b, %cm5_si : (!firrtl.sint<3>, !firrtl.sint) -> !firrtl.uint<1>
  %5 = gt %b, %cm6_si : (!firrtl.sint<3>, !firrtl.sint) -> !firrtl.uint<1>
  connect %y4, %4 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %y5, %5 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK-NEXT: strictconnect %y4, %c1_ui1
  // CHECK-NEXT: strictconnect %y5, %c1_ui1
}

// CHECK-LABEL: @ComparisonOfDifferentWidths
firrtl.module @ComparisonOfDifferentWidths(
  out %y0: !firrtl.uint<1>,
  out %y1: !firrtl.uint<1>,
  out %y2: !firrtl.uint<1>,
  out %y3: !firrtl.uint<1>,
  out %y4: !firrtl.uint<1>,
  out %y5: !firrtl.uint<1>,
  out %y6: !firrtl.uint<1>,
  out %y7: !firrtl.uint<1>,
  out %y8: !firrtl.uint<1>,
  out %y9: !firrtl.uint<1>,
  out %y10: !firrtl.uint<1>,
  out %y11: !firrtl.uint<1>
) {
  // CHECK-NEXT: [[_:.+]] = constant
  // CHECK-NEXT: [[_:.+]] = constant
  %c3_si3 = constant 3 : !firrtl.sint<3>
  %c4_si4 = constant 4 : !firrtl.sint<4>
  %c3_ui2 = constant 3 : !firrtl.uint<2>
  %c4_ui3 = constant 4 : !firrtl.uint<3>

  %0 = leq %c3_ui2, %c4_ui3 : (!firrtl.uint<2>, !firrtl.uint<3>) -> !firrtl.uint<1>
  %1 = leq %c3_si3, %c4_si4 : (!firrtl.sint<3>, !firrtl.sint<4>) -> !firrtl.uint<1>
  %2 = lt %c3_ui2, %c4_ui3 : (!firrtl.uint<2>, !firrtl.uint<3>) -> !firrtl.uint<1>
  %3 = lt %c3_si3, %c4_si4 : (!firrtl.sint<3>, !firrtl.sint<4>) -> !firrtl.uint<1>
  %4 = geq %c3_ui2, %c4_ui3 : (!firrtl.uint<2>, !firrtl.uint<3>) -> !firrtl.uint<1>
  %5 = geq %c3_si3, %c4_si4 : (!firrtl.sint<3>, !firrtl.sint<4>) -> !firrtl.uint<1>
  %6 = gt %c3_ui2, %c4_ui3 : (!firrtl.uint<2>, !firrtl.uint<3>) -> !firrtl.uint<1>
  %7 = gt %c3_si3, %c4_si4 : (!firrtl.sint<3>, !firrtl.sint<4>) -> !firrtl.uint<1>
  %8 = eq %c3_ui2, %c4_ui3 : (!firrtl.uint<2>, !firrtl.uint<3>) -> !firrtl.uint<1>
  %9 = eq %c3_si3, %c4_si4 : (!firrtl.sint<3>, !firrtl.sint<4>) -> !firrtl.uint<1>
  %10 = neq %c3_ui2, %c4_ui3 : (!firrtl.uint<2>, !firrtl.uint<3>) -> !firrtl.uint<1>
  %11 = neq %c3_si3, %c4_si4 : (!firrtl.sint<3>, !firrtl.sint<4>) -> !firrtl.uint<1>

  connect %y0, %0 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %y1, %1 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %y2, %2 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %y3, %3 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %y4, %4 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %y5, %5 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %y6, %6 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %y7, %7 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %y8, %8 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %y9, %9 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %y10, %10 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %y11, %11 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK-NEXT: strictconnect %y0, %c1_ui1
  // CHECK-NEXT: strictconnect %y1, %c1_ui1
  // CHECK-NEXT: strictconnect %y2, %c1_ui1
  // CHECK-NEXT: strictconnect %y3, %c1_ui1
  // CHECK-NEXT: strictconnect %y4, %c0_ui1
  // CHECK-NEXT: strictconnect %y5, %c0_ui1
  // CHECK-NEXT: strictconnect %y6, %c0_ui1
  // CHECK-NEXT: strictconnect %y7, %c0_ui1
  // CHECK-NEXT: strictconnect %y8, %c0_ui1
  // CHECK-NEXT: strictconnect %y9, %c0_ui1
  // CHECK-NEXT: strictconnect %y10, %c1_ui1
  // CHECK-NEXT: strictconnect %y11, %c1_ui1
}

// CHECK-LABEL: @ComparisonOfUnsizedAndSized
firrtl.module @ComparisonOfUnsizedAndSized(
  out %y0: !firrtl.uint<1>,
  out %y1: !firrtl.uint<1>,
  out %y2: !firrtl.uint<1>,
  out %y3: !firrtl.uint<1>,
  out %y4: !firrtl.uint<1>,
  out %y5: !firrtl.uint<1>,
  out %y6: !firrtl.uint<1>,
  out %y7: !firrtl.uint<1>,
  out %y8: !firrtl.uint<1>,
  out %y9: !firrtl.uint<1>,
  out %y10: !firrtl.uint<1>,
  out %y11: !firrtl.uint<1>
) {
  // CHECK-NEXT: [[_:.+]] = constant
  // CHECK-NEXT: [[_:.+]] = constant
  %c3_si = constant 3 : !firrtl.sint
  %c4_si4 = constant 4 : !firrtl.sint<4>
  %c3_ui = constant 3 : !firrtl.uint
  %c4_ui3 = constant 4 : !firrtl.uint<3>

  %0 = leq %c3_ui, %c4_ui3 : (!firrtl.uint, !firrtl.uint<3>) -> !firrtl.uint<1>
  %1 = leq %c3_si, %c4_si4 : (!firrtl.sint, !firrtl.sint<4>) -> !firrtl.uint<1>
  %2 = lt %c3_ui, %c4_ui3 : (!firrtl.uint, !firrtl.uint<3>) -> !firrtl.uint<1>
  %3 = lt %c3_si, %c4_si4 : (!firrtl.sint, !firrtl.sint<4>) -> !firrtl.uint<1>
  %4 = geq %c3_ui, %c4_ui3 : (!firrtl.uint, !firrtl.uint<3>) -> !firrtl.uint<1>
  %5 = geq %c3_si, %c4_si4 : (!firrtl.sint, !firrtl.sint<4>) -> !firrtl.uint<1>
  %6 = gt %c3_ui, %c4_ui3 : (!firrtl.uint, !firrtl.uint<3>) -> !firrtl.uint<1>
  %7 = gt %c3_si, %c4_si4 : (!firrtl.sint, !firrtl.sint<4>) -> !firrtl.uint<1>
  %8 = eq %c3_ui, %c4_ui3 : (!firrtl.uint, !firrtl.uint<3>) -> !firrtl.uint<1>
  %9 = eq %c3_si, %c4_si4 : (!firrtl.sint, !firrtl.sint<4>) -> !firrtl.uint<1>
  %10 = neq %c3_ui, %c4_ui3 : (!firrtl.uint, !firrtl.uint<3>) -> !firrtl.uint<1>
  %11 = neq %c3_si, %c4_si4 : (!firrtl.sint, !firrtl.sint<4>) -> !firrtl.uint<1>

  connect %y0, %0 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %y1, %1 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %y2, %2 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %y3, %3 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %y4, %4 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %y5, %5 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %y6, %6 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %y7, %7 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %y8, %8 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %y9, %9 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %y10, %10 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %y11, %11 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK-NEXT: strictconnect %y0, %c1_ui1
  // CHECK-NEXT: strictconnect %y1, %c1_ui1
  // CHECK-NEXT: strictconnect %y2, %c1_ui1
  // CHECK-NEXT: strictconnect %y3, %c1_ui1
  // CHECK-NEXT: strictconnect %y4, %c0_ui1
  // CHECK-NEXT: strictconnect %y5, %c0_ui1
  // CHECK-NEXT: strictconnect %y6, %c0_ui1
  // CHECK-NEXT: strictconnect %y7, %c0_ui1
  // CHECK-NEXT: strictconnect %y8, %c0_ui1
  // CHECK-NEXT: strictconnect %y9, %c0_ui1
  // CHECK-NEXT: strictconnect %y10, %c1_ui1
  // CHECK-NEXT: strictconnect %y11, %c1_ui1
}

// CHECK-LABEL: @ComparisonOfUnsized
firrtl.module @ComparisonOfUnsized(
  out %y0: !firrtl.uint<1>,
  out %y1: !firrtl.uint<1>,
  out %y2: !firrtl.uint<1>,
  out %y3: !firrtl.uint<1>,
  out %y4: !firrtl.uint<1>,
  out %y5: !firrtl.uint<1>,
  out %y6: !firrtl.uint<1>,
  out %y7: !firrtl.uint<1>,
  out %y8: !firrtl.uint<1>,
  out %y9: !firrtl.uint<1>,
  out %y10: !firrtl.uint<1>,
  out %y11: !firrtl.uint<1>
) {
  // CHECK-NEXT: [[_:.+]] = constant
  // CHECK-NEXT: [[_:.+]] = constant
  %c0_si = constant 0 : !firrtl.sint
  %c4_si = constant 4 : !firrtl.sint
  %c0_ui = constant 0 : !firrtl.uint
  %c4_ui = constant 4 : !firrtl.uint

  %0 = leq %c0_ui, %c4_ui : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint<1>
  %1 = leq %c0_si, %c4_si : (!firrtl.sint, !firrtl.sint) -> !firrtl.uint<1>
  %2 = lt %c0_ui, %c4_ui : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint<1>
  %3 = lt %c0_si, %c4_si : (!firrtl.sint, !firrtl.sint) -> !firrtl.uint<1>
  %4 = geq %c0_ui, %c4_ui : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint<1>
  %5 = geq %c0_si, %c4_si : (!firrtl.sint, !firrtl.sint) -> !firrtl.uint<1>
  %6 = gt %c0_ui, %c4_ui : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint<1>
  %7 = gt %c0_si, %c4_si : (!firrtl.sint, !firrtl.sint) -> !firrtl.uint<1>
  %8 = eq %c0_ui, %c4_ui : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint<1>
  %9 = eq %c0_si, %c4_si : (!firrtl.sint, !firrtl.sint) -> !firrtl.uint<1>
  %10 = neq %c0_ui, %c4_ui : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint<1>
  %11 = neq %c0_si, %c4_si : (!firrtl.sint, !firrtl.sint) -> !firrtl.uint<1>

  connect %y0, %0 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %y1, %1 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %y2, %2 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %y3, %3 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %y4, %4 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %y5, %5 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %y6, %6 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %y7, %7 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %y8, %8 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %y9, %9 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %y10, %10 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %y11, %11 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK-NEXT: strictconnect %y0, %c1_ui1
  // CHECK-NEXT: strictconnect %y1, %c1_ui1
  // CHECK-NEXT: strictconnect %y2, %c1_ui1
  // CHECK-NEXT: strictconnect %y3, %c1_ui1
  // CHECK-NEXT: strictconnect %y4, %c0_ui1
  // CHECK-NEXT: strictconnect %y5, %c0_ui1
  // CHECK-NEXT: strictconnect %y6, %c0_ui1
  // CHECK-NEXT: strictconnect %y7, %c0_ui1
  // CHECK-NEXT: strictconnect %y8, %c0_ui1
  // CHECK-NEXT: strictconnect %y9, %c0_ui1
  // CHECK-NEXT: strictconnect %y10, %c1_ui1
  // CHECK-NEXT: strictconnect %y11, %c1_ui1
}

// CHECK-LABEL: @ComparisonOfZeroAndNonzeroWidths
firrtl.module @ComparisonOfZeroAndNonzeroWidths(
  in %xu: !firrtl.uint<0>,
  in %xs: !firrtl.sint<0>,
  out %y0: !firrtl.uint<1>,
  out %y1: !firrtl.uint<1>,
  out %y2: !firrtl.uint<1>,
  out %y3: !firrtl.uint<1>,
  out %y4: !firrtl.uint<1>,
  out %y5: !firrtl.uint<1>,
  out %y6: !firrtl.uint<1>,
  out %y7: !firrtl.uint<1>,
  out %y8: !firrtl.uint<1>,
  out %y9: !firrtl.uint<1>,
  out %y10: !firrtl.uint<1>,
  out %y11: !firrtl.uint<1>,
  out %y12: !firrtl.uint<1>,
  out %y13: !firrtl.uint<1>,
  out %y14: !firrtl.uint<1>,
  out %y15: !firrtl.uint<1>,
  out %y16: !firrtl.uint<1>,
  out %y17: !firrtl.uint<1>,
  out %y18: !firrtl.uint<1>,
  out %y19: !firrtl.uint<1>,
  out %y20: !firrtl.uint<1>,
  out %y21: !firrtl.uint<1>,
  out %y22: !firrtl.uint<1>,
  out %y23: !firrtl.uint<1>
) {
  %c0_si4 = constant 0 : !firrtl.sint<4>
  %c0_ui4 = constant 0 : !firrtl.uint<4>
  %c4_si4 = constant 4 : !firrtl.sint<4>
  %c4_ui4 = constant 4 : !firrtl.uint<4>

  %0 = leq %xu, %c0_ui4 : (!firrtl.uint<0>, !firrtl.uint<4>) -> !firrtl.uint<1>
  %1 = leq %xu, %c4_ui4 : (!firrtl.uint<0>, !firrtl.uint<4>) -> !firrtl.uint<1>
  %2 = leq %xs, %c0_si4 : (!firrtl.sint<0>, !firrtl.sint<4>) -> !firrtl.uint<1>
  %3 = leq %xs, %c4_si4 : (!firrtl.sint<0>, !firrtl.sint<4>) -> !firrtl.uint<1>
  %4 = lt %xu, %c0_ui4 : (!firrtl.uint<0>, !firrtl.uint<4>) -> !firrtl.uint<1>
  %5 = lt %xu, %c4_ui4 : (!firrtl.uint<0>, !firrtl.uint<4>) -> !firrtl.uint<1>
  %6 = lt %xs, %c0_si4 : (!firrtl.sint<0>, !firrtl.sint<4>) -> !firrtl.uint<1>
  %7 = lt %xs, %c4_si4 : (!firrtl.sint<0>, !firrtl.sint<4>) -> !firrtl.uint<1>
  %8 = geq %xu, %c0_ui4 : (!firrtl.uint<0>, !firrtl.uint<4>) -> !firrtl.uint<1>
  %9 = geq %xu, %c4_ui4 : (!firrtl.uint<0>, !firrtl.uint<4>) -> !firrtl.uint<1>
  %10 = geq %xs, %c0_si4 : (!firrtl.sint<0>, !firrtl.sint<4>) -> !firrtl.uint<1>
  %11 = geq %xs, %c4_si4 : (!firrtl.sint<0>, !firrtl.sint<4>) -> !firrtl.uint<1>
  %12 = gt %xu, %c0_ui4 : (!firrtl.uint<0>, !firrtl.uint<4>) -> !firrtl.uint<1>
  %13 = gt %xu, %c4_ui4 : (!firrtl.uint<0>, !firrtl.uint<4>) -> !firrtl.uint<1>
  %14 = gt %xs, %c0_si4 : (!firrtl.sint<0>, !firrtl.sint<4>) -> !firrtl.uint<1>
  %15 = gt %xs, %c4_si4 : (!firrtl.sint<0>, !firrtl.sint<4>) -> !firrtl.uint<1>
  %16 = eq %xu, %c0_ui4 : (!firrtl.uint<0>, !firrtl.uint<4>) -> !firrtl.uint<1>
  %17 = eq %xu, %c4_ui4 : (!firrtl.uint<0>, !firrtl.uint<4>) -> !firrtl.uint<1>
  %18 = eq %xs, %c0_si4 : (!firrtl.sint<0>, !firrtl.sint<4>) -> !firrtl.uint<1>
  %19 = eq %xs, %c4_si4 : (!firrtl.sint<0>, !firrtl.sint<4>) -> !firrtl.uint<1>
  %20 = neq %xu, %c0_ui4 : (!firrtl.uint<0>, !firrtl.uint<4>) -> !firrtl.uint<1>
  %21 = neq %xu, %c4_ui4 : (!firrtl.uint<0>, !firrtl.uint<4>) -> !firrtl.uint<1>
  %22 = neq %xs, %c0_si4 : (!firrtl.sint<0>, !firrtl.sint<4>) -> !firrtl.uint<1>
  %23 = neq %xs, %c4_si4 : (!firrtl.sint<0>, !firrtl.sint<4>) -> !firrtl.uint<1>

  connect %y0, %0 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %y1, %1 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %y2, %2 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %y3, %3 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %y4, %4 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %y5, %5 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %y6, %6 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %y7, %7 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %y8, %8 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %y9, %9 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %y10, %10 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %y11, %11 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %y12, %12 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %y13, %13 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %y14, %14 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %y15, %15 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %y16, %16 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %y17, %17 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %y18, %18 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %y19, %19 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %y20, %20 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %y21, %21 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %y22, %22 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %y23, %23 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK: strictconnect %y0, %c1_ui1
  // CHECK: strictconnect %y1, %c1_ui1
  // CHECK: strictconnect %y2, %c1_ui1
  // CHECK: strictconnect %y3, %c1_ui1
  // CHECK: strictconnect %y4, %c0_ui1
  // CHECK: strictconnect %y5, %c1_ui1
  // CHECK: strictconnect %y6, %c0_ui1
  // CHECK: strictconnect %y7, %c1_ui1
  // CHECK: strictconnect %y8, %c1_ui1
  // CHECK: strictconnect %y9, %c0_ui1
  // CHECK: strictconnect %y10, %c1_ui1
  // CHECK: strictconnect %y11, %c0_ui1
  // CHECK: strictconnect %y12, %c0_ui1
  // CHECK: strictconnect %y13, %c0_ui1
  // CHECK: strictconnect %y14, %c0_ui1
  // CHECK: strictconnect %y15, %c0_ui1
  // CHECK: strictconnect %y16, %c1_ui1
  // CHECK: strictconnect %y17, %c0_ui1
  // CHECK: strictconnect %y18, %c1_ui1
  // CHECK: strictconnect %y19, %c0_ui1
  // CHECK: strictconnect %y20, %c0_ui1
  // CHECK: strictconnect %y21, %c1_ui1
  // CHECK: strictconnect %y22, %c0_ui1
  // CHECK: strictconnect %y23, %c1_ui1
}

// CHECK-LABEL: @ComparisonOfZeroWidths
firrtl.module @ComparisonOfZeroWidths(
  in %xu0: !firrtl.uint<0>,
  in %xu1: !firrtl.uint<0>,
  in %xs0: !firrtl.sint<0>,
  in %xs1: !firrtl.sint<0>,
  out %y0: !firrtl.uint<1>,
  out %y1: !firrtl.uint<1>,
  out %y2: !firrtl.uint<1>,
  out %y3: !firrtl.uint<1>,
  out %y4: !firrtl.uint<1>,
  out %y5: !firrtl.uint<1>,
  out %y6: !firrtl.uint<1>,
  out %y7: !firrtl.uint<1>,
  out %y8: !firrtl.uint<1>,
  out %y9: !firrtl.uint<1>,
  out %y10: !firrtl.uint<1>,
  out %y11: !firrtl.uint<1>,
  out %y12: !firrtl.uint<1>,
  out %y13: !firrtl.uint<1>,
  out %y14: !firrtl.uint<1>,
  out %y15: !firrtl.uint<1>,
  out %y16: !firrtl.uint<1>,
  out %y17: !firrtl.uint<1>,
  out %y18: !firrtl.uint<1>,
  out %y19: !firrtl.uint<1>,
  out %y20: !firrtl.uint<1>,
  out %y21: !firrtl.uint<1>,
  out %y22: !firrtl.uint<1>,
  out %y23: !firrtl.uint<1>
) {
  %0 = leq %xu0, %xu1 : (!firrtl.uint<0>, !firrtl.uint<0>) -> !firrtl.uint<1>
  %1 = leq %xs0, %xs1 : (!firrtl.sint<0>, !firrtl.sint<0>) -> !firrtl.uint<1>
  %2 = lt %xu0, %xu1 : (!firrtl.uint<0>, !firrtl.uint<0>) -> !firrtl.uint<1>
  %3 = lt %xs0, %xs1 : (!firrtl.sint<0>, !firrtl.sint<0>) -> !firrtl.uint<1>
  %4 = geq %xu0, %xu1 : (!firrtl.uint<0>, !firrtl.uint<0>) -> !firrtl.uint<1>
  %5 = geq %xs0, %xs1 : (!firrtl.sint<0>, !firrtl.sint<0>) -> !firrtl.uint<1>
  %6 = gt %xu0, %xu1 : (!firrtl.uint<0>, !firrtl.uint<0>) -> !firrtl.uint<1>
  %7 = gt %xs0, %xs1 : (!firrtl.sint<0>, !firrtl.sint<0>) -> !firrtl.uint<1>
  %8 = eq %xu0, %xu1 : (!firrtl.uint<0>, !firrtl.uint<0>) -> !firrtl.uint<1>
  %9 = eq %xs0, %xs1 : (!firrtl.sint<0>, !firrtl.sint<0>) -> !firrtl.uint<1>
  %10 = neq %xu0, %xu1 : (!firrtl.uint<0>, !firrtl.uint<0>) -> !firrtl.uint<1>
  %11 = neq %xs0, %xs1 : (!firrtl.sint<0>, !firrtl.sint<0>) -> !firrtl.uint<1>

  connect %y0, %0 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %y1, %1 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %y2, %2 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %y3, %3 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %y4, %4 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %y5, %5 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %y6, %6 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %y7, %7 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %y8, %8 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %y9, %9 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %y10, %10 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %y11, %11 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK: strictconnect %y0, %c1_ui1
  // CHECK: strictconnect %y1, %c1_ui1
  // CHECK: strictconnect %y2, %c0_ui1
  // CHECK: strictconnect %y3, %c0_ui1
  // CHECK: strictconnect %y4, %c1_ui1
  // CHECK: strictconnect %y5, %c1_ui1
  // CHECK: strictconnect %y6, %c0_ui1
  // CHECK: strictconnect %y7, %c0_ui1
  // CHECK: strictconnect %y8, %c1_ui1
  // CHECK: strictconnect %y9, %c1_ui1
  // CHECK: strictconnect %y10, %c0_ui1
  // CHECK: strictconnect %y11, %c0_ui1
}

// CHECK-LABEL: @ComparisonOfConsts
firrtl.module @ComparisonOfConsts(
  out %y0: !firrtl.uint<1>,
  out %y1: !firrtl.uint<1>,
  out %y2: !firrtl.uint<1>,
  out %y3: !firrtl.uint<1>,
  out %y4: !firrtl.uint<1>,
  out %y5: !firrtl.uint<1>,
  out %y6: !firrtl.uint<1>,
  out %y7: !firrtl.uint<1>,
  out %y8: !firrtl.uint<1>,
  out %y9: !firrtl.uint<1>,
  out %y10: !firrtl.uint<1>,
  out %y11: !firrtl.uint<1>,
  out %y12: !firrtl.uint<1>,
  out %y13: !firrtl.uint<1>,
  out %y14: !firrtl.uint<1>,
  out %y15: !firrtl.uint<1>,
  out %y16: !firrtl.uint<1>,
  out %y17: !firrtl.uint<1>,
  out %y18: !firrtl.uint<1>,
  out %y19: !firrtl.uint<1>,
  out %y20: !firrtl.uint<1>,
  out %y21: !firrtl.uint<1>,
  out %y22: !firrtl.uint<1>,
  out %y23: !firrtl.uint<1>
) {
  %c0_si0 = constant 0 : !firrtl.sint<0>
  %c2_si4 = constant 2 : !firrtl.sint<4>
  %c-3_si3 = constant -3 : !firrtl.sint<3>
  %c2_ui4 = constant 2 : !firrtl.uint<4>
  %c5_ui3 = constant 5 : !firrtl.uint<3>

  // CHECK-NEXT: [[_:.+]] = constant
  // CHECK-NEXT: [[_:.+]] = constant

  %0 = leq %c2_si4, %c-3_si3 : (!firrtl.sint<4>, !firrtl.sint<3>) -> !firrtl.uint<1>
  %1 = leq %c-3_si3, %c2_si4 : (!firrtl.sint<3>, !firrtl.sint<4>) -> !firrtl.uint<1>
  %2 = leq %c2_ui4, %c5_ui3 : (!firrtl.uint<4>, !firrtl.uint<3>) -> !firrtl.uint<1>
  %3 = leq %c5_ui3, %c2_ui4 : (!firrtl.uint<3>, !firrtl.uint<4>) -> !firrtl.uint<1>
  %4 = leq %c2_si4, %c0_si0 : (!firrtl.sint<4>, !firrtl.sint<0>) -> !firrtl.uint<1>

  %5 = lt %c2_si4, %c-3_si3 : (!firrtl.sint<4>, !firrtl.sint<3>) -> !firrtl.uint<1>
  %6 = lt %c-3_si3, %c2_si4 : (!firrtl.sint<3>, !firrtl.sint<4>) -> !firrtl.uint<1>
  %7 = lt %c2_ui4, %c5_ui3 : (!firrtl.uint<4>, !firrtl.uint<3>) -> !firrtl.uint<1>
  %8 = lt %c5_ui3, %c2_ui4 : (!firrtl.uint<3>, !firrtl.uint<4>) -> !firrtl.uint<1>
  %9 = lt %c2_si4, %c0_si0 : (!firrtl.sint<4>, !firrtl.sint<0>) -> !firrtl.uint<1>

  %10 = geq %c2_si4, %c-3_si3 : (!firrtl.sint<4>, !firrtl.sint<3>) -> !firrtl.uint<1>
  %11 = geq %c-3_si3, %c2_si4 : (!firrtl.sint<3>, !firrtl.sint<4>) -> !firrtl.uint<1>
  %12 = geq %c2_ui4, %c5_ui3 : (!firrtl.uint<4>, !firrtl.uint<3>) -> !firrtl.uint<1>
  %13 = geq %c5_ui3, %c2_ui4 : (!firrtl.uint<3>, !firrtl.uint<4>) -> !firrtl.uint<1>
  %14 = geq %c2_si4, %c0_si0 : (!firrtl.sint<4>, !firrtl.sint<0>) -> !firrtl.uint<1>

  %15 = gt %c2_si4, %c-3_si3 : (!firrtl.sint<4>, !firrtl.sint<3>) -> !firrtl.uint<1>
  %16 = gt %c-3_si3, %c2_si4 : (!firrtl.sint<3>, !firrtl.sint<4>) -> !firrtl.uint<1>
  %17 = gt %c2_ui4, %c5_ui3 : (!firrtl.uint<4>, !firrtl.uint<3>) -> !firrtl.uint<1>
  %18 = gt %c5_ui3, %c2_ui4 : (!firrtl.uint<3>, !firrtl.uint<4>) -> !firrtl.uint<1>
  %19 = gt %c2_si4, %c0_si0 : (!firrtl.sint<4>, !firrtl.sint<0>) -> !firrtl.uint<1>

  connect %y0, %0 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %y1, %1 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %y2, %2 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %y3, %3 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %y4, %4 : !firrtl.uint<1>, !firrtl.uint<1>

  connect %y5, %5 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %y6, %6 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %y7, %7 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %y8, %8 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %y9, %9 : !firrtl.uint<1>, !firrtl.uint<1>

  connect %y10, %10 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %y11, %11 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %y12, %12 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %y13, %13 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %y14, %14 : !firrtl.uint<1>, !firrtl.uint<1>

  connect %y15, %15 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %y16, %16 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %y17, %17 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %y18, %18 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %y19, %19 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK-NEXT: strictconnect %y0, %c0_ui1
  // CHECK-NEXT: strictconnect %y1, %c1_ui1
  // CHECK-NEXT: strictconnect %y2, %c1_ui1
  // CHECK-NEXT: strictconnect %y3, %c0_ui1
  // CHECK-NEXT: strictconnect %y4, %c0_ui1

  // CHECK-NEXT: strictconnect %y5, %c0_ui1
  // CHECK-NEXT: strictconnect %y6, %c1_ui1
  // CHECK-NEXT: strictconnect %y7, %c1_ui1
  // CHECK-NEXT: strictconnect %y8, %c0_ui1
  // CHECK-NEXT: strictconnect %y9, %c0_ui1

  // CHECK-NEXT: strictconnect %y10, %c1_ui1
  // CHECK-NEXT: strictconnect %y11, %c0_ui1
  // CHECK-NEXT: strictconnect %y12, %c0_ui1
  // CHECK-NEXT: strictconnect %y13, %c1_ui1
  // CHECK-NEXT: strictconnect %y14, %c1_ui1

  // CHECK-NEXT: strictconnect %y15, %c1_ui1
  // CHECK-NEXT: strictconnect %y16, %c0_ui1
  // CHECK-NEXT: strictconnect %y17, %c0_ui1
  // CHECK-NEXT: strictconnect %y18, %c1_ui1
  // CHECK-NEXT: strictconnect %y19, %c1_ui1
}

// CHECK-LABEL: @zeroWidth(
// CHECK-NEXT:   %c0_ui2 = constant 0 : !firrtl.uint<2>
// CHECK-NEXT:   strictconnect %out, %c0_ui2 : !firrtl.uint<2>
// CHECK-NEXT:  }
firrtl.module @zeroWidth(out %out: !firrtl.uint<2>, in %in1 : !firrtl.uint<0>, in %in2 : !firrtl.uint<0>) {
  %add = add %in1, %in2 : (!firrtl.uint<0>, !firrtl.uint<0>) -> !firrtl.uint<1>
  %sub = sub %in1, %in2 : (!firrtl.uint<0>, !firrtl.uint<0>) -> !firrtl.uint<1>
  %mul = mul %in1, %in2 : (!firrtl.uint<0>, !firrtl.uint<0>) -> !firrtl.uint<0>
  %div = div %in1, %in2 : (!firrtl.uint<0>, !firrtl.uint<0>) -> !firrtl.uint<0>
  %rem = rem %in1, %in2 : (!firrtl.uint<0>, !firrtl.uint<0>) -> !firrtl.uint<0>
  %dshl = dshl %in1, %in2 : (!firrtl.uint<0>, !firrtl.uint<0>) -> !firrtl.uint<0>
  %dshlw = dshlw %in1, %in2 : (!firrtl.uint<0>, !firrtl.uint<0>) -> !firrtl.uint<0>
  %dshr = dshr %in1, %in2 : (!firrtl.uint<0>, !firrtl.uint<0>) -> !firrtl.uint<0>
  %and = and %in1, %in2 : (!firrtl.uint<0>, !firrtl.uint<0>) -> !firrtl.uint<0>
  %or = or %in1, %in2 : (!firrtl.uint<0>, !firrtl.uint<0>) -> !firrtl.uint<0>
  %xor = xor %in1, %in2 : (!firrtl.uint<0>, !firrtl.uint<0>) -> !firrtl.uint<0>
  %ret1 = cat %add, %sub : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<2>
  %ret2 = cat %ret1, %mul : (!firrtl.uint<2>, !firrtl.uint<0>) -> !firrtl.uint<2>
  %ret3 = cat %ret2, %div : (!firrtl.uint<2>, !firrtl.uint<0>) -> !firrtl.uint<2>
  %ret4 = cat %ret3, %rem : (!firrtl.uint<2>, !firrtl.uint<0>) -> !firrtl.uint<2>
  %ret5 = cat %ret4, %dshl : (!firrtl.uint<2>, !firrtl.uint<0>) -> !firrtl.uint<2>
  %ret6 = cat %ret5, %dshlw : (!firrtl.uint<2>, !firrtl.uint<0>) -> !firrtl.uint<2>
  %ret7 = cat %ret6, %dshr : (!firrtl.uint<2>, !firrtl.uint<0>) -> !firrtl.uint<2>
  %ret8 = cat %ret7, %and : (!firrtl.uint<2>, !firrtl.uint<0>) -> !firrtl.uint<2>
  %ret9 = cat %ret8, %or : (!firrtl.uint<2>, !firrtl.uint<0>) -> !firrtl.uint<2>
  %ret10 = cat %ret9, %xor : (!firrtl.uint<2>, !firrtl.uint<0>) -> !firrtl.uint<2>
  strictconnect %out, %ret10 : !firrtl.uint<2>
}

// CHECK-LABEL: @zeroWidthOperand(
// CHECK-NEXT:   %c0_ui0 = constant 0 : !firrtl.uint<0>
// CHECK-NEXT:   strictconnect %y6, %c0_ui0 : !firrtl.uint<0>
// CHECK-NEXT:   strictconnect %y8, %c0_ui0 : !firrtl.uint<0>
// CHECK-NEXT:   strictconnect %y9, %c0_ui0 : !firrtl.uint<0>
// CHECK-NEXT:   strictconnect %y12, %c0_ui0 : !firrtl.uint<0>
// CHECK-NEXT:   strictconnect %y14, %c0_ui0 : !firrtl.uint<0>
// CHECK-NEXT:  }
firrtl.module @zeroWidthOperand(
  in %in0 : !firrtl.uint<0>,
  in %in1 : !firrtl.uint<1>,
  out %y6: !firrtl.uint<0>,
  out %y8: !firrtl.uint<0>,
  out %y9: !firrtl.uint<0>,
  out %y12: !firrtl.uint<0>,
  out %y14: !firrtl.uint<0>
) {
  %div1 = div %in0, %in1 : (!firrtl.uint<0>, !firrtl.uint<1>) -> !firrtl.uint<0>
  %rem1 = rem %in0, %in1 : (!firrtl.uint<0>, !firrtl.uint<1>) -> !firrtl.uint<0>
  %rem2 = rem %in1, %in0 : (!firrtl.uint<1>, !firrtl.uint<0>) -> !firrtl.uint<0>
  %dshlw1 = dshlw %in0, %in1 : (!firrtl.uint<0>, !firrtl.uint<1>) -> !firrtl.uint<0>
  %dshr1 = dshr %in0, %in1 : (!firrtl.uint<0>, !firrtl.uint<1>) -> !firrtl.uint<0>

  strictconnect %y6, %div1 : !firrtl.uint<0>
  strictconnect %y8, %rem1 : !firrtl.uint<0>
  strictconnect %y9, %rem2 : !firrtl.uint<0>
  strictconnect %y12, %dshlw1 : !firrtl.uint<0>
  strictconnect %y14, %dshr1 : !firrtl.uint<0>
}

// CHECK-LABEL: @add_cst_prop1
// CHECK-NEXT:   %c11_ui9 = constant 11 : !firrtl.uint<9>
// CHECK-NEXT:   strictconnect %out_b, %c11_ui9 : !firrtl.uint<9>
// CHECK-NEXT:  }
firrtl.module @add_cst_prop1(out %out_b: !firrtl.uint<9>) {
  %c6_ui7 = constant 6 : !firrtl.uint<7>
  %_tmp_a = wire droppable_name : !firrtl.uint<7>
  %c5_ui8 = constant 5 : !firrtl.uint<8>
  connect %_tmp_a, %c6_ui7 : !firrtl.uint<7>, !firrtl.uint<7>
  %add = add %_tmp_a, %c5_ui8 : (!firrtl.uint<7>, !firrtl.uint<8>) -> !firrtl.uint<9>
  connect %out_b, %add : !firrtl.uint<9>, !firrtl.uint<9>
}

// CHECK-LABEL: @add_cst_prop2
// CHECK-NEXT:   %c-1_si9 = constant -1 : !firrtl.sint<9>
// CHECK-NEXT:   strictconnect %out_b, %c-1_si9 : !firrtl.sint<9>
// CHECK-NEXT:  }
firrtl.module @add_cst_prop2(out %out_b: !firrtl.sint<9>) {
  %c6_ui7 = constant -6 : !firrtl.sint<7>
  %_tmp_a = wire droppable_name: !firrtl.sint<7>
  %c5_ui8 = constant 5 : !firrtl.sint<8>
  connect %_tmp_a, %c6_ui7 : !firrtl.sint<7>, !firrtl.sint<7>
  %add = add %_tmp_a, %c5_ui8 : (!firrtl.sint<7>, !firrtl.sint<8>) -> !firrtl.sint<9>
  connect %out_b, %add : !firrtl.sint<9>, !firrtl.sint<9>
}

// CHECK-LABEL: @add_cst_prop3
// CHECK-NEXT:   %c-2_si4 = constant -2 : !firrtl.sint<4>
// CHECK-NEXT:   strictconnect %out_b, %c-2_si4 : !firrtl.sint<4>
// CHECK-NEXT:  }
firrtl.module @add_cst_prop3(out %out_b: !firrtl.sint<4>) {
  %c1_si2 = constant -1 : !firrtl.sint<2>
  %_tmp_a = wire droppable_name : !firrtl.sint<2>
  %c1_si3 = constant -1 : !firrtl.sint<3>
  connect %_tmp_a, %c1_si2 : !firrtl.sint<2>, !firrtl.sint<2>
  %add = add %_tmp_a, %c1_si3 : (!firrtl.sint<2>, !firrtl.sint<3>) -> !firrtl.sint<4>
  connect %out_b, %add : !firrtl.sint<4>, !firrtl.sint<4>
}

// CHECK-LABEL: @add_cst_prop5
// CHECK: %[[pad:.+]] = pad %tmp_a, 5
// CHECK-NEXT: strictconnect %out_b, %[[pad]]
// CHECK-NEXT: %[[pad:.+]] = pad %tmp_a, 5
// CHECK-NEXT: strictconnect %out_b, %[[pad]]
firrtl.module @add_cst_prop5(out %out_b: !firrtl.uint<5>) {
  %tmp_a = wire : !firrtl.uint<4>
  %c0_ui4 = constant 0 : !firrtl.uint<4>
  %add = add %tmp_a, %c0_ui4 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<5>
  connect %out_b, %add : !firrtl.uint<5>, !firrtl.uint<5>
  %add2 = add %c0_ui4, %tmp_a : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<5>
  connect %out_b, %add2 : !firrtl.uint<5>, !firrtl.uint<5>
}

// CHECK-LABEL: @add_double
// CHECK: %[[shl:.+]] = shl %in, 1
// CHECK-NEXT: strictconnect %out, %[[shl]]
firrtl.module @add_double(out %out: !firrtl.uint<5>, in %in: !firrtl.uint<4>) {
  %add = add %in, %in : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<5>
  connect %out, %add : !firrtl.uint<5>, !firrtl.uint<5>
}

// CHECK-LABEL: @add_narrow
// CHECK-NEXT: %[[add1:.+]] = add %in2, %in1
// CHECK-NEXT: %[[pad1:.+]] = pad %[[add1]], 7
// CHECK-NEXT: %[[add2:.+]] = add %in2, %in1
// CHECK-NEXT: %[[pad2:.+]] = pad %[[add2]], 7
// CHECK-NEXT: %[[add3:.+]] = add %in1, %in2
// CHECK-NEXT: %[[pad3:.+]] = pad %[[add3]], 7
// CHECK-NEXT: strictconnect %out1, %[[pad1]]
// CHECK-NEXT: strictconnect %out2, %[[pad2]]
// CHECK-NEXT: strictconnect %out3, %[[pad3]]
firrtl.module @add_narrow(out %out1: !firrtl.uint<7>, out %out2: !firrtl.uint<7>, out %out3: !firrtl.uint<7>, in %in1: !firrtl.uint<4>, in %in2: !firrtl.uint<2>) {
  %t1 = pad %in1, 6 : (!firrtl.uint<4>) -> !firrtl.uint<6>
  %t2 = pad %in2, 6 : (!firrtl.uint<2>) -> !firrtl.uint<6>
  %add1 = add %t1, %t2 : (!firrtl.uint<6>, !firrtl.uint<6>) -> !firrtl.uint<7>
  %add2 = add %in1, %t2 : (!firrtl.uint<4>, !firrtl.uint<6>) -> !firrtl.uint<7>
  %add3 = add %t1, %in2 : (!firrtl.uint<6>, !firrtl.uint<2>) -> !firrtl.uint<7>
  strictconnect %out1, %add1 : !firrtl.uint<7>
  strictconnect %out2, %add2 : !firrtl.uint<7>
  strictconnect %out3, %add3 : !firrtl.uint<7>
}

// CHECK-LABEL: @adds_narrow
// CHECK-NEXT: %[[add1:.+]] = add %in2, %in1
// CHECK-NEXT: %[[pad1:.+]] = pad %[[add1]], 7
// CHECK-NEXT: %[[add2:.+]] = add %in2, %in1
// CHECK-NEXT: %[[pad2:.+]] = pad %[[add2]], 7
// CHECK-NEXT: %[[add3:.+]] = add %in1, %in2
// CHECK-NEXT: %[[pad3:.+]] = pad %[[add3]], 7
// CHECK-NEXT: strictconnect %out1, %[[pad1]]
// CHECK-NEXT: strictconnect %out2, %[[pad2]]
// CHECK-NEXT: strictconnect %out3, %[[pad3]]
firrtl.module @adds_narrow(out %out1: !firrtl.sint<7>, out %out2: !firrtl.sint<7>, out %out3: !firrtl.sint<7>, in %in1: !firrtl.sint<4>, in %in2: !firrtl.sint<2>) {
  %t1 = pad %in1, 6 : (!firrtl.sint<4>) -> !firrtl.sint<6>
  %t2 = pad %in2, 6 : (!firrtl.sint<2>) -> !firrtl.sint<6>
  %add1 = add %t1, %t2 : (!firrtl.sint<6>, !firrtl.sint<6>) -> !firrtl.sint<7>
  %add2 = add %in1, %t2 : (!firrtl.sint<4>, !firrtl.sint<6>) -> !firrtl.sint<7>
  %add3 = add %t1, %in2 : (!firrtl.sint<6>, !firrtl.sint<2>) -> !firrtl.sint<7>
  strictconnect %out1, %add1 : !firrtl.sint<7>
  strictconnect %out2, %add2 : !firrtl.sint<7>
  strictconnect %out3, %add3 : !firrtl.sint<7>
}

// CHECK-LABEL: @sub_narrow
// CHECK-NEXT: %[[add1:.+]] = sub %in1, %in2 : (!firrtl.uint<4>, !firrtl.uint<2>) -> !firrtl.uint<5>
// CHECK-NEXT: %[[pad1:.+]] = pad %[[add1]], 7 : (!firrtl.uint<5>) -> !firrtl.uint<7>
// CHECK-NEXT: %[[add2:.+]] = sub %in1, %in2 : (!firrtl.uint<4>, !firrtl.uint<2>) -> !firrtl.uint<5>
// CHECK-NEXT: %[[pad2:.+]] = pad %[[add2]], 7 : (!firrtl.uint<5>) -> !firrtl.uint<7>
// CHECK-NEXT: %[[add3:.+]] = sub %in1, %in2 : (!firrtl.uint<4>, !firrtl.uint<2>) -> !firrtl.uint<5>
// CHECK-NEXT: %[[pad3:.+]] = pad %[[add3]], 7 : (!firrtl.uint<5>) -> !firrtl.uint<7>
// CHECK-NEXT: strictconnect %out1, %[[pad1]]
// CHECK-NEXT: strictconnect %out2, %[[pad2]]
// CHECK-NEXT: strictconnect %out3, %[[pad3]]
firrtl.module @sub_narrow(out %out1: !firrtl.uint<7>, out %out2: !firrtl.uint<7>, out %out3: !firrtl.uint<7>, in %in1: !firrtl.uint<4>, in %in2: !firrtl.uint<2>) {
  %t1 = pad %in1, 6 : (!firrtl.uint<4>) -> !firrtl.uint<6>
  %t2 = pad %in2, 6 : (!firrtl.uint<2>) -> !firrtl.uint<6>
  %add1 = sub %t1, %t2 : (!firrtl.uint<6>, !firrtl.uint<6>) -> !firrtl.uint<7>
  %add2 = sub %in1, %t2 : (!firrtl.uint<4>, !firrtl.uint<6>) -> !firrtl.uint<7>
  %add3 = sub %t1, %in2 : (!firrtl.uint<6>, !firrtl.uint<2>) -> !firrtl.uint<7>
  strictconnect %out1, %add1 : !firrtl.uint<7>
  strictconnect %out2, %add2 : !firrtl.uint<7>
  strictconnect %out3, %add3 : !firrtl.uint<7>
}

// CHECK-LABEL: @subs_narrow
// CHECK-NEXT: %[[add1:.+]] = sub %in1, %in2 : (!firrtl.sint<4>, !firrtl.sint<2>) -> !firrtl.sint<5>
// CHECK-NEXT: %[[pad1:.+]] = pad %[[add1]], 7 : (!firrtl.sint<5>) -> !firrtl.sint<7>
// CHECK-NEXT: %[[add2:.+]] = sub %in1, %in2 : (!firrtl.sint<4>, !firrtl.sint<2>) -> !firrtl.sint<5>
// CHECK-NEXT: %[[pad2:.+]] = pad %[[add2]], 7 : (!firrtl.sint<5>) -> !firrtl.sint<7>
// CHECK-NEXT: %[[add3:.+]] = sub %in1, %in2 : (!firrtl.sint<4>, !firrtl.sint<2>) -> !firrtl.sint<5>
// CHECK-NEXT: %[[pad3:.+]] = pad %[[add3]], 7 : (!firrtl.sint<5>) -> !firrtl.sint<7>
// CHECK-NEXT: strictconnect %out1, %[[pad1]]
// CHECK-NEXT: strictconnect %out2, %[[pad2]]
// CHECK-NEXT: strictconnect %out3, %[[pad3]]
firrtl.module @subs_narrow(out %out1: !firrtl.sint<7>, out %out2: !firrtl.sint<7>, out %out3: !firrtl.sint<7>, in %in1: !firrtl.sint<4>, in %in2: !firrtl.sint<2>) {
  %t1 = pad %in1, 6 : (!firrtl.sint<4>) -> !firrtl.sint<6>
  %t2 = pad %in2, 6 : (!firrtl.sint<2>) -> !firrtl.sint<6>
  %add1 = sub %t1, %t2 : (!firrtl.sint<6>, !firrtl.sint<6>) -> !firrtl.sint<7>
  %add2 = sub %in1, %t2 : (!firrtl.sint<4>, !firrtl.sint<6>) -> !firrtl.sint<7>
  %add3 = sub %t1, %in2 : (!firrtl.sint<6>, !firrtl.sint<2>) -> !firrtl.sint<7>
  strictconnect %out1, %add1 : !firrtl.sint<7>
  strictconnect %out2, %add2 : !firrtl.sint<7>
  strictconnect %out3, %add3 : !firrtl.sint<7>
}

// CHECK-LABEL: @sub_cst_prop1
// CHECK-NEXT:      %c1_ui9 = constant 1 : !firrtl.uint<9>
// CHECK-NEXT:      strictconnect %out_b, %c1_ui9 : !firrtl.uint<9>
// CHECK-NEXT:  }
firrtl.module @sub_cst_prop1(out %out_b: !firrtl.uint<9>) {
  %c6_ui7 = constant 6 : !firrtl.uint<7>
  %_tmp_a = wire droppable_name : !firrtl.uint<7>
  %c5_ui8 = constant 5 : !firrtl.uint<8>
  connect %_tmp_a, %c6_ui7 : !firrtl.uint<7>, !firrtl.uint<7>
  %add = sub %_tmp_a, %c5_ui8 : (!firrtl.uint<7>, !firrtl.uint<8>) -> !firrtl.uint<9>
  connect %out_b, %add : !firrtl.uint<9>, !firrtl.uint<9>
}

// CHECK-LABEL: @sub_cst_prop2
// CHECK-NEXT:      %c-11_si9 = constant -11 : !firrtl.sint<9>
// CHECK-NEXT:      strictconnect %out_b, %c-11_si9 : !firrtl.sint<9>
// CHECK-NEXT:  }
firrtl.module @sub_cst_prop2(out %out_b: !firrtl.sint<9>) {
  %c6_ui7 = constant -6 : !firrtl.sint<7>
  %_tmp_a = wire droppable_name : !firrtl.sint<7>
  %c5_ui8 = constant 5 : !firrtl.sint<8>
  connect %_tmp_a, %c6_ui7 : !firrtl.sint<7>, !firrtl.sint<7>
  %add = sub %_tmp_a, %c5_ui8 : (!firrtl.sint<7>, !firrtl.sint<8>) -> !firrtl.sint<9>
  connect %out_b, %add : !firrtl.sint<9>, !firrtl.sint<9>
}

// CHECK-LABEL: @sub_double
// CHECK: %[[cst:.+]] = constant 0 : !firrtl.uint<5>
// CHECK-NEXT: strictconnect %out, %[[cst]]
firrtl.module @sub_double(out %out: !firrtl.uint<5>, in %in: !firrtl.uint<4>) {
  %add = sub %in, %in : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<5>
  connect %out, %add : !firrtl.uint<5>, !firrtl.uint<5>
}

// CHECK-LABEL: @mul_cst_prop1
// CHECK-NEXT:      %c30_ui15 = constant 30 : !firrtl.uint<15>
// CHECK-NEXT:      strictconnect %out_b, %c30_ui15 : !firrtl.uint<15>
// CHECK-NEXT:  }
firrtl.module @mul_cst_prop1(out %out_b: !firrtl.uint<15>) {
  %c6_ui7 = constant 6 : !firrtl.uint<7>
  %_tmp_a = wire droppable_name : !firrtl.uint<7>
  %c5_ui8 = constant 5 : !firrtl.uint<8>
  connect %_tmp_a, %c6_ui7 : !firrtl.uint<7>, !firrtl.uint<7>
  %add = mul %_tmp_a, %c5_ui8 : (!firrtl.uint<7>, !firrtl.uint<8>) -> !firrtl.uint<15>
  connect %out_b, %add : !firrtl.uint<15>, !firrtl.uint<15>
}

// CHECK-LABEL: @mul_cst_prop2
// CHECK-NEXT:      %c-30_si15 = constant -30 : !firrtl.sint<15>
// CHECK-NEXT:      strictconnect %out_b, %c-30_si15 : !firrtl.sint<15>
// CHECK-NEXT:  }
firrtl.module @mul_cst_prop2(out %out_b: !firrtl.sint<15>) {
  %c6_ui7 = constant -6 : !firrtl.sint<7>
  %_tmp_a = wire droppable_name : !firrtl.sint<7>
  %c5_ui8 = constant 5 : !firrtl.sint<8>
  connect %_tmp_a, %c6_ui7 : !firrtl.sint<7>, !firrtl.sint<7>
  %add = mul %_tmp_a, %c5_ui8 : (!firrtl.sint<7>, !firrtl.sint<8>) -> !firrtl.sint<15>
  connect %out_b, %add : !firrtl.sint<15>, !firrtl.sint<15>
}

// CHECK-LABEL: @mul_cst_prop3
// CHECK-NEXT:      %c30_si15 = constant 30 : !firrtl.sint<15>
// CHECK-NEXT:      strictconnect %out_b, %c30_si15 : !firrtl.sint<15>
// CHECK-NEXT:  }
firrtl.module @mul_cst_prop3(out %out_b: !firrtl.sint<15>) {
  %c6_ui7 = constant -6 : !firrtl.sint<7>
  %_tmp_a = wire droppable_name : !firrtl.sint<7>
  %c5_ui8 = constant -5 : !firrtl.sint<8>
  connect %_tmp_a, %c6_ui7 : !firrtl.sint<7>, !firrtl.sint<7>
  %add = mul %_tmp_a, %c5_ui8 : (!firrtl.sint<7>, !firrtl.sint<8>) -> !firrtl.sint<15>
  connect %out_b, %add : !firrtl.sint<15>, !firrtl.sint<15>
}

// CHECK-LABEL: module @MuxCanon
firrtl.module @MuxCanon(in %c1: !firrtl.uint<1>, in %c2: !firrtl.uint<1>, in %d1: !firrtl.uint<5>, in %d2: !firrtl.uint<5>, in %d3: !firrtl.uint<5>, out %foo: !firrtl.uint<5>, out %foo2: !firrtl.uint<5>, out %foo3: !firrtl.uint<5>, out %foo4: !firrtl.uint<5>, out %foo5: !firrtl.uint<10>, out %foo6: !firrtl.uint<10>) {
  %0 = mux(%c1, %d2, %d3) : (!firrtl.uint<1>, !firrtl.uint<5>, !firrtl.uint<5>) -> !firrtl.uint<5>
  %1 = mux(%c1, %d1, %0) : (!firrtl.uint<1>, !firrtl.uint<5>, !firrtl.uint<5>) -> !firrtl.uint<5>
  %2 = mux(%c1, %0, %d1) : (!firrtl.uint<1>, !firrtl.uint<5>, !firrtl.uint<5>) -> !firrtl.uint<5>
  %3 = mux(%c1, %d1, %d2) : (!firrtl.uint<1>, !firrtl.uint<5>, !firrtl.uint<5>) -> !firrtl.uint<5>
  %4 = mux(%c2, %3, %d2) : (!firrtl.uint<1>, !firrtl.uint<5>, !firrtl.uint<5>) -> !firrtl.uint<5>
  %5 = mux(%c2, %d1, %3) : (!firrtl.uint<1>, !firrtl.uint<5>, !firrtl.uint<5>) -> !firrtl.uint<5>
  %6 = cat %d1, %d2 : (!firrtl.uint<5>, !firrtl.uint<5>) -> !firrtl.uint<10>
  %7 = cat %d1, %d3 : (!firrtl.uint<5>, !firrtl.uint<5>) -> !firrtl.uint<10>
  %8 = cat %d1, %d2 : (!firrtl.uint<5>, !firrtl.uint<5>) -> !firrtl.uint<10>
  %9 = cat %d3, %d2 : (!firrtl.uint<5>, !firrtl.uint<5>) -> !firrtl.uint<10>
  %10 = mux(%c1, %6, %7) : (!firrtl.uint<1>, !firrtl.uint<10>, !firrtl.uint<10>) -> !firrtl.uint<10>
  %11 = mux(%c2, %8, %9) : (!firrtl.uint<1>, !firrtl.uint<10>, !firrtl.uint<10>) -> !firrtl.uint<10>
  connect %foo, %1 : !firrtl.uint<5>, !firrtl.uint<5>
  connect %foo2, %2 : !firrtl.uint<5>, !firrtl.uint<5>
  connect %foo3, %4 : !firrtl.uint<5>, !firrtl.uint<5>
  connect %foo4, %5 : !firrtl.uint<5>, !firrtl.uint<5>
  connect %foo5, %10 : !firrtl.uint<10>, !firrtl.uint<10>
  connect %foo6, %11 : !firrtl.uint<10>, !firrtl.uint<10>
  // CHECK: mux(%c1, %d1, %d3) : (!firrtl.uint<1>, !firrtl.uint<5>, !firrtl.uint<5>) -> !firrtl.uint<5>
  // CHECK: mux(%c1, %d2, %d1) : (!firrtl.uint<1>, !firrtl.uint<5>, !firrtl.uint<5>) -> !firrtl.uint<5>
  // CHECK: %[[and:.*]] = and %c2, %c1
  // CHECK: %[[andmux:.*]] = mux(%[[and]], %d1, %d2)
  // CHECK: %[[or:.*]] = or %c2, %c1
  // CHECK: %[[ormux:.*]] = mux(%[[or]], %d1, %d2)
  // CHECK: %[[mux1:.*]] = mux(%c1, %d2, %d3)
  // CHECK: cat %d1, %[[mux1]]
  // CHECK: %[[mux2:.*]] = mux(%c2, %d1, %d3)
  // CHECK: cat %[[mux2]], %d2
}

// CHECK-LABEL: module @MuxShorten
firrtl.module @MuxShorten(
  in %c1: !firrtl.uint<1>, in %c2: !firrtl.uint<1>,
  in %d1: !firrtl.uint<5>, in %d2: !firrtl.uint<5>,
  in %d3: !firrtl.uint<5>, in %d4: !firrtl.uint<5>,
  in %d5: !firrtl.uint<5>, in %d6: !firrtl.uint<5>,
  out %foo: !firrtl.uint<5>, out %foo2: !firrtl.uint<5>) {

  %0 = mux(%c1, %d2, %d3) : (!firrtl.uint<1>, !firrtl.uint<5>, !firrtl.uint<5>) -> !firrtl.uint<5>
  %1 = mux(%c2, %0, %d1) : (!firrtl.uint<1>, !firrtl.uint<5>, !firrtl.uint<5>) -> !firrtl.uint<5>
  %2 = mux(%c1, %d4, %d5) : (!firrtl.uint<1>, !firrtl.uint<5>, !firrtl.uint<5>) -> !firrtl.uint<5>
  %3 = mux(%c2, %2, %d6) : (!firrtl.uint<1>, !firrtl.uint<5>, !firrtl.uint<5>) -> !firrtl.uint<5>
  %11 = mux(%c1, %1, %3) : (!firrtl.uint<1>, !firrtl.uint<5>, !firrtl.uint<5>) -> !firrtl.uint<5>
  connect %foo, %11 : !firrtl.uint<5>, !firrtl.uint<5>
  connect %foo2, %3 : !firrtl.uint<5>, !firrtl.uint<5>

  // CHECK: %[[n1:.*]] = mux(%c2, %d2, %d1)
  // CHECK: %[[rem1:.*]] = mux(%c1, %d4, %d5)
  // CHECK: %[[rem:.*]] = mux(%c2, %[[rem1]], %d6)
  // CHECK: %[[n2:.*]] = mux(%c2, %d5, %d6)
  // CHECK: %[[prim:.*]] = mux(%c1, %[[n1]], %[[n2]])
  // CHECK: strictconnect %foo, %[[prim]]
  // CHECK: strictconnect %foo2, %[[rem]]
}


// CHECK-LABEL: module @RegresetToReg
firrtl.module @RegresetToReg(in %clock: !firrtl.clock, in %dummy : !firrtl.uint<1>, out %foo1: !firrtl.uint<1>, out %foo2: !firrtl.uint<1>) {
  %c1_ui1 = constant 1 : !firrtl.uint<1>
  %c0_ui1 = constant 0 : !firrtl.uint<1>
  %zero_asyncreset = asAsyncReset %c0_ui1 : (!firrtl.uint<1>) -> !firrtl.asyncreset
  %one_asyncreset = asAsyncReset %c1_ui1 : (!firrtl.uint<1>) -> !firrtl.asyncreset
  // CHECK: %bar1 = reg %clock : !firrtl.clock, !firrtl.uint<1>
  // CHECK: strictconnect %foo2, %dummy : !firrtl.uint<1>
  %bar1 = regreset %clock, %zero_asyncreset, %dummy : !firrtl.clock, !firrtl.asyncreset, !firrtl.uint<1>, !firrtl.uint<1>
  %bar2 = regreset %clock, %one_asyncreset, %dummy : !firrtl.clock, !firrtl.asyncreset, !firrtl.uint<1>, !firrtl.uint<1>

  strictconnect %bar2, %bar1 : !firrtl.uint<1> // Force a use to trigger a crash on a sink replacement

  connect %foo1, %bar1 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %foo2, %bar2 : !firrtl.uint<1>, !firrtl.uint<1>
}

// CHECK-LABEL: module @ForceableRegResetToNode
// Correctness, revisit if this is "valid" if forceable.
firrtl.module @ForceableRegResetToNode(in %clock: !firrtl.clock, in %dummy : !firrtl.uint<1>, out %foo: !firrtl.uint<1>, out %ref : !firrtl.rwprobe<uint<1>>) {
  %c1_ui1 = constant 1 : !firrtl.uint<1>
  %one_asyncreset = asAsyncReset %c1_ui1 : (!firrtl.uint<1>) -> !firrtl.asyncreset
  // CHECK: %reg, %reg_ref = node %dummy forceable : !firrtl.uint<1>
  %reg, %reg_f = regreset %clock, %one_asyncreset, %dummy forceable : !firrtl.clock, !firrtl.asyncreset, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.rwprobe<uint<1>>
  ref.define %ref, %reg_f: !firrtl.rwprobe<uint<1>>

  connect %reg, %dummy: !firrtl.uint<1>, !firrtl.uint<1>
  connect %foo, %reg: !firrtl.uint<1>, !firrtl.uint<1>
}

// https://github.com/llvm/circt/issues/929
// CHECK-LABEL: module @MuxInvalidTypeOpt
firrtl.module @MuxInvalidTypeOpt(in %in : !firrtl.uint<1>, out %out : !firrtl.uint<4>) {
  %c7_ui4 = constant 7 : !firrtl.uint<4>
  %c1_ui2 = constant 1 : !firrtl.uint<2>
  %c0_ui2 = constant 0 : !firrtl.uint<2>
  %0 = mux (%in, %c7_ui4, %c0_ui2) : (!firrtl.uint<1>, !firrtl.uint<4>, !firrtl.uint<2>) -> !firrtl.uint<4>
  %1 = mux (%in, %c1_ui2, %c7_ui4) : (!firrtl.uint<1>, !firrtl.uint<2>, !firrtl.uint<4>) -> !firrtl.uint<4>
  connect %out, %0 : !firrtl.uint<4>, !firrtl.uint<4>
  connect %out, %1 : !firrtl.uint<4>, !firrtl.uint<4>
}
// CHECK: mux(%in, %c7_ui4, %c0_ui4) : (!firrtl.uint<1>, !firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
// CHECK: mux(%in, %c1_ui4, %c7_ui4) : (!firrtl.uint<1>, !firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>

// CHECK-LABEL: module @issue1100
// CHECK: strictconnect %tmp62, %c1_ui1
  module @issue1100(out %tmp62: !firrtl.uint<1>) {
    %c-1_si2 = constant -1 : !firrtl.sint<2>
    %0 = orr %c-1_si2 : (!firrtl.sint<2>) -> !firrtl.uint<1>
    connect %tmp62, %0 : !firrtl.uint<1>, !firrtl.uint<1>
  }

// CHECK-LABEL: module @zeroWidthMem
// CHECK-NEXT:  }
firrtl.module @zeroWidthMem(in %clock: !firrtl.clock) {
  // FIXME(Issue #1125): Add a test for zero width memory elimination.
}

// CHECK-LABEL: module @issue1116
firrtl.module @issue1116(out %z: !firrtl.uint<1>) {
  %c844336_ui = constant 844336 : !firrtl.uint
  %c161_ui8 = constant 161 : !firrtl.uint<8>
  %0 = leq %c844336_ui, %c161_ui8 : (!firrtl.uint, !firrtl.uint<8>) -> !firrtl.uint<1>
  // CHECK: strictconnect %z, %c0_ui1
  connect %z, %0 : !firrtl.uint<1>, !firrtl.uint<1>
}

// Sign casts must not be folded into unsized constants.
// CHECK-LABEL: module @issue1118
firrtl.module @issue1118(out %z0: !firrtl.uint, out %z1: !firrtl.sint) {
  // CHECK: %0 = asUInt %c4232_si : (!firrtl.sint) -> !firrtl.uint
  // CHECK: %1 = asSInt %c4232_ui : (!firrtl.uint) -> !firrtl.sint
  // CHECK: connect %z0, %0 : !firrtl.uint, !firrtl.uint
  // CHECK: connect %z1, %1 : !firrtl.sint, !firrtl.sint
  %c4232_si = constant 4232 : !firrtl.sint
  %c4232_ui = constant 4232 : !firrtl.uint
  %0 = asUInt %c4232_si : (!firrtl.sint) -> !firrtl.uint
  %1 = asSInt %c4232_ui : (!firrtl.uint) -> !firrtl.sint
  connect %z0, %0 : !firrtl.uint, !firrtl.uint
  connect %z1, %1 : !firrtl.sint, !firrtl.sint
}

// CHECK-LABEL: module @issue1139
firrtl.module @issue1139(out %z: !firrtl.uint<4>) {
  // CHECK-NEXT: %c0_ui4 = constant 0 : !firrtl.uint<4>
  // CHECK-NEXT: strictconnect %z, %c0_ui4 : !firrtl.uint<4>
  %c4_ui4 = constant 4 : !firrtl.uint<4>
  %c674_ui = constant 674 : !firrtl.uint
  %0 = dshr %c4_ui4, %c674_ui : (!firrtl.uint<4>, !firrtl.uint) -> !firrtl.uint<4>
  connect %z, %0 : !firrtl.uint<4>, !firrtl.uint<4>
}

// CHECK-LABEL: module @issue1142
firrtl.module @issue1142(in %cond: !firrtl.uint<1>, out %z: !firrtl.uint) {
  %c0_ui1 = constant 0 : !firrtl.uint<1>
  %c1_ui1 = constant 1 : !firrtl.uint<1>
  %c42_ui = constant 42 : !firrtl.uint
  %c43_ui = constant 43 : !firrtl.uint

  // Don't fold away constant selects if widths are unknown.
  // CHECK: %0 = mux(%c0_ui1, %c42_ui, %c43_ui) : (!firrtl.uint<1>, !firrtl.uint, !firrtl.uint) -> !firrtl.uint
  // CHECK: %1 = mux(%c1_ui1, %c42_ui, %c43_ui) : (!firrtl.uint<1>, !firrtl.uint, !firrtl.uint) -> !firrtl.uint
  %0 = mux(%c0_ui1, %c42_ui, %c43_ui) : (!firrtl.uint<1>, !firrtl.uint, !firrtl.uint) -> !firrtl.uint
  %1 = mux(%c1_ui1, %c42_ui, %c43_ui) : (!firrtl.uint<1>, !firrtl.uint, !firrtl.uint) -> !firrtl.uint

  // Don't fold nested muxes with same condition if widths are unknown.
  // CHECK: %2 = mux(%cond, %c42_ui, %c43_ui) : (!firrtl.uint<1>, !firrtl.uint, !firrtl.uint) -> !firrtl.uint
  // CHECK: %3 = mux(%cond, %2, %c43_ui) : (!firrtl.uint<1>, !firrtl.uint, !firrtl.uint) -> !firrtl.uint
  // CHECK: %4 = mux(%cond, %c42_ui, %2) : (!firrtl.uint<1>, !firrtl.uint, !firrtl.uint) -> !firrtl.uint
  %2 = mux(%cond, %c42_ui, %c43_ui) : (!firrtl.uint<1>, !firrtl.uint, !firrtl.uint) -> !firrtl.uint
  %3 = mux(%cond, %2, %c43_ui) : (!firrtl.uint<1>, !firrtl.uint, !firrtl.uint) -> !firrtl.uint
  %4 = mux(%cond, %c42_ui, %2) : (!firrtl.uint<1>, !firrtl.uint, !firrtl.uint) -> !firrtl.uint

  connect %z, %0 : !firrtl.uint, !firrtl.uint
  connect %z, %1 : !firrtl.uint, !firrtl.uint
  connect %z, %3 : !firrtl.uint, !firrtl.uint
  connect %z, %4 : !firrtl.uint, !firrtl.uint
}

// CHECK-LABEL: module @PadMuxOperands
firrtl.module @PadMuxOperands(
  in %cond: !firrtl.uint<1>,
  in %ui: !firrtl.uint,
  in %ui11: !firrtl.uint<11>,
  in %ui17: !firrtl.uint<17>,
  out %z: !firrtl.uint
) {
  %c0_ui1 = constant 0 : !firrtl.uint<1>
  %c1_ui1 = constant 1 : !firrtl.uint<1>

  // Smaller operand should pad to result width.
  // CHECK: %0 = pad %ui11, 17 : (!firrtl.uint<11>) -> !firrtl.uint<17>
  // CHECK: %1 = mux(%cond, %0, %ui17) : (!firrtl.uint<1>, !firrtl.uint<17>, !firrtl.uint<17>) -> !firrtl.uint<17>
  // CHECK: %2 = pad %ui11, 17 : (!firrtl.uint<11>) -> !firrtl.uint<17>
  // CHECK: %3 = mux(%cond, %ui17, %2) : (!firrtl.uint<1>, !firrtl.uint<17>, !firrtl.uint<17>) -> !firrtl.uint<17>
  %0 = mux(%cond, %ui11, %ui17) : (!firrtl.uint<1>, !firrtl.uint<11>, !firrtl.uint<17>) -> !firrtl.uint<17>
  %1 = mux(%cond, %ui17, %ui11) : (!firrtl.uint<1>, !firrtl.uint<17>, !firrtl.uint<11>) -> !firrtl.uint<17>

  // Unknown result width should prevent padding.
  // CHECK: %4 = mux(%cond, %ui11, %ui) : (!firrtl.uint<1>, !firrtl.uint<11>, !firrtl.uint) -> !firrtl.uint
  // CHECK: %5 = mux(%cond, %ui, %ui11) : (!firrtl.uint<1>, !firrtl.uint, !firrtl.uint<11>) -> !firrtl.uint
  %2 = mux(%cond, %ui11, %ui) : (!firrtl.uint<1>, !firrtl.uint<11>, !firrtl.uint) -> !firrtl.uint
  %3 = mux(%cond, %ui, %ui11) : (!firrtl.uint<1>, !firrtl.uint, !firrtl.uint<11>) -> !firrtl.uint

  // Padding to equal width operands should enable constant-select folds.
  // CHECK: %6 = pad %ui11, 17 : (!firrtl.uint<11>) -> !firrtl.uint<17>
  // CHECK: %7 = pad %ui11, 17 : (!firrtl.uint<11>) -> !firrtl.uint<17>
  // CHECK: connect %z, %ui17 : !firrtl.uint, !firrtl.uint<17>
  // CHECK: connect %z, %6 : !firrtl.uint, !firrtl.uint<17>
  // CHECK: connect %z, %7 : !firrtl.uint, !firrtl.uint<17>
  // CHECK: connect %z, %ui17 : !firrtl.uint, !firrtl.uint<17>
  %4 = mux(%c0_ui1, %ui11, %ui17) : (!firrtl.uint<1>, !firrtl.uint<11>, !firrtl.uint<17>) -> !firrtl.uint<17>
  %5 = mux(%c0_ui1, %ui17, %ui11) : (!firrtl.uint<1>, !firrtl.uint<17>, !firrtl.uint<11>) -> !firrtl.uint<17>
  %6 = mux(%c1_ui1, %ui11, %ui17) : (!firrtl.uint<1>, !firrtl.uint<11>, !firrtl.uint<17>) -> !firrtl.uint<17>
  %7 = mux(%c1_ui1, %ui17, %ui11) : (!firrtl.uint<1>, !firrtl.uint<17>, !firrtl.uint<11>) -> !firrtl.uint<17>

  connect %z, %0 : !firrtl.uint, !firrtl.uint<17>
  connect %z, %1 : !firrtl.uint, !firrtl.uint<17>
  connect %z, %2 : !firrtl.uint, !firrtl.uint
  connect %z, %3 : !firrtl.uint, !firrtl.uint
  connect %z, %4 : !firrtl.uint, !firrtl.uint<17>
  connect %z, %5 : !firrtl.uint, !firrtl.uint<17>
  connect %z, %6 : !firrtl.uint, !firrtl.uint<17>
  connect %z, %7 : !firrtl.uint, !firrtl.uint<17>
}

// CHECK-LABEL: module @regsyncreset
firrtl.module @regsyncreset(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %foo : !firrtl.uint<2>, out %bar: !firrtl.uint<2>) attributes {firrtl.random_init_width = 2 : ui64} {
  // CHECK: %[[const:.*]] = constant 1
  // CHECK-NEXT: regreset %clock, %reset, %[[const]] {firrtl.random_init_end = 1 : ui64, random_init_start = 0 : ui64}
  // CHECK-NEXT:  strictconnect %bar, %d : !firrtl.uint<2>
  // CHECK-NEXT:  strictconnect %d, %foo : !firrtl.uint<2>
  // CHECK-NEXT: }
  %d = reg %clock {firrtl.random_init_end = 1 : ui64, random_init_start = 0 : ui64} : !firrtl.clock, !firrtl.uint<2>
  connect %bar, %d : !firrtl.uint<2>, !firrtl.uint<2>
  %c1_ui2 = constant 1 : !firrtl.uint<2>
  %1 = mux(%reset, %c1_ui2, %foo) : (!firrtl.uint<1>, !firrtl.uint<2>, !firrtl.uint<2>) -> !firrtl.uint<2>
  connect %d, %1 : !firrtl.uint<2>, !firrtl.uint<2>
}

// CHECK-LABEL: module @regsyncreset_no
firrtl.module @regsyncreset_no(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %foo : !firrtl.uint, out %bar: !firrtl.uint) {
  // CHECK: %[[const:.*]] = constant 1
  // CHECK: reg %clock
  // CHECK-NEXT:  connect %bar, %d : !firrtl.uint, !firrtl.uint
  // CHECK-NEXT:  %0 = mux(%reset, %[[const]], %foo) : (!firrtl.uint<1>, !firrtl.uint, !firrtl.uint) -> !firrtl.uint
  // CHECK-NEXT:  connect %d, %0 : !firrtl.uint, !firrtl.uint
  // CHECK-NEXT: }
  %d = reg %clock  : !firrtl.clock, !firrtl.uint
  connect %bar, %d : !firrtl.uint, !firrtl.uint
  %c1_ui2 = constant 1 : !firrtl.uint
  %1 = mux(%reset, %c1_ui2, %foo) : (!firrtl.uint<1>, !firrtl.uint, !firrtl.uint) -> !firrtl.uint
  connect %d, %1 : !firrtl.uint, !firrtl.uint
}

// https://github.com/llvm/circt/issues/1215
// CHECK-LABEL: module @dshifts_to_ishifts
firrtl.module @dshifts_to_ishifts(in %a_in: !firrtl.sint<58>,
                                  out %a_out: !firrtl.sint<58>,
                                  in %b_in: !firrtl.uint<8>,
                                  out %b_out: !firrtl.uint<23>,
                                  in %c_in: !firrtl.sint<58>,
                                  out %c_out: !firrtl.sint<58>) {
  // CHECK: %0 = bits %a_in 57 to 4 : (!firrtl.sint<58>) -> !firrtl.uint<54>
  // CHECK: %1 = asSInt %0 : (!firrtl.uint<54>) -> !firrtl.sint<54>
  // CHECK: %2 = pad %1, 58 : (!firrtl.sint<54>) -> !firrtl.sint<58>
  // CHECK: strictconnect %a_out, %2 : !firrtl.sint<58>
  %c4_ui10 = constant 4 : !firrtl.uint<10>
  %0 = dshr %a_in, %c4_ui10 : (!firrtl.sint<58>, !firrtl.uint<10>) -> !firrtl.sint<58>
  connect %a_out, %0 : !firrtl.sint<58>, !firrtl.sint<58>

  // CHECK: %3 = shl %b_in, 4 : (!firrtl.uint<8>) -> !firrtl.uint<12>
  // CHECK: %4 = pad %3, 23 : (!firrtl.uint<12>) -> !firrtl.uint<23>
  // CHECK: strictconnect %b_out, %4 : !firrtl.uint<23>
  %c4_ui4 = constant 4 : !firrtl.uint<4>
  %1 = dshl %b_in, %c4_ui4 : (!firrtl.uint<8>, !firrtl.uint<4>) -> !firrtl.uint<23>
  connect %b_out, %1 : !firrtl.uint<23>, !firrtl.uint<23>

  // CHECK: %5 = bits %c_in 57 to 57 : (!firrtl.sint<58>) -> !firrtl.uint<1>
  // CHECK: %6 = asSInt %5 : (!firrtl.uint<1>) -> !firrtl.sint<1>
  // CHECK: %7 = pad %6, 58 : (!firrtl.sint<1>) -> !firrtl.sint<58>
  // CHECK: strictconnect %c_out, %7 : !firrtl.sint<58>
  %c438_ui10 = constant 438 : !firrtl.uint<10>
  %2 = dshr %c_in, %c438_ui10 : (!firrtl.sint<58>, !firrtl.uint<10>) -> !firrtl.sint<58>
  connect %c_out, %2 : !firrtl.sint<58>, !firrtl.sint<58>
}

// CHECK-LABEL: module @constReg
firrtl.module @constReg(in %clock: !firrtl.clock,
              in %en: !firrtl.uint<1>, out %out: !firrtl.uint<1>) {
  %r1 = reg %clock  : !firrtl.clock, !firrtl.uint<1>
  %c1_ui1 = constant 1 : !firrtl.uint<1>
  %0 = mux(%en, %c1_ui1, %r1) : (!firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
  connect %r1, %0 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %out, %r1 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK:  %[[C11:.+]] = constant 1 : !firrtl.uint<1>
  // CHECK:  strictconnect %out, %[[C11]]
}

// CHECK-LABEL: module @constReg
firrtl.module @constReg2(in %clock: !firrtl.clock,
              in %en: !firrtl.uint<1>, out %out: !firrtl.uint<1>) {
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
  // CHECK:  %[[C12:.+]] = constant 1 : !firrtl.uint<1>
  // CHECK:  strictconnect %out, %[[C12]]
}

// CHECK-LABEL: module @constReg3
firrtl.module @constReg3(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %cond: !firrtl.uint<1>, out %z: !firrtl.uint<8>) {
  %c11_ui8 = constant 11 : !firrtl.uint<8>
  %r = regreset %clock, %reset, %c11_ui8  : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<8>, !firrtl.uint<8>
  %0 = mux(%cond, %c11_ui8, %r) : (!firrtl.uint<1>, !firrtl.uint<8>, !firrtl.uint<8>) -> !firrtl.uint<8>
  connect %r, %0 : !firrtl.uint<8>, !firrtl.uint<8>
  // CHECK:  %[[C14:.+]] = constant 11
  // CHECK: strictconnect %z, %[[C14]]
  connect %z, %r : !firrtl.uint<8>, !firrtl.uint<8>
}

// CHECK-LABEL: module @constReg4
firrtl.module @constReg4(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %cond: !firrtl.uint<1>, out %z: !firrtl.uint<8>) {
  %c11_ui8 = constant 11 : !firrtl.uint<8>
  %c11_ui4 = constant 11 : !firrtl.uint<8>
  %r = regreset %clock, %reset, %c11_ui4  : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<8>, !firrtl.uint<8>
  %0 = mux(%cond, %c11_ui8, %r) : (!firrtl.uint<1>, !firrtl.uint<8>, !firrtl.uint<8>) -> !firrtl.uint<8>
  connect %r, %0 : !firrtl.uint<8>, !firrtl.uint<8>
  // CHECK:  %[[C13:.+]] = constant 11
  // CHECK: strictconnect %z, %[[C13]]
  connect %z, %r : !firrtl.uint<8>, !firrtl.uint<8>
}

// CHECK-LABEL: module @constReg6
firrtl.module @constReg6(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %cond: !firrtl.uint<1>, out %z: !firrtl.uint<8>) {
  %c11_ui8 = constant 11 : !firrtl.uint<8>
  %c11_ui4 = constant 11 : !firrtl.uint<8>
  %resCond = and %reset, %cond : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
  %r = regreset %clock, %resCond, %c11_ui4  : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<8>, !firrtl.uint<8>
  %0 = mux(%cond, %c11_ui8, %r) : (!firrtl.uint<1>, !firrtl.uint<8>, !firrtl.uint<8>) -> !firrtl.uint<8>
  connect %r, %0 : !firrtl.uint<8>, !firrtl.uint<8>
  // CHECK:  %[[C13:.+]] = constant 11
  // CHECK: strictconnect %z, %[[C13]]
  connect %z, %r : !firrtl.uint<8>, !firrtl.uint<8>
}

// Cannot optimize if bit mismatch with constant reset.
// CHECK-LABEL: module @constReg5
firrtl.module @constReg5(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %cond: !firrtl.uint<1>, out %z: !firrtl.uint<8>) {
  %c11_ui8 = constant 11 : !firrtl.uint<8>
  %c11_ui4 = constant 11 : !firrtl.uint<4>
  %r = regreset %clock, %reset, %c11_ui4  : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<4>, !firrtl.uint<8>
  // CHECK: %0 = mux(%cond, %c11_ui8, %r)
  %0 = mux(%cond, %c11_ui8, %r) : (!firrtl.uint<1>, !firrtl.uint<8>, !firrtl.uint<8>) -> !firrtl.uint<8>
  // CHECK: strictconnect %r, %0
  connect %r, %0 : !firrtl.uint<8>, !firrtl.uint<8>
  connect %z, %r : !firrtl.uint<8>, !firrtl.uint<8>
}

// Should not crash when the reset value is a block argument.
firrtl.module @constReg7(in %v: !firrtl.uint<1>, in %clock: !firrtl.clock, in %reset: !firrtl.reset) {
  %r = regreset %clock, %reset, %v  : !firrtl.clock, !firrtl.reset, !firrtl.uint<1>, !firrtl.uint<4>
}

// Check that regreset reset mux folding doesn't respects
// DontTouchAnnotations or other annotations.
// CHECK-LABEL: module @constReg8
firrtl.module @constReg8(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, out %out1: !firrtl.uint<1>, out %out2: !firrtl.uint<1>) {
  %c1_ui1 = constant 1 : !firrtl.uint<1>
  // CHECK: regreset sym @s2
  %r1 = regreset sym @s2 %clock, %reset, %c1_ui1 : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>
  %0 = mux(%reset, %c1_ui1, %r1) : (!firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
  connect %r1, %0 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %out1, %r1 : !firrtl.uint<1>, !firrtl.uint<1>

  // CHECK: regreset
  // CHECK-SAME: Foo
  %r2 = regreset  %clock, %reset, %c1_ui1 {annotations = [{class = "Foo"}]} : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>
  %1 = mux(%reset, %c1_ui1, %r2) : (!firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
  connect %r2, %1 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %out2, %r2 : !firrtl.uint<1>, !firrtl.uint<1>
}

firrtl.module @BitCast(out %o:!firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<1>> ) {
  %a = wire : !firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<1>>
  %b = bitcast %a : (!firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<1>>) -> (!firrtl.uint<3>)
  %b2 = bitcast %b : (!firrtl.uint<3>) -> (!firrtl.uint<3>)
  %c = bitcast %b2 :  (!firrtl.uint<3>)-> (!firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<1>>)
  connect %o, %c : !firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<1>>, !firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<1>>
  // CHECK: strictconnect %o, %a : !firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<1>>
}

// Check that we can create bundles directly
// CHECK-LABEL: module @MergeBundle
firrtl.module @MergeBundle(out %o:!firrtl.bundle<valid: uint<1>, ready: uint<1>>, in %i:!firrtl.uint<1> ) {
  %a = wire : !firrtl.bundle<valid: uint<1>, ready: uint<1>>
  %a0 = subfield %a[valid] : !firrtl.bundle<valid: uint<1>, ready: uint<1>>
  %a1 = subfield %a[ready] : !firrtl.bundle<valid: uint<1>, ready: uint<1>>
  strictconnect %a0, %i : !firrtl.uint<1>
  strictconnect %a1, %i : !firrtl.uint<1>
  strictconnect %o, %a : !firrtl.bundle<valid: uint<1>, ready: uint<1>>
  // CHECK: %0 = bundlecreate %i, %i
  // CHECK-NEXT: strictconnect %a, %0
}

// Check that we can create vectors directly
// CHECK-LABEL: module @MergeVector
firrtl.module @MergeVector(out %o:!firrtl.vector<uint<1>, 3>, in %i:!firrtl.uint<1> ) {
  %a = wire : !firrtl.vector<uint<1>, 3>
  %a0 = subindex %a[0] : !firrtl.vector<uint<1>, 3>
  %a1 = subindex %a[1] : !firrtl.vector<uint<1>, 3>
  %a2 = subindex %a[2] : !firrtl.vector<uint<1>, 3>
  strictconnect %a0, %i : !firrtl.uint<1>
  strictconnect %a1, %i : !firrtl.uint<1>
  strictconnect %a2, %i : !firrtl.uint<1>
  strictconnect %o, %a : !firrtl.vector<uint<1>, 3>
  // CHECK: %0 = vectorcreate %i, %i, %i
  // CHECK-NEXT: strictconnect %a, %0
}

// Check that we can create vectors directly
// CHECK-LABEL: module @MergeAgg
firrtl.module @MergeAgg(out %o: !firrtl.vector<bundle<valid: uint<1>, ready: uint<1>>, 3> ) {
  %c = constant 0 : !firrtl.uint<1>
  %a = wire : !firrtl.vector<bundle<valid: uint<1>, ready: uint<1>>, 3>
  %a0 = subindex %a[0] : !firrtl.vector<bundle<valid: uint<1>, ready: uint<1>>, 3>
  %a1 = subindex %a[1] : !firrtl.vector<bundle<valid: uint<1>, ready: uint<1>>, 3>
  %a2 = subindex %a[2] : !firrtl.vector<bundle<valid: uint<1>, ready: uint<1>>, 3>
  %a00 = subfield %a0[valid] : !firrtl.bundle<valid: uint<1>, ready: uint<1>>
  %a01 = subfield %a0[ready] : !firrtl.bundle<valid: uint<1>, ready: uint<1>>
  %a10 = subfield %a1[valid] : !firrtl.bundle<valid: uint<1>, ready: uint<1>>
  %a11 = subfield %a1[ready] : !firrtl.bundle<valid: uint<1>, ready: uint<1>>
  %a20 = subfield %a2[valid] : !firrtl.bundle<valid: uint<1>, ready: uint<1>>
  %a21 = subfield %a2[ready] : !firrtl.bundle<valid: uint<1>, ready: uint<1>>
  strictconnect %a00, %c : !firrtl.uint<1>
  strictconnect %a01, %c : !firrtl.uint<1>
  strictconnect %a10, %c : !firrtl.uint<1>
  strictconnect %a11, %c : !firrtl.uint<1>
  strictconnect %a20, %c : !firrtl.uint<1>
  strictconnect %a21, %c : !firrtl.uint<1>
  strictconnect %o, %a :  !firrtl.vector<bundle<valid: uint<1>, ready: uint<1>>, 3>
// CHECK: [0 : ui1, 0 : ui1], [0 : ui1, 0 : ui1], [0 : ui1, 0 : ui1]] : !firrtl.vector<bundle<valid: uint<1>, ready: uint<1>>, 3>
// CHECK-NEXT: %a = wire   : !firrtl.vector<bundle<valid: uint<1>, ready: uint<1>>, 3>
// CHECK-NEXT: strictconnect %o, %a : !firrtl.vector<bundle<valid: uint<1>, ready: uint<1>>, 3>
// CHECK-NEXT: strictconnect %a, %0 : !firrtl.vector<bundle<valid: uint<1>, ready: uint<1>>, 3>
}

// TODO: Move to an apporpriate place
// Issue #2197
// CHECK-LABEL: @Issue2197
firrtl.module @Issue2197(in %clock: !firrtl.clock, out %x: !firrtl.uint<2>) {
//  // _HECK: [[ZERO:%.+]] = constant 0 : !firrtl.uint<2>
//  // _HECK-NEXT: strictconnect %x, [[ZERO]] : !firrtl.uint<2>
//  %invalid_ui1 = invalidvalue : !firrtl.uint<1>
//  %_reg = reg droppable_name %clock : !firrtl.clock, !firrtl.uint<2>
//  %0 = pad %invalid_ui1, 2 : (!firrtl.uint<1>) -> !firrtl.uint<2>
//  connect %_reg, %0 : !firrtl.uint<2>, !firrtl.uint<2>
//  connect %x, %_reg : !firrtl.uint<2>, !firrtl.uint<2>
}

// This is checking the behavior of sign extension of zero-width constants that
// results from trying to primops.
// CHECK-LABEL: @ZeroWidthAdd
firrtl.module @ZeroWidthAdd(out %a: !firrtl.sint<1>) {
  %zw = constant 0 : !firrtl.sint<0>
  %0 = constant 0 : !firrtl.sint<0>
  %1 = add %0, %zw : (!firrtl.sint<0>, !firrtl.sint<0>) -> !firrtl.sint<1>
  connect %a, %1 : !firrtl.sint<1>, !firrtl.sint<1>
  // CHECK:      %[[zero:.+]] = constant 0 : !firrtl.sint<1>
  // CHECK-NEXT: strictconnect %a, %[[zero]]
}

// CHECK-LABEL: @ZeroWidthDshr
firrtl.module @ZeroWidthDshr(in %a: !firrtl.sint<0>, out %b: !firrtl.sint<0>) {
  %zw = constant 0 : !firrtl.uint<0>
  %0 = dshr %a, %zw : (!firrtl.sint<0>, !firrtl.uint<0>) -> !firrtl.sint<0>
  connect %b, %0 : !firrtl.sint<0>, !firrtl.sint<0>
  // CHECK:      %[[zero:.+]] = constant 0 : !firrtl.sint<0>
  // CHECK-NEXT: strictconnect %b, %[[zero]]
}

// CHECK-LABEL: @ZeroWidthPad
firrtl.module @ZeroWidthPad(out %b: !firrtl.sint<1>) {
  %zw = constant 0 : !firrtl.sint<0>
  %0 = pad %zw, 1 : (!firrtl.sint<0>) -> !firrtl.sint<1>
  connect %b, %0 : !firrtl.sint<1>, !firrtl.sint<1>
  // CHECK:      %[[zero:.+]] = constant 0 : !firrtl.sint<1>
  // CHECK-NEXT: strictconnect %b, %[[zero]]
}

// CHECK-LABEL: @ZeroWidthCat
firrtl.module @ZeroWidthCat(out %a: !firrtl.uint<1>) {
  %one = constant 1 : !firrtl.uint<1>
  %zw = constant 0 : !firrtl.uint<0>
  %0 = cat %one, %zw : (!firrtl.uint<1>, !firrtl.uint<0>) -> !firrtl.uint<1>
  connect %a, %0 : !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK:      %[[one:.+]] = constant 1 : !firrtl.uint<1>
  // CHECK-NEXT: strictconnect %a, %[[one]]
}

//TODO: Move to an appropriate place
// Issue mentioned in PR #2251
// CHECK-LABEL: @Issue2251
firrtl.module @Issue2251(out %o: !firrtl.sint<15>) {
//  // pad used to always return an unsigned constant
//  %invalid_si1 = invalidvalue : !firrtl.sint<1>
//  %0 = pad %invalid_si1, 15 : (!firrtl.sint<1>) -> !firrtl.sint<15>
//  connect %o, %0 : !firrtl.sint<15>, !firrtl.sint<15>
//  // _HECK:      %[[zero:.+]] = constant 0 : !firrtl.sint<15>
//  // _HECK-NEXT: strictconnect %o, %[[zero]]
}

// Issue mentioned in #2289
// CHECK-LABEL: @Issue2289
firrtl.module @Issue2289(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, out %out: !firrtl.uint<5>) {
  %r = reg %clock  : !firrtl.clock, !firrtl.uint<1>
  connect %r, %r : !firrtl.uint<1>, !firrtl.uint<1>
  %c0_ui4 = constant 0 : !firrtl.uint<4>
  %c1_ui1 = constant 1 : !firrtl.uint<1>
  %0 = dshl %c1_ui1, %r : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<2>
  %1 = sub %c0_ui4, %0 : (!firrtl.uint<4>, !firrtl.uint<2>) -> !firrtl.uint<5>
  connect %out, %1 : !firrtl.uint<5>, !firrtl.uint<5>
  // CHECK:      %[[dshl:.+]] = dshl
  // CHECK-NEXT: %[[neg:.+]] = neg %[[dshl]]
  // CHECK-NEXT: %[[pad:.+]] = pad %[[neg]], 5
  // CHECK-NEXT: %[[cast:.+]] = asUInt %[[pad]]
  // CHECK-NEXT: strictconnect %out, %[[cast]]
}

// Issue mentioned in #2291
// CHECK-LABEL: @Issue2291
firrtl.module @Issue2291(out %out: !firrtl.uint<1>) {
  %c1_ui1 = constant 1 : !firrtl.uint<1>
  %clock = asClock %c1_ui1 : (!firrtl.uint<1>) -> !firrtl.clock
  %0 = asUInt %clock : (!firrtl.clock) -> !firrtl.uint<1>
  connect %out, %0 : !firrtl.uint<1>, !firrtl.uint<1>
}

// Check that canonicalizing connects to zero works for clock, reset, and async
// reset.  All these types require special constants as opposed to constants.
//
// CHECK-LABEL: @Issue2314
firrtl.module @Issue2314(out %clock: !firrtl.clock, out %reset: !firrtl.reset, out %asyncReset: !firrtl.asyncreset) {
  // CHECK-DAG: %[[zero_clock:.+]] = specialconstant 0 : !firrtl.clock
  // CHECK-DAG: %[[zero_reset:.+]] = specialconstant 0 : !firrtl.reset
  // CHECK-DAG: %[[zero_asyncReset:.+]] = specialconstant 0 : !firrtl.asyncreset
  %inv_clock = wire  : !firrtl.clock
  %invalid_clock = invalidvalue : !firrtl.clock
  connect %inv_clock, %invalid_clock : !firrtl.clock, !firrtl.clock
  connect %clock, %inv_clock : !firrtl.clock, !firrtl.clock
  // CHECK: strictconnect %clock, %[[zero_clock]]
  %inv_reset = wire  : !firrtl.reset
  %invalid_reset = invalidvalue : !firrtl.reset
  connect %inv_reset, %invalid_reset : !firrtl.reset, !firrtl.reset
  connect %reset, %inv_reset : !firrtl.reset, !firrtl.reset
  // CHECK: strictconnect %reset, %[[zero_reset]]
  %inv_asyncReset = wire  : !firrtl.asyncreset
  %invalid_asyncreset = invalidvalue : !firrtl.asyncreset
  connect %inv_asyncReset, %invalid_asyncreset : !firrtl.asyncreset, !firrtl.asyncreset
  connect %asyncReset, %inv_asyncReset : !firrtl.asyncreset, !firrtl.asyncreset
  // CHECK: strictconnect %asyncReset, %[[zero_asyncReset]]
}

// Crasher from issue #3043
// CHECK-LABEL: @Issue3043
firrtl.module @Issue3043(out %a: !firrtl.vector<uint<5>, 3>) {
  %_b = wire  : !firrtl.vector<uint<5>, 3>
  %b = node sym @b %_b  : !firrtl.vector<uint<5>, 3>
  %invalid = invalidvalue : !firrtl.vector<uint<5>, 3>
  strictconnect %_b, %invalid : !firrtl.vector<uint<5>, 3>
  connect %a, %_b : !firrtl.vector<uint<5>, 3>, !firrtl.vector<uint<5>, 3>
}

// Test behaviors folding with zero-width constants, issue #2514.
// CHECK-LABEL: @Issue2514
firrtl.module @Issue2514(
  in %s: !firrtl.sint<0>,
  in %u: !firrtl.uint<0>,
  out %geq_0: !firrtl.uint<1>,
  out %geq_1: !firrtl.uint<1>,
  out %geq_2: !firrtl.uint<1>,
  out %geq_3: !firrtl.uint<1>,
  out %gt_0:  !firrtl.uint<1>,
  out %gt_1:  !firrtl.uint<1>,
  out %gt_2:  !firrtl.uint<1>,
  out %gt_3:  !firrtl.uint<1>,
  out %lt_0:  !firrtl.uint<1>,
  out %lt_1:  !firrtl.uint<1>,
  out %lt_2:  !firrtl.uint<1>,
  out %lt_3:  !firrtl.uint<1>,
  out %leq_0: !firrtl.uint<1>,
  out %leq_1: !firrtl.uint<1>,
  out %leq_2: !firrtl.uint<1>,
  out %leq_3: !firrtl.uint<1>
) {
  %t = constant 0: !firrtl.sint<0>
  %v = constant 0: !firrtl.uint<0>

  // CHECK-DAG: %[[zero_i1:.+]] = constant 0 : !firrtl.uint<1>
  // CHECK-DAG: %[[one_i1:.+]] = constant 1 : !firrtl.uint<1>

  // geq(x, y) -> 1 when x and y are both zero-width (and here, one is a constant)
  %3 = geq %s, %t : (!firrtl.sint<0>, !firrtl.sint<0>) -> !firrtl.uint<1>
  %4 = geq %t, %s : (!firrtl.sint<0>, !firrtl.sint<0>) -> !firrtl.uint<1>
  %5 = geq %u, %v : (!firrtl.uint<0>, !firrtl.uint<0>) -> !firrtl.uint<1>
  %6 = geq %v, %u : (!firrtl.uint<0>, !firrtl.uint<0>) -> !firrtl.uint<1>
  strictconnect %geq_0, %3 : !firrtl.uint<1>
  strictconnect %geq_1, %4 : !firrtl.uint<1>
  strictconnect %geq_2, %5 : !firrtl.uint<1>
  strictconnect %geq_3, %6 : !firrtl.uint<1>
  // CHECK: strictconnect %geq_0, %[[one_i1]]
  // CHECK: strictconnect %geq_1, %[[one_i1]]
  // CHECK: strictconnect %geq_2, %[[one_i1]]
  // CHECK: strictconnect %geq_3, %[[one_i1]]

  // gt(x, y) -> 0 when x and y are both zero-width (and here, one is a constant)
  %7 = gt %s, %t : (!firrtl.sint<0>, !firrtl.sint<0>) -> !firrtl.uint<1>
  %8 = gt %t, %s : (!firrtl.sint<0>, !firrtl.sint<0>) -> !firrtl.uint<1>
  %9 = gt %u, %v : (!firrtl.uint<0>, !firrtl.uint<0>) -> !firrtl.uint<1>
  %10 = gt %v, %u : (!firrtl.uint<0>, !firrtl.uint<0>) -> !firrtl.uint<1>
  strictconnect %gt_0, %7 : !firrtl.uint<1>
  strictconnect %gt_1, %8 : !firrtl.uint<1>
  strictconnect %gt_2, %9 : !firrtl.uint<1>
  strictconnect %gt_3, %10 : !firrtl.uint<1>
  // CHECK: strictconnect %gt_0, %[[zero_i1]]
  // CHECK: strictconnect %gt_1, %[[zero_i1]]
  // CHECK: strictconnect %gt_2, %[[zero_i1]]
  // CHECK: strictconnect %gt_3, %[[zero_i1]]

  // lt(x, y) -> 0 when x and y are both zero-width (and here, one is a constant)
  %11 = lt %s, %t : (!firrtl.sint<0>, !firrtl.sint<0>) -> !firrtl.uint<1>
  %12 = lt %t, %s : (!firrtl.sint<0>, !firrtl.sint<0>) -> !firrtl.uint<1>
  %13 = lt %u, %v : (!firrtl.uint<0>, !firrtl.uint<0>) -> !firrtl.uint<1>
  %14 = lt %v, %u : (!firrtl.uint<0>, !firrtl.uint<0>) -> !firrtl.uint<1>
  strictconnect %lt_0, %11 : !firrtl.uint<1>
  strictconnect %lt_1, %12 : !firrtl.uint<1>
  strictconnect %lt_2, %13 : !firrtl.uint<1>
  strictconnect %lt_3, %14 : !firrtl.uint<1>
  // CHECK: strictconnect %lt_0, %[[zero_i1]]
  // CHECK: strictconnect %lt_1, %[[zero_i1]]
  // CHECK: strictconnect %lt_2, %[[zero_i1]]
  // CHECK: strictconnect %lt_3, %[[zero_i1]]

  // leq(x, y) -> 1 when x and y are both zero-width (and here, one is a constant)
  %15 = leq %s, %t : (!firrtl.sint<0>, !firrtl.sint<0>) -> !firrtl.uint<1>
  %16 = leq %t, %s : (!firrtl.sint<0>, !firrtl.sint<0>) -> !firrtl.uint<1>
  %17 = leq %u, %v : (!firrtl.uint<0>, !firrtl.uint<0>) -> !firrtl.uint<1>
  %18 = leq %v, %u : (!firrtl.uint<0>, !firrtl.uint<0>) -> !firrtl.uint<1>
  strictconnect %leq_0, %15 : !firrtl.uint<1>
  strictconnect %leq_1, %16 : !firrtl.uint<1>
  strictconnect %leq_2, %17 : !firrtl.uint<1>
  strictconnect %leq_3, %18 : !firrtl.uint<1>
  // CHECK: strictconnect %leq_0, %[[one_i1]]
  // CHECK: strictconnect %leq_1, %[[one_i1]]
  // CHECK: strictconnect %leq_2, %[[one_i1]]
  // CHECK: strictconnect %leq_3, %[[one_i1]]
}

// CHECK-LABEL: @NamePropagation
firrtl.module @NamePropagation(in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>, in %c: !firrtl.uint<4>, out %res1: !firrtl.uint<2>, out %res2: !firrtl.uint<2>) {
  // CHECK-NEXT: %e = bits %c 1 to 0 {name = "e"}
  %1 = bits %c 2 to 0 : (!firrtl.uint<4>) -> !firrtl.uint<3>
  %e = bits %1 1 to 0 {name = "e"}: (!firrtl.uint<3>) -> !firrtl.uint<2>
  // CHECK-NEXT: strictconnect %res1, %e
  strictconnect %res1, %e : !firrtl.uint<2>

  // CHECK-NEXT: %name_node = not %e {name = "name_node"} : (!firrtl.uint<2>) -> !firrtl.uint<2>
  // CHECK-NEXT: strictconnect %res2, %name_node
  %2 = not %e : (!firrtl.uint<2>) -> !firrtl.uint<2>
  %name_node = node droppable_name %2 : !firrtl.uint<2>
  strictconnect %res2, %name_node : !firrtl.uint<2>
}

// Issue 3319: https://github.com/llvm/circt/issues/3319
// CHECK-LABEL: @Foo3319
firrtl.module @Foo3319(in %i: !firrtl.uint<1>, out %o : !firrtl.uint<1>) {
  %c0_ui1 = constant 0 : !firrtl.uint<1>
  %c1_ui1 = constant 1 : !firrtl.uint<1>
  %0 = and %c0_ui1, %i : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
  // CHECK: %n = node interesting_name %c0_ui1
  %n = node interesting_name %0  : !firrtl.uint<1>
  // CHECK: strictconnect %o, %n
  strictconnect %o, %n : !firrtl.uint<1>
}

// CHECK-LABEL: @WireByPass
firrtl.module @WireByPass(in %i: !firrtl.uint<1>, out %o : !firrtl.uint<1>) {
  %c0_ui1 = constant 0 : !firrtl.uint<1>
  %n = wire interesting_name : !firrtl.uint<1>
  // CHECK: strictconnect %n, %c0_ui1
  strictconnect %n, %c0_ui1 : !firrtl.uint<1>
  // CHECK: strictconnect %o, %n
  strictconnect %o, %n : !firrtl.uint<1>
}

// Check that canonicalizeSingleSetConnect doesn't remove a wire with an
// Annotation on it.
//
// CHECK-LABEL: @AnnotationsBlockRemoval
firrtl.module @AnnotationsBlockRemoval(
  in %a: !firrtl.uint<1>,
  out %b: !firrtl.uint<1>
) {
  // CHECK: %w = wire
  %w = wire droppable_name {annotations = [{class = "Foo"}]} : !firrtl.uint<1>
  strictconnect %w, %a : !firrtl.uint<1>
  strictconnect %b, %w : !firrtl.uint<1>
}

// CHECK-LABEL: module @Verification
firrtl.module @Verification(in %clock: !firrtl.clock, in %p: !firrtl.uint<1>, out %o : !firrtl.uint<1>) {
  %c0 = constant 0 : !firrtl.uint<1>
  %c1 = constant 1 : !firrtl.uint<1>

  // Never enabled.
  // CHECK-NOT: assert
  assert %clock, %p, %c0, "assert0" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK-NOT: assume
  assume %clock, %p, %c0, "assume0" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK-NOT: cover
  cover %clock, %p, %c0, "cover0" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>

  // Never fired.
  // CHECK-NOT: assert
  assert %clock, %c1, %p, "assert1" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK-NOT: assume
  assume %clock, %c1, %p, "assume1" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK-NOT: cover
  cover %clock, %c0, %p, "cover0" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>
  // CHECK-NOT: int.isX
  %x = int.isX %c0 : !firrtl.uint<1>
  strictconnect %o, %x : !firrtl.uint<1>
}

// COMMON-LABEL:  module @MultibitMux
// COMMON-NEXT:      %0 = subaccess %a[%sel] : !firrtl.vector<uint<1>, 3>, !firrtl.uint<2>
// COMMON-NEXT:      strictconnect %b, %0 : !firrtl.uint<1>
firrtl.module @MultibitMux(in %a: !firrtl.vector<uint<1>, 3>, in %sel: !firrtl.uint<2>, out %b: !firrtl.uint<1>) {
  %0 = subindex %a[2] : !firrtl.vector<uint<1>, 3>
  %1 = subindex %a[1] : !firrtl.vector<uint<1>, 3>
  %2 = subindex %a[0] : !firrtl.vector<uint<1>, 3>
  %3 = multibit_mux %sel, %0, %1, %2 : !firrtl.uint<2>, !firrtl.uint<1>
  strictconnect %b, %3 : !firrtl.uint<1>
}

// CHECK-LABEL: module @NameProp
firrtl.module @NameProp(in %in0: !firrtl.uint<1>, in %in1: !firrtl.uint<1>, out %out: !firrtl.uint<1>) {
  %0 = or %in0, %in1 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
  %_useless_name_1 = node  %0  : !firrtl.uint<1>
  %useful_name = node %_useless_name_1  : !firrtl.uint<1>
  %_useless_name_2 = node  %useful_name  : !firrtl.uint<1>
  // CHECK-NEXT: %useful_name = or %in0, %in1
  // CHECK-NEXT: strictconnect %out, %useful_name
  strictconnect %out, %_useless_name_2 : !firrtl.uint<1>
}

// CHECK-LABEL: module @CrashAllUnusedPorts
firrtl.module @CrashAllUnusedPorts() {
  %c0_ui1 = constant 0 : !firrtl.uint<1>
  %foo, %bar = mem  Undefined  {depth = 3 : i64, groupID = 4 : ui32, name = "whatever", portNames = ["MPORT_1", "MPORT_5"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<2>, en: uint<1>, clk: clock, data: uint<2>, mask: uint<1>>, !firrtl.bundle<addr: uint<2>, en: uint<1>, clk: clock, data flip: uint<2>>
  %26 = subfield %foo[en] : !firrtl.bundle<addr: uint<2>, en: uint<1>, clk: clock, data: uint<2>, mask: uint<1>>
  strictconnect %26, %c0_ui1 : !firrtl.uint<1>
}

// CHECK-LABEL: module @CrashRegResetWithOneReset
firrtl.module @CrashRegResetWithOneReset(in %clock: !firrtl.clock, in %reset: !firrtl.asyncreset, in %io_d: !firrtl.uint<1>, out %io_q: !firrtl.uint<1>, in %io_en: !firrtl.uint<1>) {
  %c1_asyncreset = specialconstant 1 : !firrtl.asyncreset
  %c0_ui1 = constant 0 : !firrtl.uint<1>
  %reg = regreset  %clock, %c1_asyncreset, %c0_ui1  : !firrtl.clock, !firrtl.asyncreset, !firrtl.uint<1>, !firrtl.uint<1>
  %0 = mux(%io_en, %io_d, %reg) : (!firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
  connect %reg, %0 : !firrtl.uint<1>, !firrtl.uint<1>
  connect %io_q, %reg : !firrtl.uint<1>, !firrtl.uint<1>
}

// A read-only memory with memory initialization should not be removed.
// CHECK-LABEL: module @ReadOnlyFileInitialized
firrtl.module @ReadOnlyFileInitialized(
  in %clock: !firrtl.clock,
  in %reset: !firrtl.uint<1>,
  in %read_en: !firrtl.uint<1>,
  out %read_data: !firrtl.uint<8>,
  in %read_addr: !firrtl.uint<5>
) {
  // CHECK-NEXT: mem
  // CHECK-SAME:   name = "withInit"
  %m_r = mem Undefined {
    depth = 32 : i64,
    groupID = 1 : ui32,
    init = #firrtl.meminit<"mem1.hex.txt", false, true>,
    name = "withInit",
    portNames = ["m_r"],
    readLatency = 1 : i32,
    writeLatency = 1 : i32
  } : !firrtl.bundle<addr: uint<5>, en: uint<1>, clk: clock, data flip: uint<8>>
  %0 = subfield %m_r[addr] :
    !firrtl.bundle<addr: uint<5>, en: uint<1>, clk: clock, data flip: uint<8>>
  %1 = subfield %m_r[en] :
    !firrtl.bundle<addr: uint<5>, en: uint<1>, clk: clock, data flip: uint<8>>
  %2 = subfield %m_r[clk] :
    !firrtl.bundle<addr: uint<5>, en: uint<1>, clk: clock, data flip: uint<8>>
  %3 = subfield %m_r[data] :
    !firrtl.bundle<addr: uint<5>, en: uint<1>, clk: clock, data flip: uint<8>>
  strictconnect %0, %read_addr : !firrtl.uint<5>
  strictconnect %1, %read_en : !firrtl.uint<1>
  strictconnect %2, %clock : !firrtl.clock
  strictconnect %read_data, %3 : !firrtl.uint<8>
}

// CHECK-LABEL: @MuxCondWidth
firrtl.module @MuxCondWidth(in %cond: !firrtl.uint<1>, out %foo: !firrtl.uint<3>) {
  // Don't canonicalize if the type is not UInt<1>
  // CHECK: %0 = mux(%cond, %c0_ui3, %c1_ui3) : (!firrtl.uint<1>, !firrtl.uint<3>, !firrtl.uint<3>) -> !firrtl.uint<3>
  // CHECK-NEXT:  strictconnect %foo, %0
  %c0_ui1 = constant 0 : !firrtl.uint<1>
  %c1_ui3 = constant 1 : !firrtl.uint<3>
  %0 = mux(%cond, %c0_ui1, %c1_ui3) : (!firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<3>) -> !firrtl.uint<3>
  strictconnect %foo, %0 : !firrtl.uint<3>
}

// CHECK-LABEL: module @RemoveUnusedInvalid
firrtl.module @RemoveUnusedInvalid() {
  // CHECK-NOT: invalidvalue
  %0 = invalidvalue : !firrtl.uint<1>
}
// CHECK-NEXT: }

// CHECK-LABEL: module @AggregateCreate(
firrtl.module @AggregateCreate(in %vector_in: !firrtl.vector<uint<1>, 2>,
                               in %bundle_in: !firrtl.bundle<a: uint<1>, b: uint<1>>,
                               out %vector_out: !firrtl.vector<uint<1>, 2>,
                               out %bundle_out: !firrtl.bundle<a: uint<1>, b: uint<1>>) {
  %0 = subindex %vector_in[0] : !firrtl.vector<uint<1>, 2>
  %1 = subindex %vector_in[1] : !firrtl.vector<uint<1>, 2>
  %vector = vectorcreate %0, %1 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.vector<uint<1>, 2>
  strictconnect %vector_out, %vector : !firrtl.vector<uint<1>, 2>

  %2 = subfield %bundle_in["a"] : !firrtl.bundle<a: uint<1>, b: uint<1>>
  %3 = subfield %bundle_in["b"] : !firrtl.bundle<a: uint<1>, b: uint<1>>
  %bundle = bundlecreate %2, %3 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.bundle<a: uint<1>, b: uint<1>>
  strictconnect %bundle_out, %bundle : !firrtl.bundle<a: uint<1>, b: uint<1>>
  // CHECK-NEXT: strictconnect %vector_out, %vector_in : !firrtl.vector<uint<1>, 2>
  // CHECK-NEXT: strictconnect %bundle_out, %bundle_in : !firrtl.bundle<a: uint<1>, b: uint<1>>
}

// CHECK-LABEL: module @AggregateCreateSingle(
firrtl.module @AggregateCreateSingle(in %vector_in: !firrtl.vector<uint<1>, 1>,
                               in %bundle_in: !firrtl.bundle<a: uint<1>>,
                               out %vector_out: !firrtl.vector<uint<1>, 1>,
                               out %bundle_out: !firrtl.bundle<a: uint<1>>) {

  %0 = subindex %vector_in[0] : !firrtl.vector<uint<1>, 1>
  %vector = vectorcreate %0 : (!firrtl.uint<1>) -> !firrtl.vector<uint<1>, 1>
  strictconnect %vector_out, %vector : !firrtl.vector<uint<1>, 1>

  %2 = subfield %bundle_in["a"] : !firrtl.bundle<a: uint<1>>
  %bundle = bundlecreate %2 : (!firrtl.uint<1>) -> !firrtl.bundle<a: uint<1>>
  strictconnect %bundle_out, %bundle : !firrtl.bundle<a: uint<1>>
  // CHECK-NEXT: strictconnect %vector_out, %vector_in : !firrtl.vector<uint<1>, 1>
  // CHECK-NEXT: strictconnect %bundle_out, %bundle_in : !firrtl.bundle<a: uint<1>>
}

// CHECK-LABEL: module @AggregateCreateEmpty(
firrtl.module @AggregateCreateEmpty(
                               out %vector_out: !firrtl.vector<uint<1>, 0>,
                               out %bundle_out: !firrtl.bundle<>) {

  %vector = vectorcreate : () -> !firrtl.vector<uint<1>, 0>
  strictconnect %vector_out, %vector : !firrtl.vector<uint<1>, 0>

  %bundle = bundlecreate : () -> !firrtl.bundle<>
  strictconnect %bundle_out, %bundle : !firrtl.bundle<>
  // CHECK-DAG: %[[VEC:.+]] = aggregateconstant [] : !firrtl.vector<uint<1>, 0>
  // CHECK-DAG: %[[BUNDLE:.+]] = aggregateconstant [] : !firrtl.bundle<>
  // CHECK-DAG: strictconnect %vector_out, %[[VEC]] : !firrtl.vector<uint<1>, 0>
  // CHECK-DAG: strictconnect %bundle_out, %[[BUNDLE]] : !firrtl.bundle<>
}

// CHECK-LABEL: module @AggregateCreateConst(
firrtl.module @AggregateCreateConst(
                               out %vector_out: !firrtl.vector<uint<1>, 2>,
                               out %bundle_out: !firrtl.bundle<a: uint<1>, b: uint<1>>) {

  %const = constant 0 : !firrtl.uint<1>
  %vector = vectorcreate %const, %const : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.vector<uint<1>, 2>
  strictconnect %vector_out, %vector : !firrtl.vector<uint<1>, 2>

  %bundle = bundlecreate %const, %const : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.bundle<a: uint<1>, b: uint<1>>
  strictconnect %bundle_out, %bundle : !firrtl.bundle<a: uint<1>, b: uint<1>>
  // CHECK-DAG: %[[VEC:.+]] = aggregateconstant [0 : ui1, 0 : ui1] : !firrtl.vector<uint<1>, 2>
  // CHECK-DAG: %[[BUNDLE:.+]] = aggregateconstant [0 : ui1, 0 : ui1] : !firrtl.bundle<a: uint<1>, b: uint<1>>
  // CHECK-DAG: strictconnect %vector_out, %[[VEC]] : !firrtl.vector<uint<1>, 2>
  // CHECK-DAG: strictconnect %bundle_out, %[[BUNDLE]] : !firrtl.bundle<a: uint<1>, b: uint<1>>
}


// CHECK-LABEL: module private @RWProbeUnused
firrtl.module private @RWProbeUnused(in %in: !firrtl.uint<4>, in %clk: !firrtl.clock, out %out: !firrtl.uint) {
  // CHECK-NOT: forceable
  %n, %n_ref = node interesting_name %in forceable : !firrtl.uint<4>
  %w, %w_ref = wire interesting_name forceable : !firrtl.uint, !firrtl.rwprobe<uint>
  connect %w, %n : !firrtl.uint, !firrtl.uint<4>
  %r, %r_ref = reg interesting_name %clk forceable : !firrtl.clock, !firrtl.uint, !firrtl.rwprobe<uint>
  connect %r, %w : !firrtl.uint, !firrtl.uint
  connect %out, %r : !firrtl.uint, !firrtl.uint
}


// CHECK-LABEL: module @ClockGateIntrinsic
firrtl.module @ClockGateIntrinsic(in %clock: !firrtl.clock, in %enable: !firrtl.uint<1>, in %testEnable: !firrtl.uint<1>) {
  // CHECK-NEXT: specialconstant 0
  %c0_clock = specialconstant 0 : !firrtl.clock
  %c0_ui1 = constant 0 : !firrtl.uint<1>
  %c1_ui1 = constant 1 : !firrtl.uint<1>

  // CHECK-NEXT: %zeroClock = node interesting_name %c0_clock
  %0 = int.clock_gate %c0_clock, %enable
  %zeroClock = node interesting_name %0 : !firrtl.clock

  // CHECK-NEXT: %alwaysOff1 = node interesting_name %c0_clock
  // CHECK-NEXT: %alwaysOff2 = node interesting_name %c0_clock
  %1 = int.clock_gate %clock, %c0_ui1
  %2 = int.clock_gate %clock, %c0_ui1, %c0_ui1
  %alwaysOff1 = node interesting_name %1 : !firrtl.clock
  %alwaysOff2 = node interesting_name %2 : !firrtl.clock

  // CHECK-NEXT: %alwaysOn1 = node interesting_name %clock
  // CHECK-NEXT: %alwaysOn2 = node interesting_name %clock
  // CHECK-NEXT: %alwaysOn3 = node interesting_name %clock
  %3 = int.clock_gate %clock, %c1_ui1
  %4 = int.clock_gate %clock, %c1_ui1, %testEnable
  %5 = int.clock_gate %clock, %enable, %c1_ui1
  %alwaysOn1 = node interesting_name %3 : !firrtl.clock
  %alwaysOn2 = node interesting_name %4 : !firrtl.clock
  %alwaysOn3 = node interesting_name %5 : !firrtl.clock

  // CHECK-NEXT: [[TMP:%.+]] = int.clock_gate %clock, %enable
  // CHECK-NEXT: %dropTestEnable = node interesting_name [[TMP]]
  %6 = int.clock_gate %clock, %enable, %c0_ui1
  %dropTestEnable = node interesting_name %6 : !firrtl.clock
}

// CHECK-LABEL: module @RefTypes
firrtl.module @RefTypes(
    out %x: !firrtl.bundle<a flip: uint<1>>,
    out %y: !firrtl.bundle<a: uint<1>>) {

  %a = wire : !firrtl.uint<1>
  %b = wire : !firrtl.uint<1>
  %a_ref = ref.send  %a : !firrtl.uint<1>
  %a_read_ref = ref.resolve %a_ref : !firrtl.probe<uint<1>>
  // CHECK: strictconnect %b, %a
  strictconnect %b, %a_read_ref : !firrtl.uint<1>

  // Don't collapse if types don't match.
  // CHECK: ref.resolve
  %x_ref = ref.send %x : !firrtl.bundle<a flip: uint<1>>
  %x_read = ref.resolve %x_ref : !firrtl.probe<bundle<a: uint<1>>>
  strictconnect %y, %x_read : !firrtl.bundle<a: uint<1>>

  // CHECK-NOT: forceable
  // CHECK: strictconnect %f_wire, %b
  // CHECK-NOT: forceable
  %f, %f_rw = node %b forceable : !firrtl.uint<1>
  %f_read = ref.resolve %f_rw : !firrtl.rwprobe<uint<1>>
  %f_wire = wire : !firrtl.uint<1>
  strictconnect %f_wire, %f_read : !firrtl.uint<1>

  // CHECK: wire forceable
  // CHECK: ref.resolve
  %flipbundle, %flipbundle_rw = wire forceable : !firrtl.bundle<a flip: uint<1>>, !firrtl.rwprobe<bundle<a: uint<1>>>
  %flipbundle_read = ref.resolve %flipbundle_rw : !firrtl.rwprobe<bundle<a: uint<1>>>
  %flipbundle_wire = wire : !firrtl.bundle<a : uint<1>>
  strictconnect %flipbundle_wire, %flipbundle_read : !firrtl.bundle<a: uint<1>>
}

// Do not rename InstanceOp: https://github.com/llvm/circt/issues/5351
firrtl.extmodule @System(out foo: !firrtl.uint<1>)
firrtl.module @DonotUpdateInstanceName(in %in: !firrtl.uint<1>, out %a: !firrtl.uint<1>) attributes {convention = #firrtl<convention scalarized>} {
  %system_foo = instance system @System(out foo: !firrtl.uint<1>)
  // CHECK: instance system
  %b = node %system_foo : !firrtl.uint<1>
  strictconnect %a, %b : !firrtl.uint<1>
}

// CHECK-LABEL: @RefCastSame
firrtl.module @RefCastSame(in %in: !firrtl.probe<uint<1>>, out %out: !firrtl.probe<uint<1>>) {
  // Drop no-op ref.cast's.
  // CHECK-NEXT:  ref.define %out, %in
  // CHECK-NEXT:  }
  %same_as_in = ref.cast %in : (!firrtl.probe<uint<1>>) -> !firrtl.probe<uint<1>>
  ref.define %out, %same_as_in : !firrtl.probe<uint<1>>
}

}

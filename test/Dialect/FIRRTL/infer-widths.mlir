// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl-infer-widths))' --verify-diagnostics %s | FileCheck %s

firrtl.circuit "Foo" {
  // CHECK-LABEL: @InferConstant
  // CHECK-SAME: out %out0: !firrtl.uint<42>
  // CHECK-SAME: out %out1: !firrtl.sint<42>
  module @InferConstant(out %out0: !firrtl.uint, out %out1: !firrtl.sint) {
    %0 = constant 1 : !firrtl.uint<42>
    %1 = constant 2 : !firrtl.sint<42>
    // CHECK: {{.+}} = constant 0 : !firrtl.uint<1>
    // CHECK: {{.+}} = constant 0 : !firrtl.sint<1>
    // CHECK: {{.+}} = constant 200 : !firrtl.uint<8>
    // CHECK: {{.+}} = constant 200 : !firrtl.sint<9>
    // CHECK: {{.+}} = constant -200 : !firrtl.sint<9>
    %2 = constant 0 : !firrtl.uint
    %3 = constant 0 : !firrtl.sint
    %4 = constant 200 : !firrtl.uint
    %5 = constant 200 : !firrtl.sint
    %6 = constant -200 : !firrtl.sint
    connect %out0, %0 : !firrtl.uint, !firrtl.uint<42>
    connect %out1, %1 : !firrtl.sint, !firrtl.sint<42>
  }

  // CHECK-LABEL: @InferSpecialConstant
  module @InferSpecialConstant() {
    // CHECK: %c0_clock = specialconstant 0 : !firrtl.clock
    %c0_clock = specialconstant 0 : !firrtl.clock
  }

  // CHECK-LABEL: @InferInvalidValue
  module @InferInvalidValue(out %out: !firrtl.uint) {
    // CHECK: %invalid_ui6 = invalidvalue : !firrtl.uint<6>
    %invalid_ui = invalidvalue : !firrtl.uint
    %c42_ui = constant 42 : !firrtl.uint
    connect %out, %invalid_ui : !firrtl.uint, !firrtl.uint
    connect %out, %c42_ui : !firrtl.uint, !firrtl.uint

    // Check that the invalid values are duplicated, and a corner case where the
    // wire won't be updated with a width until after updating the invalid value
    // above.
    // CHECK: %invalid_ui2 = invalidvalue : !firrtl.uint<2>
    %w = wire : !firrtl.uint
    %c2_ui = constant 2 : !firrtl.uint
    connect %w, %invalid_ui : !firrtl.uint, !firrtl.uint
    connect %w, %c2_ui : !firrtl.uint, !firrtl.uint

    // Check that invalid values are inferred to width zero if not used in a
    // connect.
    // CHECK: invalidvalue : !firrtl.uint<0>
    // CHECK: invalidvalue : !firrtl.bundle<x: uint<0>>
    // CHECK: invalidvalue : !firrtl.vector<uint<0>, 2>
    // CHECK: invalidvalue : !firrtl.enum<a: uint<0>>
    %invalid_0 = invalidvalue : !firrtl.uint
    %invalid_1 = invalidvalue : !firrtl.bundle<x: uint>
    %invalid_2 = invalidvalue : !firrtl.vector<uint, 2>
    %invalid_3 = invalidvalue : !firrtl.enum<a: uint>
  }

  // CHECK-LABEL: @InferOutput
  // CHECK-SAME: out %out: !firrtl.uint<2>
  module @InferOutput(in %in: !firrtl.uint<2>, out %out: !firrtl.uint) {
    connect %out, %in : !firrtl.uint, !firrtl.uint<2>
  }

  // CHECK-LABEL: @InferOutput2
  // CHECK-SAME: out %out: !firrtl.uint<2>
  module @InferOutput2(in %in: !firrtl.uint<2>, out %out: !firrtl.uint) {
    connect %out, %in : !firrtl.uint, !firrtl.uint<2>
  }

  module @InferNode() {
    %w = wire : !firrtl.uint
    %c2_ui3 = constant 2 : !firrtl.uint<3>
    connect %w, %c2_ui3 : !firrtl.uint, !firrtl.uint<3>
    // CHECK: %node = node %w : !firrtl.uint<3>
    %node = node %w : !firrtl.uint
  }

  module @InferNode2() {
    %c2_ui3 = constant 2 : !firrtl.uint<3>
    %w = wire : !firrtl.uint
    connect %w, %c2_ui3 : !firrtl.uint, !firrtl.uint<3>

    %node2 = node %w : !firrtl.uint

    %w1 = wire : !firrtl.uint
    connect %w1, %node2 : !firrtl.uint, !firrtl.uint
  }

  // CHECK-LABEL: @AddSubOp
  module @AddSubOp() {
    // CHECK: %0 = wire : !firrtl.uint<2>
    // CHECK: %1 = wire : !firrtl.uint<3>
    // CHECK: %2 = add {{.*}} -> !firrtl.uint<4>
    // CHECK: %3 = sub {{.*}} -> !firrtl.uint<5>
    %0 = wire : !firrtl.uint
    %1 = wire : !firrtl.uint
    %2 = add %0, %1 : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    %3 = sub %0, %2 : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    %c1_ui2 = constant 1 : !firrtl.uint<2>
    %c2_ui3 = constant 2 : !firrtl.uint<3>
    connect %0, %c1_ui2 : !firrtl.uint, !firrtl.uint<2>
    connect %1, %c2_ui3 : !firrtl.uint, !firrtl.uint<3>
  }

  // CHECK-LABEL: @MulDivRemOp
  module @MulDivRemOp() {
    // CHECK: %0 = wire : !firrtl.uint<2>
    // CHECK: %1 = wire : !firrtl.uint<3>
    // CHECK: %2 = wire : !firrtl.sint<2>
    // CHECK: %3 = wire : !firrtl.sint<3>
    // CHECK: %4 = mul {{.*}} -> !firrtl.uint<5>
    // CHECK: %5 = div {{.*}} -> !firrtl.uint<3>
    // CHECK: %6 = div {{.*}} -> !firrtl.sint<4>
    // CHECK: %7 = rem {{.*}} -> !firrtl.uint<2>
    %0 = wire : !firrtl.uint
    %1 = wire : !firrtl.uint
    %2 = wire : !firrtl.sint
    %3 = wire : !firrtl.sint
    %4 = mul %1, %0 : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    %5 = div %1, %0 : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    %6 = div %3, %2 : (!firrtl.sint, !firrtl.sint) -> !firrtl.sint
    %7 = rem %1, %0 : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    %c1_ui2 = constant 1 : !firrtl.uint<2>
    %c2_ui3 = constant 2 : !firrtl.uint<3>
    %c1_si2 = constant 1 : !firrtl.sint<2>
    %c2_si3 = constant 2 : !firrtl.sint<3>
    connect %0, %c1_ui2 : !firrtl.uint, !firrtl.uint<2>
    connect %1, %c2_ui3 : !firrtl.uint, !firrtl.uint<3>
    connect %2, %c1_si2 : !firrtl.sint, !firrtl.sint<2>
    connect %3, %c2_si3 : !firrtl.sint, !firrtl.sint<3>
  }

  // CHECK-LABEL: @AndOrXorOp
  module @AndOrXorOp() {
    // CHECK: %0 = wire : !firrtl.uint<2>
    // CHECK: %1 = wire : !firrtl.uint<3>
    // CHECK: %2 = and {{.*}} -> !firrtl.uint<3>
    // CHECK: %3 = or {{.*}} -> !firrtl.uint<3>
    // CHECK: %4 = xor {{.*}} -> !firrtl.uint<3>
    %0 = wire : !firrtl.uint
    %1 = wire : !firrtl.uint
    %2 = and %0, %1 : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    %3 = or %0, %1 : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    %4 = xor %0, %1 : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    %c1_ui2 = constant 1 : !firrtl.uint<2>
    %c2_ui3 = constant 2 : !firrtl.uint<3>
    connect %0, %c1_ui2 : !firrtl.uint, !firrtl.uint<2>
    connect %1, %c2_ui3 : !firrtl.uint, !firrtl.uint<3>
  }

  // CHECK-LABEL: @ComparisonOp
  module @ComparisonOp(in %a: !firrtl.uint<2>, in %b: !firrtl.uint<3>) {
    // CHECK: %6 = wire : !firrtl.uint<1>
    // CHECK: %7 = wire : !firrtl.uint<1>
    // CHECK: %8 = wire : !firrtl.uint<1>
    // CHECK: %9 = wire : !firrtl.uint<1>
    // CHECK: %10 = wire : !firrtl.uint<1>
    // CHECK: %11 = wire : !firrtl.uint<1>
    %0 = leq %a, %b : (!firrtl.uint<2>, !firrtl.uint<3>) -> !firrtl.uint<1>
    %1 = lt %a, %b : (!firrtl.uint<2>, !firrtl.uint<3>) -> !firrtl.uint<1>
    %2 = geq %a, %b : (!firrtl.uint<2>, !firrtl.uint<3>) -> !firrtl.uint<1>
    %3 = gt %a, %b : (!firrtl.uint<2>, !firrtl.uint<3>) -> !firrtl.uint<1>
    %4 = eq %a, %b : (!firrtl.uint<2>, !firrtl.uint<3>) -> !firrtl.uint<1>
    %5 = neq %a, %b : (!firrtl.uint<2>, !firrtl.uint<3>) -> !firrtl.uint<1>
    %6 = wire : !firrtl.uint
    %7 = wire : !firrtl.uint
    %8 = wire : !firrtl.uint
    %9 = wire : !firrtl.uint
    %10 = wire : !firrtl.uint
    %11 = wire : !firrtl.uint
    connect %6, %0 : !firrtl.uint, !firrtl.uint<1>
    connect %7, %1 : !firrtl.uint, !firrtl.uint<1>
    connect %8, %2 : !firrtl.uint, !firrtl.uint<1>
    connect %9, %3 : !firrtl.uint, !firrtl.uint<1>
    connect %10, %4 : !firrtl.uint, !firrtl.uint<1>
    connect %11, %5 : !firrtl.uint, !firrtl.uint<1>
  }

  // CHECK-LABEL: @CatDynShiftOp
  module @CatDynShiftOp() {
    // CHECK: %0 = wire : !firrtl.uint<2>
    // CHECK: %1 = wire : !firrtl.uint<3>
    // CHECK: %2 = wire : !firrtl.sint<2>
    // CHECK: %3 = wire : !firrtl.sint<3>
    // CHECK: %4 = cat {{.*}} -> !firrtl.uint<5>
    // CHECK: %5 = cat {{.*}} -> !firrtl.uint<5>
    // CHECK: %6 = dshl {{.*}} -> !firrtl.uint<10>
    // CHECK: %7 = dshl {{.*}} -> !firrtl.sint<10>
    // CHECK: %8 = dshlw {{.*}} -> !firrtl.uint<3>
    // CHECK: %9 = dshlw {{.*}} -> !firrtl.sint<3>
    // CHECK: %10 = dshr {{.*}} -> !firrtl.uint<3>
    // CHECK: %11 = dshr {{.*}} -> !firrtl.sint<3>
    %0 = wire : !firrtl.uint
    %1 = wire : !firrtl.uint
    %2 = wire : !firrtl.sint
    %3 = wire : !firrtl.sint
    %4 = cat %0, %1 : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    %5 = cat %2, %3 : (!firrtl.sint, !firrtl.sint) -> !firrtl.uint
    %6 = dshl %1, %1 : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    %7 = dshl %3, %1 : (!firrtl.sint, !firrtl.uint) -> !firrtl.sint
    %8 = dshlw %1, %1 : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    %9 = dshlw %3, %1 : (!firrtl.sint, !firrtl.uint) -> !firrtl.sint
    %10 = dshr %1, %1 : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    %11 = dshr %3, %1 : (!firrtl.sint, !firrtl.uint) -> !firrtl.sint
    %c1_ui2 = constant 1 : !firrtl.uint<2>
    %c2_ui3 = constant 2 : !firrtl.uint<3>
    %c1_si2 = constant 1 : !firrtl.sint<2>
    %c2_si3 = constant 2 : !firrtl.sint<3>
    connect %0, %c1_ui2 : !firrtl.uint, !firrtl.uint<2>
    connect %1, %c2_ui3 : !firrtl.uint, !firrtl.uint<3>
    connect %2, %c1_si2 : !firrtl.sint, !firrtl.sint<2>
    connect %3, %c2_si3 : !firrtl.sint, !firrtl.sint<3>
  }

  // CHECK-LABEL: @CastOp
  module @CastOp() {
    %c0_ui1 = constant 0 : !firrtl.uint<1>
    // CHECK: %0 = wire : !firrtl.uint<2>
    // CHECK: %1 = wire : !firrtl.sint<3>
    // CHECK: %4 = asSInt {{.*}} -> !firrtl.sint<2>
    // CHECK: %5 = asUInt {{.*}} -> !firrtl.uint<3>
    %0 = wire : !firrtl.uint
    %1 = wire : !firrtl.sint
    %2 = wire : !firrtl.clock
    %3 = wire : !firrtl.asyncreset
    %4 = asSInt %0 : (!firrtl.uint) -> !firrtl.sint
    %5 = asUInt %1 : (!firrtl.sint) -> !firrtl.uint
    %6 = asUInt %2 : (!firrtl.clock) -> !firrtl.uint<1>
    %7 = asUInt %3 : (!firrtl.asyncreset) -> !firrtl.uint<1>
    %8 = asClock %c0_ui1 : (!firrtl.uint<1>) -> !firrtl.clock
    %9 = asAsyncReset %c0_ui1 : (!firrtl.uint<1>) -> !firrtl.asyncreset
    %c1_ui2 = constant 1 : !firrtl.uint<2>
    %c2_si3 = constant 2 : !firrtl.sint<3>
    connect %0, %c1_ui2 : !firrtl.uint, !firrtl.uint<2>
    connect %1, %c2_si3 : !firrtl.sint, !firrtl.sint<3>
  }

  // CHECK-LABEL: @ConstCastOp
  module @ConstCastOp() {
    %c0_ui1 = constant 0 : !firrtl.const.uint<1>
    // CHECK: %0 = wire : !firrtl.uint<2>
    // CHECK: %1 = wire : !firrtl.sint<3>
    %0 = wire : !firrtl.uint
    %1 = wire : !firrtl.sint
    %c1 = constant 1 : !firrtl.const.uint<2>
    %c2 = constant 2 : !firrtl.const.sint<3>
    %3 = constCast %c1 : (!firrtl.const.uint<2>) -> !firrtl.uint<2>
    %4 = constCast %c2 : (!firrtl.const.sint<3>) -> !firrtl.sint<3>
    connect %0, %3 : !firrtl.uint, !firrtl.uint<2>
    connect %1, %4 : !firrtl.sint, !firrtl.sint<3>
  }

  // CHECK-LABEL: @CvtOp
  module @CvtOp() {
    // CHECK: %0 = wire : !firrtl.uint<2>
    // CHECK: %1 = wire : !firrtl.sint<3>
    // CHECK: %2 = cvt {{.*}} -> !firrtl.sint<3>
    // CHECK: %3 = cvt {{.*}} -> !firrtl.sint<3>
    %0 = wire : !firrtl.uint
    %1 = wire : !firrtl.sint
    %2 = cvt %0 : (!firrtl.uint) -> !firrtl.sint
    %3 = cvt %1 : (!firrtl.sint) -> !firrtl.sint
    %c1_ui2 = constant 1 : !firrtl.uint<2>
    %c2_si3 = constant 2 : !firrtl.sint<3>
    connect %0, %c1_ui2 : !firrtl.uint, !firrtl.uint<2>
    connect %1, %c2_si3 : !firrtl.sint, !firrtl.sint<3>
  }

  // CHECK-LABEL: @NegOp
  module @NegOp() {
    // CHECK: %0 = wire : !firrtl.uint<2>
    // CHECK: %1 = wire : !firrtl.sint<3>
    // CHECK: %2 = neg {{.*}} -> !firrtl.sint<3>
    // CHECK: %3 = neg {{.*}} -> !firrtl.sint<4>
    %0 = wire : !firrtl.uint
    %1 = wire : !firrtl.sint
    %2 = neg %0 : (!firrtl.uint) -> !firrtl.sint
    %3 = neg %1 : (!firrtl.sint) -> !firrtl.sint
    %c1_ui2 = constant 1 : !firrtl.uint<2>
    %c2_si3 = constant 2 : !firrtl.sint<3>
    connect %0, %c1_ui2 : !firrtl.uint, !firrtl.uint<2>
    connect %1, %c2_si3 : !firrtl.sint, !firrtl.sint<3>
  }

  // CHECK-LABEL: @NotOp
  module @NotOp() {
    // CHECK: %0 = wire : !firrtl.uint<2>
    // CHECK: %1 = wire : !firrtl.sint<3>
    // CHECK: %2 = not {{.*}} -> !firrtl.uint<2>
    // CHECK: %3 = not {{.*}} -> !firrtl.uint<3>
    %0 = wire : !firrtl.uint
    %1 = wire : !firrtl.sint
    %2 = not %0 : (!firrtl.uint) -> !firrtl.uint
    %3 = not %1 : (!firrtl.sint) -> !firrtl.uint
    %c1_ui2 = constant 1 : !firrtl.uint<2>
    %c2_si3 = constant 2 : !firrtl.sint<3>
    connect %0, %c1_ui2 : !firrtl.uint, !firrtl.uint<2>
    connect %1, %c2_si3 : !firrtl.sint, !firrtl.sint<3>
  }

  // CHECK-LABEL: @AndOrXorReductionOp
  module @AndOrXorReductionOp() {
    // CHECK: %0 = wire : !firrtl.uint<1>
    // CHECK: %1 = wire : !firrtl.uint<1>
    // CHECK: %2 = wire : !firrtl.uint<1>
    // CHECK: %3 = andr {{.*}} -> !firrtl.uint<1>
    // CHECK: %4 = orr {{.*}} -> !firrtl.uint<1>
    // CHECK: %5 = xorr {{.*}} -> !firrtl.uint<1>
    %c0_ui16 = constant 0 : !firrtl.uint<16>
    %0 = wire : !firrtl.uint
    %1 = wire : !firrtl.uint
    %2 = wire : !firrtl.uint
    %3 = andr %c0_ui16 : (!firrtl.uint<16>) -> !firrtl.uint<1>
    %4 = orr %c0_ui16 : (!firrtl.uint<16>) -> !firrtl.uint<1>
    %5 = xorr %c0_ui16 : (!firrtl.uint<16>) -> !firrtl.uint<1>
    connect %0, %3 : !firrtl.uint, !firrtl.uint<1>
    connect %1, %4 : !firrtl.uint, !firrtl.uint<1>
    connect %2, %5 : !firrtl.uint, !firrtl.uint<1>
  }

  // CHECK-LABEL: @BitsHeadTailPadOp
  module @BitsHeadTailPadOp() {
    // CHECK: %0 = wire : !firrtl.uint<3>
    // CHECK: %1 = wire : !firrtl.uint<3>
    // CHECK: %2 = wire : !firrtl.uint<5>
    // CHECK: %3 = wire : !firrtl.uint<5>
    // CHECK: %8 = tail {{.*}} -> !firrtl.uint<12>
    // CHECK: %9 = tail {{.*}} -> !firrtl.uint<12>
    // CHECK: %10 = pad {{.*}} -> !firrtl.uint<42>
    // CHECK: %11 = pad {{.*}} -> !firrtl.sint<42>
    // CHECK: %12 = pad {{.*}} -> !firrtl.uint<99>
    // CHECK: %13 = pad {{.*}} -> !firrtl.sint<99>
    %ui = wire : !firrtl.uint
    %si = wire : !firrtl.sint
    %0 = wire : !firrtl.uint
    %1 = wire : !firrtl.uint
    %2 = wire : !firrtl.uint
    %3 = wire : !firrtl.uint

    %4 = bits %ui 3 to 1 : (!firrtl.uint) -> !firrtl.uint<3>
    %5 = bits %si 3 to 1 : (!firrtl.sint) -> !firrtl.uint<3>
    %6 = head %ui, 5 : (!firrtl.uint) -> !firrtl.uint<5>
    %7 = head %si, 5 : (!firrtl.sint) -> !firrtl.uint<5>
    %8 = tail %ui, 30 : (!firrtl.uint) -> !firrtl.uint
    %9 = tail %si, 30 : (!firrtl.sint) -> !firrtl.uint
    %10 = pad %ui, 13 : (!firrtl.uint) -> !firrtl.uint
    %11 = pad %si, 13 : (!firrtl.sint) -> !firrtl.sint
    %12 = pad %ui, 99 : (!firrtl.uint) -> !firrtl.uint
    %13 = pad %si, 99 : (!firrtl.sint) -> !firrtl.sint

    connect %0, %4 : !firrtl.uint, !firrtl.uint<3>
    connect %1, %5 : !firrtl.uint, !firrtl.uint<3>
    connect %2, %6 : !firrtl.uint, !firrtl.uint<5>
    connect %3, %7 : !firrtl.uint, !firrtl.uint<5>

    %c0_ui42 = constant 0 : !firrtl.uint<42>
    %c0_si42 = constant 0 : !firrtl.sint<42>
    connect %ui, %c0_ui42 : !firrtl.uint, !firrtl.uint<42>
    connect %si, %c0_si42 : !firrtl.sint, !firrtl.sint<42>
  }

  // CHECK-LABEL: @MuxOp
  module @MuxOp() {
    // CHECK: %0 = wire : !firrtl.uint<2>
    // CHECK: %1 = wire : !firrtl.uint<3>
    // CHECK: %2 = wire : !firrtl.uint<1>
    // CHECK: %3 = mux{{.*}} -> !firrtl.uint<3>
    %0 = wire : !firrtl.uint
    %1 = wire : !firrtl.uint
    %2 = wire : !firrtl.uint
    %3 = mux(%2, %0, %1) : (!firrtl.uint, !firrtl.uint, !firrtl.uint) -> !firrtl.uint
    // CHECK: %4 = wire : !firrtl.uint<1>
    %c1_ui1 = constant 1 : !firrtl.uint<1>
    %4 = wire : !firrtl.uint
    %5 = mux(%4, %c1_ui1, %c1_ui1) : (!firrtl.uint, !firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    %c1_ui2 = constant 1 : !firrtl.uint<2>
    %c2_ui3 = constant 2 : !firrtl.uint<3>
    connect %0, %c1_ui2 : !firrtl.uint, !firrtl.uint<2>
    connect %1, %c2_ui3 : !firrtl.uint, !firrtl.uint<3>
  }

  // see https://github.com/llvm/circt/issues/3070
  // CHECK-LABEL: @MuxBundle
  module @MuxBundleOperands(in %a: !firrtl.bundle<a: uint<8>>, in %p: !firrtl.uint<1>, out %c: !firrtl.bundle<a: uint>) {
    // CHECK: %w = wire  : !firrtl.bundle<a: uint<8>>
    %w = wire  : !firrtl.bundle<a: uint>
    %0 = subfield %w[a] : !firrtl.bundle<a: uint>
    %1 = subfield %a[a] : !firrtl.bundle<a: uint<8>>
    connect %0, %1 : !firrtl.uint, !firrtl.uint<8>
    // CHECK: %2 = mux(%p, %a, %w) : (!firrtl.uint<1>, !firrtl.bundle<a: uint<8>>, !firrtl.bundle<a: uint<8>>) -> !firrtl.bundle<a: uint<8>>
    %2 = mux(%p, %a, %w) : (!firrtl.uint<1>, !firrtl.bundle<a: uint<8>>, !firrtl.bundle<a: uint>) -> !firrtl.bundle<a: uint>
    connect %c, %2 : !firrtl.bundle<a: uint>, !firrtl.bundle<a: uint>
  }

  // CHECK-LABEL: @ShlShrOp
  module @ShlShrOp() {
    // CHECK: %0 = shl {{.*}} -> !firrtl.uint<8>
    // CHECK: %1 = shl {{.*}} -> !firrtl.sint<8>
    // CHECK: %2 = shr {{.*}} -> !firrtl.uint<2>
    // CHECK: %3 = shr {{.*}} -> !firrtl.sint<2>
    // CHECK: %4 = shr {{.*}} -> !firrtl.uint<1>
    // CHECK: %5 = shr {{.*}} -> !firrtl.sint<1>
    %ui = wire : !firrtl.uint
    %si = wire : !firrtl.sint

    %0 = shl %ui, 3 : (!firrtl.uint) -> !firrtl.uint
    %1 = shl %si, 3 : (!firrtl.sint) -> !firrtl.sint
    %2 = shr %ui, 3 : (!firrtl.uint) -> !firrtl.uint
    %3 = shr %si, 3 : (!firrtl.sint) -> !firrtl.sint
    %4 = shr %ui, 9 : (!firrtl.uint) -> !firrtl.uint
    %5 = shr %si, 9 : (!firrtl.sint) -> !firrtl.sint

    %c0_ui5 = constant 0 : !firrtl.uint<5>
    %c0_si5 = constant 0 : !firrtl.sint<5>
    connect %ui, %c0_ui5 : !firrtl.uint, !firrtl.uint<5>
    connect %si, %c0_si5 : !firrtl.sint, !firrtl.sint<5>
  }

  // CHECK-LABEL: @PassiveCastOp
  module @PassiveCastOp() {
    // CHECK: %0 = wire : !firrtl.uint<5>
    // CHECK: %1 = builtin.unrealized_conversion_cast %ui : !firrtl.uint<5> to !firrtl.uint<5>
    %ui = wire : !firrtl.uint
    %0 = wire : !firrtl.uint
    %1 = builtin.unrealized_conversion_cast %ui : !firrtl.uint to !firrtl.uint
    connect %0, %1 : !firrtl.uint, !firrtl.uint
    %c0_ui5 = constant 0 : !firrtl.uint<5>
    connect %ui, %c0_ui5 : !firrtl.uint, !firrtl.uint<5>
  }

  // CHECK-LABEL: @TransparentOps
  module @TransparentOps(in %clk: !firrtl.clock, in %a: !firrtl.uint<1>) {
    %false = constant 0 : !firrtl.uint<1>
    %true = constant 1 : !firrtl.uint<1>
    %c0_ui4 = constant 0 : !firrtl.uint<4>
    %c0_ui5 = constant 0 : !firrtl.uint<5>

    // CHECK: %ui = wire : !firrtl.uint<5>
    %ui = wire : !firrtl.uint

    printf %clk, %false, "foo" : !firrtl.clock, !firrtl.uint<1>
    skip
    stop %clk, %false, 0 : !firrtl.clock, !firrtl.uint<1>
    when %a : !firrtl.uint<1> {
      connect %ui, %c0_ui4 : !firrtl.uint, !firrtl.uint<4>
    } else  {
      connect %ui, %c0_ui5 : !firrtl.uint, !firrtl.uint<5>
    }
    assert %clk, %true, %true, "foo" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>
    assume %clk, %true, %true, "foo" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>
    cover %clk, %true, %true, "foo" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>
  }

  // Issue #1088
  // CHECK-LABEL: @Issue1088
  module @Issue1088(out %y: !firrtl.sint<4>) {
    // CHECK: %x = wire : !firrtl.sint<9>
    // CHECK: %c200_si9 = constant 200 : !firrtl.sint<9>
    // CHECK: %0 = tail %x, 5 : (!firrtl.sint<9>) -> !firrtl.uint<4>
    // CHECK: %1 = asSInt %0 : (!firrtl.uint<4>) -> !firrtl.sint<4>
    // CHECK: connect %y, %1 : !firrtl.sint<4>, !firrtl.sint<4>
    // CHECK: connect %x, %c200_si9 : !firrtl.sint<9>, !firrtl.sint<9>
    %x = wire : !firrtl.sint
    %c200_si = constant 200 : !firrtl.sint
    connect %y, %x : !firrtl.sint<4>, !firrtl.sint
    connect %x, %c200_si : !firrtl.sint, !firrtl.sint
  }

  // Should truncate all the way to 0 bits if its has to.
  // CHECK-LABEL: @TruncateConnect
  module @TruncateConnect() {
    %w = wire  : !firrtl.uint
    %c1_ui1 = constant 1 : !firrtl.uint<1>
    connect %w, %c1_ui1 : !firrtl.uint, !firrtl.uint<1>
    %w1 = wire  : !firrtl.uint<0>
    // CHECK: %0 = tail %w, 1 : (!firrtl.uint<1>) -> !firrtl.uint<0>
    // CHECK: connect %w1, %0 : !firrtl.uint<0>, !firrtl.uint<0>
    connect %w1, %w : !firrtl.uint<0>, !firrtl.uint
  }

  // Issue #1110: Width inference should infer 0 width when appropriate
  // CHECK-LABEL: @Issue1110
  // CHECK-SAME: out %y: !firrtl.uint<0>
  module @Issue1110(in %x: !firrtl.uint<0>, out %y: !firrtl.uint) {
    connect %y, %x : !firrtl.uint, !firrtl.uint<0>
  }

  // Issue #1118: Width inference should infer 0 width when appropriate
  // CHECK-LABEL: @Issue1118
  // CHECK-SAME: out %x: !firrtl.sint<13>
  module @Issue1118(out %x: !firrtl.sint) {
    %c4232_ui = constant 4232 : !firrtl.uint
    %0 = asSInt %c4232_ui : (!firrtl.uint) -> !firrtl.sint
    connect %x, %0 : !firrtl.sint, !firrtl.sint
  }

  // CHECK-LABEL: @RegSimple
  module @RegSimple(in %clk: !firrtl.clock, in %x: !firrtl.uint<6>) {
    // CHECK: %0 = reg %clk : !firrtl.clock, !firrtl.uint<6>
    // CHECK: %1 = reg %clk : !firrtl.clock, !firrtl.uint<6>
    %0 = reg %clk : !firrtl.clock, !firrtl.uint
    %1 = reg %clk : !firrtl.clock, !firrtl.uint
    %2 = wire : !firrtl.uint
    %3 = xor %1, %2 : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    connect %0, %x : !firrtl.uint, !firrtl.uint<6>
    connect %1, %3 : !firrtl.uint, !firrtl.uint
    connect %2, %x : !firrtl.uint, !firrtl.uint<6>
  }

  // CHECK-LABEL: @RegShr
  module @RegShr(in %clk: !firrtl.clock, in %x: !firrtl.uint<6>) {
    // CHECK: %0 = reg %clk : !firrtl.clock, !firrtl.uint<6>
    // CHECK: %1 = reg %clk : !firrtl.clock, !firrtl.uint<6>
    %0 = reg %clk : !firrtl.clock, !firrtl.uint
    %1 = reg %clk : !firrtl.clock, !firrtl.uint
    %2 = shr %0, 0 : (!firrtl.uint) -> !firrtl.uint
    %3 = shr %1, 3 : (!firrtl.uint) -> !firrtl.uint
    connect %0, %x : !firrtl.uint, !firrtl.uint<6>
    connect %1, %x : !firrtl.uint, !firrtl.uint<6>
    connect %0, %2 : !firrtl.uint, !firrtl.uint
    connect %1, %3 : !firrtl.uint, !firrtl.uint
  }

  // CHECK-LABEL: @RegShl
  module @RegShl(in %clk: !firrtl.clock, in %x: !firrtl.uint<6>) {
    // CHECK: %0 = reg %clk : !firrtl.clock, !firrtl.uint<6>
    %0 = reg %clk : !firrtl.clock, !firrtl.uint
    %1 = reg %clk : !firrtl.clock, !firrtl.uint
    %2 = reg %clk : !firrtl.clock, !firrtl.uint
    %3 = shl %0, 0 : (!firrtl.uint) -> !firrtl.uint
    %4 = shl %1, 3 : (!firrtl.uint) -> !firrtl.uint
    %5 = shr %4, 3 : (!firrtl.uint) -> !firrtl.uint
    %6 = shr %1, 3 : (!firrtl.uint) -> !firrtl.uint
    %7 = shl %6, 3 : (!firrtl.uint) -> !firrtl.uint
    connect %0, %x : !firrtl.uint, !firrtl.uint<6>
    connect %1, %x : !firrtl.uint, !firrtl.uint<6>
    connect %2, %x : !firrtl.uint, !firrtl.uint<6>
    connect %0, %2 : !firrtl.uint, !firrtl.uint
    connect %1, %5 : !firrtl.uint, !firrtl.uint
    connect %2, %7 : !firrtl.uint, !firrtl.uint
  }

  // CHECK-LABEL: @RegResetSimple
  module @RegResetSimple(
    in %clk: !firrtl.clock,
    in %rst: !firrtl.asyncreset,
    in %x: !firrtl.uint<6>
  ) {
    // CHECK: %0 = regreset %clk, %rst, %c0_ui1 : !firrtl.clock, !firrtl.asyncreset, !firrtl.uint<1>, !firrtl.uint<6>
    // CHECK: %1 = regreset %clk, %rst, %c0_ui1 : !firrtl.clock, !firrtl.asyncreset, !firrtl.uint<1>, !firrtl.uint<6>
    // CHECK: %2:2 = regreset %clk, %rst, %c0_ui17 forceable : !firrtl.clock, !firrtl.asyncreset, !firrtl.uint<17>, !firrtl.uint<17>, !firrtl.rwprobe<uint<17>>
    // CHECK: %3 = regreset %clk, %rst, %c0_ui17 : !firrtl.clock, !firrtl.asyncreset, !firrtl.uint<17>, !firrtl.uint<17>
    %c0_ui = constant 0 : !firrtl.uint
    %c0_ui17 = constant 0 : !firrtl.uint<17>
    %0 = regreset %clk, %rst, %c0_ui : !firrtl.clock, !firrtl.asyncreset, !firrtl.uint, !firrtl.uint
    %1 = regreset %clk, %rst, %c0_ui : !firrtl.clock, !firrtl.asyncreset, !firrtl.uint, !firrtl.uint
    %2:2 = regreset %clk, %rst, %c0_ui17 forceable : !firrtl.clock, !firrtl.asyncreset, !firrtl.uint<17>, !firrtl.uint, !firrtl.rwprobe<uint>
    %3 = regreset %clk, %rst, %c0_ui17 : !firrtl.clock, !firrtl.asyncreset, !firrtl.uint<17>, !firrtl.uint
    %4 = wire : !firrtl.uint
    %5 = xor %1, %4 : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    connect %0, %x : !firrtl.uint, !firrtl.uint<6>
    connect %1, %5 : !firrtl.uint, !firrtl.uint
    connect %2, %x : !firrtl.uint, !firrtl.uint<6>
    connect %3, %5 : !firrtl.uint, !firrtl.uint
    connect %4, %x : !firrtl.uint, !firrtl.uint<6>
  }

  // Inter-module width inference for one-to-one module-instance correspondence.
  // CHECK-LABEL: @InterModuleSimpleFoo
  // CHECK-SAME: in %in: !firrtl.uint<42>
  // CHECK-SAME: out %out: !firrtl.uint<43>
  // CHECK-LABEL: @InterModuleSimpleBar
  // CHECK-SAME: in %in: !firrtl.uint<42>
  // CHECK-SAME: out %out: !firrtl.uint<44>
  module @InterModuleSimpleFoo(in %in: !firrtl.uint, out %out: !firrtl.uint) {
    %0 = add %in, %in : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    connect %out, %0 : !firrtl.uint, !firrtl.uint
  }
  module @InterModuleSimpleBar(in %in: !firrtl.uint<42>, out %out: !firrtl.uint) {
    %inst_in, %inst_out = instance inst @InterModuleSimpleFoo(in in: !firrtl.uint, out out: !firrtl.uint)
    %0 = add %inst_out, %inst_out : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    connect %inst_in, %in : !firrtl.uint, !firrtl.uint<42>
    connect %out, %0 : !firrtl.uint, !firrtl.uint
  }

  // Inter-module width inference for multiple instances per module.
  // CHECK-LABEL: @InterModuleMultipleFoo
  // CHECK-SAME: in %in: !firrtl.uint<42>
  // CHECK-SAME: out %out: !firrtl.uint<43>
  // CHECK-LABEL: @InterModuleMultipleBar
  // CHECK-SAME: in %in1: !firrtl.uint<17>
  // CHECK-SAME: in %in2: !firrtl.uint<42>
  // CHECK-SAME: out %out: !firrtl.uint<43>
  module @InterModuleMultipleFoo(in %in: !firrtl.uint, out %out: !firrtl.uint) {
    %0 = add %in, %in : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    connect %out, %0 : !firrtl.uint, !firrtl.uint
  }
  module @InterModuleMultipleBar(in %in1: !firrtl.uint<17>, in %in2: !firrtl.uint<42>, out %out: !firrtl.uint) {
    %inst1_in, %inst1_out = instance inst1 @InterModuleMultipleFoo(in in: !firrtl.uint, out out: !firrtl.uint)
    %inst2_in, %inst2_out = instance inst2 @InterModuleMultipleFoo(in in: !firrtl.uint, out out: !firrtl.uint)
    %0 = xor %inst1_out, %inst2_out : (!firrtl.uint, !firrtl.uint) -> !firrtl.uint
    connect %inst1_in, %in1 : !firrtl.uint, !firrtl.uint<17>
    connect %inst2_in, %in2 : !firrtl.uint, !firrtl.uint<42>
    connect %out, %0 : !firrtl.uint, !firrtl.uint
  }

  // CHECK-LABEL: @InferBundle
  module @InferBundle(in %in : !firrtl.uint<3>, in %clk : !firrtl.clock) {
    // CHECK: wire : !firrtl.bundle<a: uint<3>>
    // CHECK: reg %clk : !firrtl.clock, !firrtl.bundle<a: uint<3>>
    %w = wire : !firrtl.bundle<a: uint>
    %r = reg %clk : !firrtl.clock, !firrtl.bundle<a: uint>
    %w_a = subfield %w[a] : !firrtl.bundle<a: uint>
    %r_a = subfield %r[a] : !firrtl.bundle<a: uint>
    connect %w_a, %in : !firrtl.uint, !firrtl.uint<3>
    connect %r_a, %in : !firrtl.uint, !firrtl.uint<3>
  }

  // CHECK-LABEL: @InferEmptyBundle
  module @InferEmptyBundle(in %in : !firrtl.uint<3>) {
    // CHECK: %w = wire : !firrtl.bundle<a: bundle<>, b: uint<3>>
    %w = wire : !firrtl.bundle<a: bundle<>, b: uint>
    %w_a = subfield %w[a] : !firrtl.bundle<a: bundle<>, b: uint>
    %w_b = subfield %w[b] : !firrtl.bundle<a: bundle<>, b: uint>
    connect %w_b, %in : !firrtl.uint, !firrtl.uint<3>
  }

  // CHECK-LABEL: @InferBundlePort
  module @InferBundlePort(in %in: !firrtl.bundle<a: uint<2>, b: uint<3>>, out %out: !firrtl.bundle<a: uint, b: uint>) {
    // CHECK: connect %out, %in : !firrtl.bundle<a: uint<2>, b: uint<3>>, !firrtl.bundle<a: uint<2>, b: uint<3>>
    connect %out, %in : !firrtl.bundle<a: uint, b: uint>, !firrtl.bundle<a: uint<2>, b: uint<3>>
  }

  // CHECK-LABEL: @InferVectorSubindex
  module @InferVectorSubindex(in %in : !firrtl.uint<4>, in %clk : !firrtl.clock) {
    // CHECK: wire : !firrtl.vector<uint<4>, 10>
    // CHECK: reg %clk : !firrtl.clock, !firrtl.vector<uint<4>, 10>
    %w = wire : !firrtl.vector<uint, 10>
    %r = reg %clk : !firrtl.clock, !firrtl.vector<uint, 10>
    %w_5 = subindex %w[5] : !firrtl.vector<uint, 10>
    %r_5 = subindex %r[5] : !firrtl.vector<uint, 10>
    connect %w_5, %in : !firrtl.uint, !firrtl.uint<4>
    connect %r_5, %in : !firrtl.uint, !firrtl.uint<4>
  }

  // CHECK-LABEL: @InferVectorSubaccess
  module @InferVectorSubaccess(in %in : !firrtl.uint<4>, in %addr : !firrtl.uint<32>, in %clk : !firrtl.clock) {
    // CHECK: wire : !firrtl.vector<uint<4>, 10>
    // CHECK: reg %clk : !firrtl.clock, !firrtl.vector<uint<4>, 10>
    %w = wire : !firrtl.vector<uint, 10>
    %r = reg %clk : !firrtl.clock, !firrtl.vector<uint, 10>
    %w_addr = subaccess %w[%addr] : !firrtl.vector<uint, 10>, !firrtl.uint<32>
    %r_addr = subaccess %r[%addr] : !firrtl.vector<uint, 10>, !firrtl.uint<32>
    connect %w_addr, %in : !firrtl.uint, !firrtl.uint<4>
    connect %r_addr, %in : !firrtl.uint, !firrtl.uint<4>
  }

  // CHECK-LABEL: @InferVectorPort
  module @InferVectorPort(in %in: !firrtl.vector<uint<4>, 2>, out %out: !firrtl.vector<uint, 2>) {
    // CHECK: connect %out, %in : !firrtl.vector<uint<4>, 2>, !firrtl.vector<uint<4>, 2>
    connect %out, %in : !firrtl.vector<uint, 2>, !firrtl.vector<uint<4>, 2>
  }

  // CHECK-LABEL: @InferVectorFancy
  module @InferVectorFancy(in %in : !firrtl.uint<4>) {
    // CHECK: wire : !firrtl.vector<uint<4>, 10>
    %wv = wire : !firrtl.vector<uint, 10>
    %wv_5 = subindex %wv[5] : !firrtl.vector<uint, 10>
    connect %wv_5, %in : !firrtl.uint, !firrtl.uint<4>

    // CHECK: wire : !firrtl.bundle<a: uint<4>>
    %wb = wire : !firrtl.bundle<a: uint>
    %wb_a = subfield %wb[a] : !firrtl.bundle<a: uint>

    %wv_2 = subindex %wv[2] : !firrtl.vector<uint, 10>
    connect %wb_a, %wv_2 : !firrtl.uint, !firrtl.uint
  }

  // CHECK-LABEL: InferElementAfterVector
  module @InferElementAfterVector() {
    // CHECK: %w = wire : !firrtl.bundle<a: vector<uint<10>, 10>, b: uint<3>>
    %w = wire : !firrtl.bundle<a: vector<uint<10>, 10>, b :uint>
    %w_a = subfield %w[b] : !firrtl.bundle<a: vector<uint<10>, 10>, b: uint>
    %c2_ui3 = constant 2 : !firrtl.uint<3>
    connect %w_a, %c2_ui3 : !firrtl.uint, !firrtl.uint<3>
  }

  // CHECK-LABEL: @InferEnum
  module @InferEnum(in %in : !firrtl.enum<a: uint<3>>) {
    // CHECK: %w = wire : !firrtl.enum<a: uint<3>>
    %w = wire : !firrtl.enum<a: uint>
    connect %w, %in : !firrtl.enum<a: uint>, !firrtl.enum<a: uint<3>>
    // CHECK: %0 = subtag %w[a] : !firrtl.enum<a: uint<3>>
    %0 = subtag %w[a] : !firrtl.enum<a: uint>
  }

  // CHECK-LABEL: InferComplexBundles
  module @InferComplexBundles() {
    // CHECK: %w = wire : !firrtl.bundle<a: bundle<v: vector<uint<3>, 10>>, b: bundle<v: vector<uint<3>, 10>>>
    %w = wire : !firrtl.bundle<a: bundle<v: vector<uint, 10>>, b: bundle <v: vector<uint, 10>>>
    %w_a = subfield %w[a] : !firrtl.bundle<a: bundle<v: vector<uint, 10>>, b: bundle <v: vector<uint, 10>>>
    %w_a_v = subfield %w_a[v] : !firrtl.bundle<v : vector<uint, 10>>
    %w_b = subfield %w[b] : !firrtl.bundle<a: bundle<v: vector<uint, 10>>, b: bundle <v: vector<uint, 10>>>
    %w_b_v = subfield %w_b[v] : !firrtl.bundle<v : vector<uint, 10>>
    connect %w_a_v, %w_b_v : !firrtl.vector<uint, 10>, !firrtl.vector<uint, 10>
    %w_b_v_2 = subindex %w_b_v[2] : !firrtl.vector<uint, 10>
    %c2_ui3 = constant 2 : !firrtl.uint<3>
    connect %w_b_v_2, %c2_ui3 : !firrtl.uint, !firrtl.uint<3>
  }

  // CHECK-LABEL: InferComplexVectors
  module @InferComplexVectors() {
    // CHECK: %w = wire : !firrtl.vector<bundle<a: uint<3>, b: uint<3>>, 10>
    %w = wire : !firrtl.vector<bundle<a: uint, b:uint>, 10>
    %w_2 = subindex %w[2] : !firrtl.vector<bundle<a: uint, b:uint>, 10>
    %w_2_a = subfield %w_2[a] : !firrtl.bundle<a: uint, b: uint>
    %w_4 = subindex %w[4] : !firrtl.vector<bundle<a: uint, b:uint>, 10>
    %w_4_b = subfield %w_4[b] : !firrtl.bundle<a: uint, b: uint>
    connect %w_4_b, %w_2_a : !firrtl.uint, !firrtl.uint
    %c2_ui3 = constant 2 : !firrtl.uint<3>
    connect %w_2_a, %c2_ui3 : !firrtl.uint, !firrtl.uint<3>
  }

  // CHECK-LABEL: @AttachOne
  // CHECK-SAME: in %a0: !firrtl.analog<8>
  module @AttachOne(in %a0: !firrtl.analog<8>) {
    attach %a0 : !firrtl.analog<8>
  }

  // CHECK-LABEL: @AttachTwo
  // CHECK-SAME: in %a0: !firrtl.analog<8>
  // CHECK-SAME: in %a1: !firrtl.analog<8>
  module @AttachTwo(in %a0: !firrtl.analog<8>, in %a1: !firrtl.analog) {
    attach %a0, %a1 : !firrtl.analog<8>, !firrtl.analog
  }

  // CHECK-LABEL: @AttachMany
  // CHECK-SAME: in %a0: !firrtl.analog<8>
  // CHECK-SAME: in %a1: !firrtl.analog<8>
  // CHECK-SAME: in %a2: !firrtl.analog<8>
  // CHECK-SAME: in %a3: !firrtl.analog<8>
  module @AttachMany(
    in %a0: !firrtl.analog<8>,
    in %a1: !firrtl.analog,
    in %a2: !firrtl.analog<8>,
    in %a3: !firrtl.analog) {
    attach %a0, %a1, %a2, %a3 : !firrtl.analog<8>, !firrtl.analog, !firrtl.analog<8>, !firrtl.analog
  }

  // CHECK-LABEL: @MemScalar
  // CHECK-SAME: out %out: !firrtl.uint<7>
  // CHECK-SAME: out %dbg: !firrtl.probe<vector<uint<7>, 8>>
  module @MemScalar(out %out: !firrtl.uint, out %dbg: !firrtl.probe<vector<uint, 8>>) {
    // CHECK: mem
    // CHECK-SAME: !firrtl.probe<vector<uint<7>, 8>>
    // CHECK-SAME: data flip: uint<7>
    // CHECK-SAME: data: uint<7>
    // CHECK-SAME: data: uint<7>
    %m_dbg, %m_p0, %m_p1, %m_p2 = mem Undefined {
      depth = 8 : i64,
      name = "m",
      portNames = ["dbg", "p0", "p1", "p2"],
      readLatency = 0 : i32,
      writeLatency = 1 : i32} :
      !firrtl.probe<vector<uint, 8>>,
      !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data flip: uint>,
      !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data: uint, mask: uint<1>>,
      !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, rdata flip: uint, wmode: uint<1>, wdata: uint, wmask: uint<1>>
    %m_p0_data = subfield %m_p0[data] : !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data flip: uint>
    %m_p1_data = subfield %m_p1[data] : !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data: uint, mask: uint<1>>
    %m_p2_wdata = subfield %m_p2[wdata] : !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, rdata flip: uint, wmode: uint<1>, wdata: uint, wmask: uint<1>>
    %c0_ui5 = constant 0 : !firrtl.uint<5>
    %c0_ui7 = constant 0 : !firrtl.uint<7>
    connect %m_p1_data, %c0_ui5 : !firrtl.uint, !firrtl.uint<5>
    connect %m_p2_wdata, %c0_ui7 : !firrtl.uint, !firrtl.uint<7>
    connect %out, %m_p0_data : !firrtl.uint, !firrtl.uint
    ref.define %dbg, %m_dbg : !firrtl.probe<vector<uint, 8>>
    // CHECK:  ref.define %dbg, %m_dbg : !firrtl.probe<vector<uint<7>, 8>>
  }

  // CHECK-LABEL: @MemBundle
  // CHECK-SAME: out %out: !firrtl.bundle<a: uint<7>>
  module @MemBundle(out %out: !firrtl.bundle<a: uint>) {
    // CHECK: mem
    // CHECK-SAME: data flip: bundle<a: uint<7>>
    // CHECK-SAME: data: bundle<a: uint<7>>
    // CHECK-SAME: data: bundle<a: uint<7>>
    %m_p0, %m_p1, %m_p2 = mem Undefined {
      depth = 8 : i64,
      name = "m",
      portNames = ["p0", "p1", "p2"],
      readLatency = 0 : i32,
      writeLatency = 1 : i32} :
      !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data flip: bundle<a: uint>>,
      !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data: bundle<a: uint>, mask: bundle<a: uint<1>>>,
      !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, rdata flip: bundle<a: uint>, wmode: uint<1>, wdata: bundle<a: uint>, wmask: bundle<a: uint<1>>>
    %m_p0_data = subfield %m_p0[data] : !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data flip: bundle<a: uint>>
    %m_p1_data = subfield %m_p1[data] : !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data: bundle<a: uint>, mask: bundle<a: uint<1>>>
    %m_p2_wdata = subfield %m_p2[wdata] : !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, rdata flip: bundle<a: uint>, wmode: uint<1>, wdata: bundle<a: uint>, wmask: bundle<a: uint<1>>>
    %m_p1_data_a = subfield %m_p1_data[a] : !firrtl.bundle<a: uint>
    %m_p2_wdata_a = subfield %m_p2_wdata[a] : !firrtl.bundle<a: uint>
    %c0_ui5 = constant 0 : !firrtl.uint<5>
    %c0_ui7 = constant 0 : !firrtl.uint<7>
    connect %m_p1_data_a, %c0_ui5 : !firrtl.uint, !firrtl.uint<5>
    connect %m_p2_wdata_a, %c0_ui7 : !firrtl.uint, !firrtl.uint<7>
    connect %out, %m_p0_data : !firrtl.bundle<a: uint>, !firrtl.bundle<a: uint>
  }

  // Breakable cycles in inter-module width inference.
  // CHECK-LABEL: @InterModuleGoodCycleFoo
  // CHECK-SAME: in %in: !firrtl.uint<42>
  // CHECK-SAME: out %out: !firrtl.uint<39>
  module @InterModuleGoodCycleFoo(in %in: !firrtl.uint, out %out: !firrtl.uint) {
    %0 = shr %in, 3 : (!firrtl.uint) -> !firrtl.uint
    connect %out, %0 : !firrtl.uint, !firrtl.uint
  }
  // CHECK-LABEL: @InterModuleGoodCycleBar
  // CHECK-SAME: out %out: !firrtl.uint<39>
  module @InterModuleGoodCycleBar(in %in: !firrtl.uint<42>, out %out: !firrtl.uint) {
    %inst_in, %inst_out = instance inst  @InterModuleGoodCycleFoo(in in: !firrtl.uint, out out: !firrtl.uint)
    connect %inst_in, %in : !firrtl.uint, !firrtl.uint<42>
    connect %inst_in, %inst_out : !firrtl.uint, !firrtl.uint
    connect %out, %inst_out : !firrtl.uint, !firrtl.uint
  }

  // CHECK-LABEL: @Issue1271
  module @Issue1271(in %clock: !firrtl.clock, in %cond: !firrtl.uint<1>) {
    // CHECK: %a = reg %clock  : !firrtl.clock, !firrtl.uint<2>
    // CHECK: %b = node %0  : !firrtl.uint<3>
    // CHECK: %c = node %1  : !firrtl.uint<2>
    %a = reg %clock  : !firrtl.clock, !firrtl.uint
    %c0_ui1 = constant 0 : !firrtl.uint<1>
    %0 = add %a, %c0_ui1 : (!firrtl.uint, !firrtl.uint<1>) -> !firrtl.uint
    %b = node %0  : !firrtl.uint
    %1 = tail %b, 1 : (!firrtl.uint) -> !firrtl.uint
    %c = node %1  : !firrtl.uint
    %c0_ui2 = constant 0 : !firrtl.uint<2>
    %2 = mux(%cond, %c0_ui2, %c) : (!firrtl.uint<1>, !firrtl.uint<2>, !firrtl.uint) -> !firrtl.uint
    connect %a, %2 : !firrtl.uint, !firrtl.uint
  }

  module @Foo() {}

  // CHECK-LABEL: @SubRef
  // CHECK-SAME: out %x: !firrtl.probe<uint<2>>
  // CHECK-SAME: out %y: !firrtl.rwprobe<uint<2>>
  // CHECK-SAME: out %bov_ref: !firrtl.rwprobe<bundle<a: vector<uint<2>, 2>, b: uint<2>>>
  module private @SubRef(out %x: !firrtl.probe<uint>, out %y : !firrtl.rwprobe<uint>, out %bov_ref : !firrtl.rwprobe<bundle<a: vector<uint, 2>, b : uint>>) {
    // CHECK: wire forceable : !firrtl.uint<2>, !firrtl.rwprobe<uint<2>>
    %w, %w_rw = wire forceable : !firrtl.uint, !firrtl.rwprobe<uint>
    %bov, %bov_rw = wire forceable : !firrtl.bundle<a: vector<uint, 2>, b flip: uint>, !firrtl.rwprobe<bundle<a: vector<uint, 2>, b : uint>>
    ref.define %bov_ref, %bov_rw : !firrtl.rwprobe<bundle<a: vector<uint, 2>, b : uint>>

    %ref_w = ref.send %w : !firrtl.uint
    ref.define %x, %ref_w : !firrtl.probe<uint>
    ref.define %y, %w_rw : !firrtl.rwprobe<uint>

    %c0_ui2 = constant 0 : !firrtl.uint<2>
    connect %w, %c0_ui2 : !firrtl.uint, !firrtl.uint<2>
    
    %bov_a = subfield %bov[a] : !firrtl.bundle<a: vector<uint, 2>, b flip: uint>
    %bov_a_1 = subindex %bov_a[1] : !firrtl.vector<uint, 2>
    %bov_b = subfield %bov[b] : !firrtl.bundle<a: vector<uint, 2>, b flip: uint>

    connect %w, %c0_ui2 : !firrtl.uint, !firrtl.uint<2>
    connect %bov_a_1, %c0_ui2 : !firrtl.uint, !firrtl.uint<2>
    connect %bov_b, %c0_ui2 : !firrtl.uint, !firrtl.uint<2>
  }
  // CHECK-LABEL: @Ref
  // CHECK: out x: !firrtl.probe<uint<2>>
  // CHECK-SAME: out y: !firrtl.rwprobe<uint<2>>
  // CHECK: ref.resolve %sub_x : !firrtl.probe<uint<2>>
  // CHECK: ref.resolve %sub_y : !firrtl.rwprobe<uint<2>>
  module @Ref(out %r : !firrtl.uint, out %s : !firrtl.uint) {
    %sub_x, %sub_y, %sub_bov_ref = instance sub @SubRef(out x: !firrtl.probe<uint>, out y: !firrtl.rwprobe<uint>, out bov_ref : !firrtl.rwprobe<bundle<a: vector<uint, 2>, b : uint>>)
    %res_x = ref.resolve %sub_x : !firrtl.probe<uint>
    %res_y = ref.resolve %sub_y : !firrtl.rwprobe<uint>
    connect %r, %res_x : !firrtl.uint, !firrtl.uint
    connect %s, %res_y : !firrtl.uint, !firrtl.uint

    // CHECK: !firrtl.rwprobe<bundle<a: vector<uint<2>, 2>, b: uint<2>>>
    %read_bov = ref.resolve %sub_bov_ref : !firrtl.rwprobe<bundle<a: vector<uint, 2>, b : uint>>
    // CHECK: !firrtl.rwprobe<bundle<a: vector<uint<2>, 2>, b: uint<2>>>
    %bov_ref_a = ref.sub %sub_bov_ref[0] : !firrtl.rwprobe<bundle<a: vector<uint, 2>, b : uint>>
    // CHECK: !firrtl.rwprobe<vector<uint<2>, 2>>
    %bov_ref_a_1 = ref.sub %bov_ref_a[1] : !firrtl.rwprobe<vector<uint, 2>>
    // CHECK: !firrtl.rwprobe<bundle<a: vector<uint<2>, 2>, b: uint<2>>>
    %bov_ref_b  = ref.sub %sub_bov_ref[1] : !firrtl.rwprobe<bundle<a: vector<uint, 2>, b : uint>>

    // CHECK: !firrtl.rwprobe<vector<uint<2>, 2>>
    %bov_a = ref.resolve %bov_ref_a : !firrtl.rwprobe<vector<uint,2>>
    // CHECK: !firrtl.rwprobe<uint<2>>
    %bov_a_1 = ref.resolve %bov_ref_a_1 : !firrtl.rwprobe<uint>
    // CHECK: !firrtl.rwprobe<uint<2>>
    %bov_b = ref.resolve %bov_ref_b : !firrtl.rwprobe<uint>
  }

  // CHECK-LABEL: @ForeignTypes
  module @ForeignTypes(in %a: !firrtl.uint<42>, out %b: !firrtl.uint) {
    %0 = wire : index
    %1 = wire : index
    connect %0, %1 : index, index
    connect %b, %a : !firrtl.uint, !firrtl.uint<42>
    // CHECK-NEXT: [[W0:%.+]] = wire : index
    // CHECK-NEXT: [[W1:%.+]] = wire : index
    // CHECK-NEXT: connect [[W0]], [[W1]] : index
  }

  // CHECK-LABEL: @Issue4859
  module @Issue4859() {
    %invalid = invalidvalue : !firrtl.bundle<a: vector<uint, 2>>
    %0 = subfield %invalid[a] : !firrtl.bundle<a: vector<uint, 2>>
    %1 = subindex %0[0] : !firrtl.vector<uint, 2>
  }
  
  // CHECK-LABEL: @InferConst
  // CHECK-SAME: out %out: !firrtl.const.bundle<a: uint<1>, b: sint<2>, c: analog<3>, d: vector<uint<4>, 2>>
  module @InferConst(in %a: !firrtl.const.uint<1>, in %b: !firrtl.const.sint<2>, in %c: !firrtl.const.analog<3>, in %d: !firrtl.const.vector<uint<4>, 2>,
    out %out: !firrtl.const.bundle<a: uint, b: sint, c: analog, d: vector<uint, 2>>) {
    %0 = subfield %out[a] : !firrtl.const.bundle<a: uint, b: sint, c: analog, d: vector<uint, 2>>
    %1 = subfield %out[b] : !firrtl.const.bundle<a: uint, b: sint, c: analog, d: vector<uint, 2>>
    %2 = subfield %out[c] : !firrtl.const.bundle<a: uint, b: sint, c: analog, d: vector<uint, 2>>
    %3 = subfield %out[d] : !firrtl.const.bundle<a: uint, b: sint, c: analog, d: vector<uint, 2>>

    connect %0, %a : !firrtl.const.uint, !firrtl.const.uint<1>
    connect %1, %b : !firrtl.const.sint, !firrtl.const.sint<2>
    attach %2, %c : !firrtl.const.analog, !firrtl.const.analog<3>
    connect %3, %d : !firrtl.const.vector<uint, 2>, !firrtl.const.vector<uint<4>, 2>
  }
  
  // Should not crash when encountering property types.
  // CHECK: module @Property(in %a: !firrtl.string)
  module @Property(in %a: !firrtl.string) { }
}

// RUN: circt-opt --lower-firrtl-to-hw --verify-diagnostics %s | FileCheck %s

firrtl.circuit "Intrinsics" {
  // CHECK-LABEL: hw.module @Intrinsics
  module @Intrinsics(in %clk: !firrtl.clock, in %a: !firrtl.uint<1>) {
    // CHECK-NEXT: %x_i1 = sv.constantX : i1
    // CHECK-NEXT: [[T0:%.+]] = comb.icmp bin ceq %a, %x_i1
    // CHECK-NEXT: [[T1:%.+]] = comb.icmp bin ceq %clk, %x_i1
    // CHECK-NEXT: %x0 = hw.wire [[T0]]
    // CHECK-NEXT: %x1 = hw.wire [[T1]]
    %0 = int.isX %a : !firrtl.uint<1>
    %1 = int.isX %clk : !firrtl.clock
    %x0 = node interesting_name %0 : !firrtl.uint<1>
    %x1 = node interesting_name %1 : !firrtl.uint<1>

    // CHECK-NEXT: [[FOO_STR:%.*]] = sv.constantStr "foo"
    // CHECK-NEXT: [[FOO_DECL:%.*]] = sv.reg : !hw.inout<i1>
    // CHECK-NEXT: [[FOO:%.*]] = sv.read_inout [[FOO_DECL]]
    // CHECK-NEXT: [[BAR_STR:%.*]] = sv.constantStr "bar"
    // CHECK-NEXT: [[BAR_VALUE_DECL:%.*]] = sv.reg : !hw.inout<i5>
    // CHECK-NEXT: [[BAR_FOUND_DECL:%.*]] = sv.reg : !hw.inout<i1>
    // CHECK-NEXT: sv.initial {
    // CHECK-NEXT:   [[TMP:%.*]] = sv.system "test$plusargs"([[FOO_STR]])
    // CHECK-NEXT:   sv.passign [[FOO_DECL]], [[TMP]]
    // CHECK-NEXT:   [[TMP:%.*]] = sv.system "value$plusargs"([[BAR_STR]], [[BAR_VALUE_DECL]])
    // CHECK-NEXT:   sv.passign [[BAR_FOUND_DECL]], [[TMP]]
    // CHECK-NEXT: }
    // CHECK-NEXT: [[BAR_FOUND:%.*]] = sv.read_inout [[BAR_FOUND_DECL]]
    // CHECK-NEXT: [[BAR_VALUE:%.*]] = sv.read_inout [[BAR_VALUE_DECL]]
    // CHECK-NEXT: %x2 = hw.wire [[FOO]]
    // CHECK-NEXT: %x3 = hw.wire [[BAR_FOUND]]
    // CHECK-NEXT: %x4 = hw.wire [[BAR_VALUE]]
    %2 = int.plusargs.test "foo"
    %3, %4 = int.plusargs.value "bar" : !firrtl.uint<5>
    %x2 = node interesting_name %2 : !firrtl.uint<1>
    %x3 = node interesting_name %3 : !firrtl.uint<1>
    %x4 = node interesting_name %4 : !firrtl.uint<5>
  }

  // CHECK-LABEL: hw.module @LTLAndVerif
  module @LTLAndVerif(in %clk: !firrtl.clock, in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>) {
    // CHECK-NEXT: [[D0:%.+]] = ltl.delay %a, 42 : i1
    // CHECK-NEXT: [[D1:%.+]] = ltl.delay %b, 42, 1337 : i1
    %d0 = int.ltl.delay %a, 42 : (!firrtl.uint<1>) -> !firrtl.uint<1>
    %d1 = int.ltl.delay %b, 42, 1337 : (!firrtl.uint<1>) -> !firrtl.uint<1>

    // CHECK-NEXT: [[L0:%.+]] = ltl.and [[D0]], [[D1]] : !ltl.sequence, !ltl.sequence
    // CHECK-NEXT: [[L1:%.+]] = ltl.or %a, [[L0]] : i1, !ltl.sequence
    %l0 = int.ltl.and %d0, %d1 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    %l1 = int.ltl.or %a, %l0 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>

    // CHECK-NEXT: [[C0:%.+]] = ltl.concat [[D0]], [[L1]] : !ltl.sequence, !ltl.sequence
    %c0 = int.ltl.concat %d0, %l1 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>

    // CHECK-NEXT: [[N0:%.+]] = ltl.not [[C0]] : !ltl.sequence
    %n0 = int.ltl.not %c0 : (!firrtl.uint<1>) -> !firrtl.uint<1>

    // CHECK-NEXT: [[I0:%.+]] = ltl.implication [[C0]], [[N0]] : !ltl.sequence, !ltl.property
    %i0 = int.ltl.implication %c0, %n0 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>

    // CHECK-NEXT: [[E0:%.+]] = ltl.eventually [[I0]] : !ltl.property
    %e0 = int.ltl.eventually %i0 : (!firrtl.uint<1>) -> !firrtl.uint<1>

    // CHECK-NEXT: [[K0:%.+]] = ltl.clock [[I0]], posedge %clk : !ltl.property
    %k0 = int.ltl.clock %i0, %clk : (!firrtl.uint<1>, !firrtl.clock) -> !firrtl.uint<1>

    // CHECK-NEXT: [[D2:%.+]] = ltl.disable [[K0]] if %b : !ltl.property
    %d2 = int.ltl.disable %k0, %b : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>

    // CHECK-NEXT: verif.assert %a : i1
    // CHECK-NEXT: verif.assert %a label "hello" : i1
    // CHECK-NEXT: verif.assume [[C0]] : !ltl.sequence
    // CHECK-NEXT: verif.assume [[C0]] label "hello" : !ltl.sequence
    // CHECK-NEXT: verif.cover [[K0]] : !ltl.property
    // CHECK-NEXT: verif.cover [[K0]] label "hello" : !ltl.property
    int.verif.assert %a : !firrtl.uint<1>
    int.verif.assert %a {label = "hello"} : !firrtl.uint<1>
    int.verif.assume %c0 : !firrtl.uint<1>
    int.verif.assume %c0 {label = "hello"} : !firrtl.uint<1>
    int.verif.cover %k0 : !firrtl.uint<1>
    int.verif.cover %k0 {label = "hello"} : !firrtl.uint<1>
  }

  // CHECK-LABEL: hw.module @LowerIntrinsicStyle
  module @LowerIntrinsicStyle(in %a: !firrtl.uint<1>, in %b: !firrtl.uint<1>) {
    // Wires can make the lowering really weird. Try some strange setup where
    // the ops are totally backwards. This is tricky to lower since a lot of the
    // LTL ops' result type depends on the inputs, and LowerToHW lowers them
    // before their operands have been lowered (and have the correct LTL type).
    // CHECK-NOT: hw.wire
    %c = wire : !firrtl.uint<1>
    %d = wire : !firrtl.uint<1>
    %e = wire : !firrtl.uint<1>
    %f = wire : !firrtl.uint<1>
    %g = wire : !firrtl.uint<1>

    // CHECK-NEXT: verif.assert [[E:%.+]] : !ltl.sequence
    // CHECK-NEXT: verif.assert [[F:%.+]] : !ltl.property
    // CHECK-NEXT: verif.assert [[G:%.+]] : !ltl.property
    int.verif.assert %e : !firrtl.uint<1>
    int.verif.assert %f : !firrtl.uint<1>
    int.verif.assert %g : !firrtl.uint<1>

    // !ltl.property
    // CHECK-NEXT: [[G]] = ltl.implication [[E]], [[F]] : !ltl.sequence, !ltl.property
    %4 = int.ltl.implication %e, %f : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    strictconnect %g, %4 : !firrtl.uint<1>

    // inferred as !ltl.property
    // CHECK-NEXT: [[F]] = ltl.or %b, [[D:%.+]] : i1, !ltl.property
    %3 = int.ltl.or %b, %d : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    strictconnect %f, %3 : !firrtl.uint<1>

    // inferred as !ltl.sequence
    // CHECK-NEXT: [[E]] = ltl.and %b, [[C:%.+]] : i1, !ltl.sequence
    %2 = int.ltl.and %b, %c : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<1>
    strictconnect %e, %2 : !firrtl.uint<1>

    // !ltl.property
    // CHECK-NEXT: [[D]] = ltl.not %b : i1
    %1 = int.ltl.not %b : (!firrtl.uint<1>) -> !firrtl.uint<1>
    strictconnect %d, %1 : !firrtl.uint<1>

    // !ltl.sequence
    // CHECK-NEXT: [[C]] = ltl.delay %a, 42 : i1
    %0 = int.ltl.delay %a, 42 : (!firrtl.uint<1>) -> !firrtl.uint<1>
    strictconnect %c, %0 : !firrtl.uint<1>
  }
}

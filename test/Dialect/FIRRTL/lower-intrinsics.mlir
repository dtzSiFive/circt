// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl-lower-intrinsics))' %s   | FileCheck %s

// CHECK-LABEL: "Foo"
firrtl.circuit "Foo" {
  // CHECK-NOT: NameDoesNotMatter
  extmodule @NameDoesNotMatter(in i : !firrtl.clock, out size : !firrtl.uint<32>) attributes
                                     {annotations = [{class = "circt.Intrinsic", intrinsic = "circt.sizeof"}]}
  // CHECK-NOT: NameDoesNotMatter2
  extmodule @NameDoesNotMatter2(in i : !firrtl.clock, out found : !firrtl.uint<1>) attributes
                                     {annotations = [{class = "circt.Intrinsic", intrinsic = "circt.isX"}]}
  // CHECK-NOT: NameDoesNotMatter3
  extmodule @NameDoesNotMatter3<FORMAT: none = "foo">(out found : !firrtl.uint<1>) attributes
                                     {annotations = [{class = "circt.Intrinsic", intrinsic = "circt.plusargs.test"}]}
  // CHECK-NOT: NameDoesNotMatter4
  extmodule @NameDoesNotMatter4<FORMAT: none = "foo">(out found : !firrtl.uint<1>, out result: !firrtl.uint<5>) attributes
                                     {annotations = [{class = "circt.Intrinsic", intrinsic = "circt.plusargs.value"}]}

  // CHECK: Foo
  module @Foo(in %clk : !firrtl.clock, out %s : !firrtl.uint<32>, out %io1 : !firrtl.uint<1>, out %io2 : !firrtl.uint<1>, out %io3 : !firrtl.uint<1>, out %io4 : !firrtl.uint<5>) {
    %i1, %size = instance "" @NameDoesNotMatter(in i : !firrtl.clock, out size : !firrtl.uint<32>)
    // CHECK-NOT: NameDoesNotMatter
    // CHECK: int.sizeof
    strictconnect %i1, %clk : !firrtl.clock
    strictconnect %s, %size : !firrtl.uint<32>

    %i2, %found2 = instance "" @NameDoesNotMatter2(in i : !firrtl.clock, out found : !firrtl.uint<1>)
    // CHECK-NOT: NameDoesNotMatter2
    // CHECK: int.isX
    strictconnect %i2, %clk : !firrtl.clock
    strictconnect %io1, %found2 : !firrtl.uint<1>

    %found3 = instance "" @NameDoesNotMatter3(out found : !firrtl.uint<1>)
    // CHECK-NOT: NameDoesNotMatter3
    // CHECK: int.plusargs.test "foo"
    strictconnect %io2, %found3 : !firrtl.uint<1>

    %found4, %result1 = instance "" @NameDoesNotMatter4(out found : !firrtl.uint<1>, out result: !firrtl.uint<5>)
    // CHECK-NOT: NameDoesNotMatter4
    // CHECK: int.plusargs.value "foo" : !firrtl.uint<5>
    strictconnect %io3, %found4 : !firrtl.uint<1>
    strictconnect %io4, %result1 : !firrtl.uint<5>
  }

  // CHECK-NOT: NameDoesNotMatte5
  intmodule @NameDoesNotMatter5(in i : !firrtl.clock, out size : !firrtl.uint<32>) attributes
                                     {intrinsic = "circt.sizeof"}
  // CHECK-NOT: NameDoesNotMatter6
  intmodule @NameDoesNotMatter6(in i : !firrtl.clock, out found : !firrtl.uint<1>) attributes
                                     {intrinsic = "circt.isX"}
  // CHECK-NOT: NameDoesNotMatter7
  intmodule @NameDoesNotMatter7<FORMAT: none = "foo">(out found : !firrtl.uint<1>) attributes
                                     {intrinsic = "circt.plusargs.test"}
  // CHECK-NOT: NameDoesNotMatter8
  intmodule @NameDoesNotMatter8<FORMAT: none = "foo">(out found : !firrtl.uint<1>, out result: !firrtl.uint<5>) attributes
                                     {intrinsic = "circt.plusargs.value"}

  // CHECK: Bar
  module @Bar(in %clk : !firrtl.clock, out %s : !firrtl.uint<32>, out %io1 : !firrtl.uint<1>, out %io2 : !firrtl.uint<1>, out %io3 : !firrtl.uint<1>, out %io4 : !firrtl.uint<5>) {
    %i1, %size = instance "" @NameDoesNotMatter5(in i : !firrtl.clock, out size : !firrtl.uint<32>)
    // CHECK-NOT: NameDoesNotMatter5
    // CHECK: int.sizeof
    strictconnect %i1, %clk : !firrtl.clock
    strictconnect %s, %size : !firrtl.uint<32>

    %i2, %found2 = instance "" @NameDoesNotMatter6(in i : !firrtl.clock, out found : !firrtl.uint<1>)
    // CHECK-NOT: NameDoesNotMatter6
    // CHECK: int.isX
    strictconnect %i2, %clk : !firrtl.clock
    strictconnect %io1, %found2 : !firrtl.uint<1>

    %found3 = instance "" @NameDoesNotMatter7(out found : !firrtl.uint<1>)
    // CHECK-NOT: NameDoesNotMatter7
    // CHECK: int.plusargs.test "foo"
    strictconnect %io2, %found3 : !firrtl.uint<1>

    %found4, %result1 = instance "" @NameDoesNotMatter8(out found : !firrtl.uint<1>, out result: !firrtl.uint<5>)
    // CHECK-NOT: NameDoesNotMatter8
    // CHECK: int.plusargs.value "foo" : !firrtl.uint<5>
    strictconnect %io3, %found4 : !firrtl.uint<1>
    strictconnect %io4, %result1 : !firrtl.uint<5>
  }

  // CHECK-NOT: ClockGate0
  // CHECK-NOT: ClockGate1
  extmodule @ClockGate0(in in: !firrtl.clock, in en: !firrtl.uint<1>, out out: !firrtl.clock) attributes {annotations = [{class = "circt.Intrinsic", intrinsic = "circt.clock_gate"}]}
  intmodule @ClockGate1(in in: !firrtl.clock, in en: !firrtl.uint<1>, out out: !firrtl.clock) attributes {intrinsic = "circt.clock_gate"}

  // CHECK: ClockGate
  module @ClockGate(in %clk: !firrtl.clock, in %en: !firrtl.uint<1>) {
    // CHECK-NOT: ClockGate0
    // CHECK: int.clock_gate
    %in1, %en1, %out1 = instance "" @ClockGate0(in in: !firrtl.clock, in en: !firrtl.uint<1>, out out: !firrtl.clock)
    strictconnect %in1, %clk : !firrtl.clock
    strictconnect %en1, %en : !firrtl.uint<1>

    // CHECK-NOT: ClockGate1
    // CHECK: int.clock_gate
    %in2, %en2, %out2 = instance "" @ClockGate1(in in: !firrtl.clock, in en: !firrtl.uint<1>, out out: !firrtl.clock)
    strictconnect %in2, %clk : !firrtl.clock
    strictconnect %en2, %en : !firrtl.uint<1>
  }

  // CHECK-NOT: LTLAnd
  // CHECK-NOT: LTLOr
  // CHECK-NOT: LTLDelay1
  // CHECK-NOT: LTLDelay2
  // CHECK-NOT: LTLConcat
  // CHECK-NOT: LTLNot
  // CHECK-NOT: LTLImplication
  // CHECK-NOT: LTLEventually
  // CHECK-NOT: LTLClock
  // CHECK-NOT: LTLDisable
  intmodule @LTLAnd(in lhs: !firrtl.uint<1>, in rhs: !firrtl.uint<1>, out out: !firrtl.uint<1>) attributes {intrinsic = "circt.ltl.and"}
  intmodule @LTLOr(in lhs: !firrtl.uint<1>, in rhs: !firrtl.uint<1>, out out: !firrtl.uint<1>) attributes {intrinsic = "circt.ltl.or"}
  intmodule @LTLDelay1<delay: i64 = 42>(in in: !firrtl.uint<1>, out out: !firrtl.uint<1>) attributes {intrinsic = "circt.ltl.delay"}
  intmodule @LTLDelay2<delay: i64 = 42, length: i64 = 1337>(in in: !firrtl.uint<1>, out out: !firrtl.uint<1>) attributes {intrinsic = "circt.ltl.delay"}
  intmodule @LTLConcat(in lhs: !firrtl.uint<1>, in rhs: !firrtl.uint<1>, out out: !firrtl.uint<1>) attributes {intrinsic = "circt.ltl.concat"}
  intmodule @LTLNot(in in: !firrtl.uint<1>, out out: !firrtl.uint<1>) attributes {intrinsic = "circt.ltl.not"}
  intmodule @LTLImplication(in lhs: !firrtl.uint<1>, in rhs: !firrtl.uint<1>, out out: !firrtl.uint<1>) attributes {intrinsic = "circt.ltl.implication"}
  intmodule @LTLEventually(in in: !firrtl.uint<1>, out out: !firrtl.uint<1>) attributes {intrinsic = "circt.ltl.eventually"}
  intmodule @LTLClock(in in: !firrtl.uint<1>, in clock: !firrtl.clock, out out: !firrtl.uint<1>) attributes {intrinsic = "circt.ltl.clock"}
  intmodule @LTLDisable(in in: !firrtl.uint<1>, in condition: !firrtl.uint<1>, out out: !firrtl.uint<1>) attributes {intrinsic = "circt.ltl.disable"}

  // CHECK: module @LTL()
  module @LTL() {
    // CHECK-NOT: LTLAnd
    // CHECK-NOT: LTLOr
    // CHECK: int.ltl.and {{%.+}}, {{%.+}} :
    // CHECK: int.ltl.or {{%.+}}, {{%.+}} :
    %and.lhs, %and.rhs, %and.out = instance "and" @LTLAnd(in lhs: !firrtl.uint<1>, in rhs: !firrtl.uint<1>, out out: !firrtl.uint<1>)
    %or.lhs, %or.rhs, %or.out = instance "or" @LTLOr(in lhs: !firrtl.uint<1>, in rhs: !firrtl.uint<1>, out out: !firrtl.uint<1>)

    // CHECK-NOT: LTLDelay1
    // CHECK-NOT: LTLDelay2
    // CHECK: int.ltl.delay {{%.+}}, 42 :
    // CHECK: int.ltl.delay {{%.+}}, 42, 1337 :
    %delay1.in, %delay1.out = instance "delay1" @LTLDelay1(in in: !firrtl.uint<1>, out out: !firrtl.uint<1>)
    %delay2.in, %delay2.out = instance "delay2" @LTLDelay2(in in: !firrtl.uint<1>, out out: !firrtl.uint<1>)

    // CHECK-NOT: LTLConcat
    // CHECK-NOT: LTLNot
    // CHECK-NOT: LTLImplication
    // CHECK-NOT: LTLEventually
    // CHECK: int.ltl.concat {{%.+}}, {{%.+}} :
    // CHECK: int.ltl.not {{%.+}} :
    // CHECK: int.ltl.implication {{%.+}}, {{%.+}} :
    // CHECK: int.ltl.eventually {{%.+}} :
    %concat.lhs, %concat.rhs, %concat.out = instance "concat" @LTLConcat(in lhs: !firrtl.uint<1>, in rhs: !firrtl.uint<1>, out out: !firrtl.uint<1>)
    %not.in, %not.out = instance "not" @LTLNot(in in: !firrtl.uint<1>, out out: !firrtl.uint<1>)
    %implication.lhs, %implication.rhs, %implication.out = instance "implication" @LTLImplication(in lhs: !firrtl.uint<1>, in rhs: !firrtl.uint<1>, out out: !firrtl.uint<1>)
    %eventually.in, %eventually.out = instance "eventually" @LTLEventually(in in: !firrtl.uint<1>, out out: !firrtl.uint<1>)

    // CHECK-NOT: LTLClock
    // CHECK: int.ltl.clock {{%.+}}, {{%.+}} :
    %clock.in, %clock.clock, %clock.out = instance "clock" @LTLClock(in in: !firrtl.uint<1>, in clock: !firrtl.clock, out out: !firrtl.uint<1>)

    // CHECK-NOT: LTLDisable
    // CHECK: int.ltl.disable {{%.+}}, {{%.+}} :
    %disable.in, %disable.condition, %disable.out = instance "disable" @LTLDisable(in in: !firrtl.uint<1>, in condition: !firrtl.uint<1>, out out: !firrtl.uint<1>)
  }

  // CHECK-NOT: VerifAssert1
  // CHECK-NOT: VerifAssert2
  // CHECK-NOT: VerifAssume
  // CHECK-NOT: VerifCover
  intmodule @VerifAssert1(in property: !firrtl.uint<1>) attributes {intrinsic = "circt.verif.assert"}
  intmodule @VerifAssert2<label: none = "hello">(in property: !firrtl.uint<1>) attributes {intrinsic = "circt.verif.assert"}
  intmodule @VerifAssume(in property: !firrtl.uint<1>) attributes {intrinsic = "circt.verif.assume"}
  intmodule @VerifCover(in property: !firrtl.uint<1>) attributes {intrinsic = "circt.verif.cover"}

  // CHECK: module @Verif()
  module @Verif() {
    // CHECK-NOT: VerifAssert1
    // CHECK-NOT: VerifAssert2
    // CHECK-NOT: VerifAssume
    // CHECK-NOT: VerifCover
    // CHECK: int.verif.assert {{%.+}} :
    // CHECK: int.verif.assert {{%.+}} {label = "hello"} :
    // CHECK: int.verif.assume {{%.+}} :
    // CHECK: int.verif.cover {{%.+}} :
    %assert1.property = instance "assert1" @VerifAssert1(in property: !firrtl.uint<1>)
    %assert2.property = instance "assert2" @VerifAssert2(in property: !firrtl.uint<1>)
    %assume.property = instance "assume" @VerifAssume(in property: !firrtl.uint<1>)
    %cover.property = instance "cover" @VerifCover(in property: !firrtl.uint<1>)
  }
}

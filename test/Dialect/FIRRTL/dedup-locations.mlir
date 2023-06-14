// RUN: circt-opt -mlir-print-debuginfo -mlir-print-local-scope -pass-pipeline='builtin.module(firrtl.circuit(firrtl-dedup))' %s | FileCheck %s

firrtl.circuit "Test" {
// CHECK-LABEL: @Dedup0()
firrtl.module @Dedup0() {
  // CHECK: %w = wire  : !firrtl.uint<1> loc(fused["foo", "bar"])
  %w = wire : !firrtl.uint<1> loc("foo")
} loc("dedup0")
// CHECK: loc(fused["dedup0", "dedup1"])
// CHECK-NOT: @Dedup1()
firrtl.module @Dedup1() {
  %w = wire : !firrtl.uint<1> loc("bar")
} loc("dedup1")
firrtl.module @Test() {
  instance dedup0 @Dedup0()
  instance dedup1 @Dedup1()
}
}

// CHECK-LABEL: "PortLocations"
firrtl.circuit "PortLocations" {
// CHECK: module @PortLocs0(in %in: !firrtl.uint<1> loc(fused["1", "2"]))
firrtl.module @PortLocs0(in %in : !firrtl.uint<1> loc("1")) { }
firrtl.module @PortLocs1(in %in : !firrtl.uint<1> loc("2")) { }
firrtl.module @PortLocations() {
  instance portLocs0 @PortLocs0(in in : !firrtl.uint<1>)
  instance portLocs1 @PortLocs1(in in : !firrtl.uint<1>)
}
}

// Check that locations are limited.
// CHECK-LABEL: circuit "LimitLoc"
firrtl.circuit "LimitLoc" {
  // CHECK: module @Simple0()
  // CHECK-NEXT: loc(fused["A.fir":0:1, "A.fir":1:1, "A.fir":2:1, "A.fir":3:1, "A.fir":4:1, "A.fir":5:1, "A.fir":6:1, "A.fir":7:1])
  module @Simple0() { } loc("A.fir":0:1)
  // CHECK-NOT: @Simple1
  module @Simple1() { } loc("A.fir":1:1)
  // CHECK-NOT: @Simple2
  module @Simple2() { } loc("A.fir":2:1)
  // CHECK-NOT: @Simple3
  module @Simple3() { } loc("A.fir":3:1)
  // CHECK-NOT: @Simple4
  module @Simple4() { } loc("A.fir":4:1)
  // CHECK-NOT: @Simple5
  module @Simple5() { } loc("A.fir":5:1)
  // CHECK-NOT: @Simple6
  module @Simple6() { } loc("A.fir":6:1)
  // CHECK-NOT: @Simple7
  module @Simple7() { } loc("A.fir":7:1)
  // CHECK-NOT: @Simple8
  module @Simple8() { } loc("A.fir":8:1)
  // CHECK-NOT: @Simple9
  module @Simple9() { } loc("A.fir":9:1)
  module @LimitLoc() {
    instance simple0 @Simple0()
    instance simple1 @Simple1()
    instance simple2 @Simple2()
    instance simple3 @Simple3()
    instance simple4 @Simple4()
    instance simple5 @Simple5()
    instance simple6 @Simple6()
    instance simple7 @Simple7()
    instance simple8 @Simple8()
    instance simple9 @Simple9()
  }
}

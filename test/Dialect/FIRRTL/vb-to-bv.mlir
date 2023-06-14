// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl-vb-to-bv))' %s | FileCheck %s

firrtl.circuit "Test" {
  module @Test() {}
  //===--------------------------------------------------------------------===//
  // Port Type-Change Tests
  //===--------------------------------------------------------------------===//

  // CHECK:     @VG(in %port: !firrtl.vector<uint<8>, 4>)
  module @VG(in %port: !firrtl.vector<uint<8>, 4>) {}

  // CHECK:      BG(in %port: !firrtl.bundle<a: uint<8>>)
  module @BG(in %port: !firrtl.bundle<a: uint<8>>) {}

  // CHECK:     @VB(in %port: !firrtl.bundle<a: vector<uint<8>, 4>>)
  module @VB(in %port: !firrtl.vector<bundle<a: uint<8>>, 4>) {}

  // CHECK:     @VB2(in %port: !firrtl.bundle<a: vector<uint<8>, 4>>)
  module @VB2(in %port: !firrtl.vector<bundle<a: uint<8>>, 4>) {}

  // CHECK:     @VBB(in %port: !firrtl.bundle<nested: bundle<field: vector<uint<1>, 4>>>)
  module @VBB(in %port: !firrtl.vector<bundle<nested: bundle<field: uint<1>>>, 4>) {}

  // CHECK:     @VVB(in %port: !firrtl.bundle<field: vector<vector<uint<1>, 6>, 4>>)
  module @VVB(in %port: !firrtl.vector<vector<bundle<field: uint<1>>, 6>, 4>) {}

  // CHECK:     @VBVB(in %port: !firrtl.bundle<field_a: bundle<field_b: vector<vector<uint<1>, 4>, 8>>>)    
  module @VBVB(in %port: !firrtl.vector<bundle<field_a: vector<bundle<field_b: uint<1>>, 4>>, 8>) {}

  //===--------------------------------------------------------------------===//
  // Aggregate Create/Constant Ops
  //===--------------------------------------------------------------------===//

  // CHECK-LABEL: @TestAggregateConstants
  module @TestAggregateConstants() {
    // CHECK{LITERAL}: aggregateconstant [1, 2, 3] : !firrtl.bundle<a: uint<8>, b: uint<5>, c: uint<4>>
    aggregateconstant [1, 2, 3] : !firrtl.bundle<a: uint<8>, b: uint<5>, c: uint<4>>
    // CHECK{LITERAL}: aggregateconstant [1, 2, 3] : !firrtl.vector<uint<8>, 3>
    aggregateconstant [1, 2, 3] : !firrtl.vector<uint<8>, 3>
    // CHECK{LITERAL}: aggregateconstant [[1, 3], [2, 4]] : !firrtl.bundle<a: vector<uint<8>, 2>, b: vector<uint<5>, 2>>
    aggregateconstant [[1, 2], [3, 4]] : !firrtl.vector<bundle<a: uint<8>, b: uint<5>>, 2>
    // CHECK{LITERAL}: aggregateconstant [[[1, 3], [2, 4]], [[5, 7], [6, 8]]] : !firrtl.bundle<a: bundle<c: vector<uint<8>, 2>, d: vector<uint<5>, 2>>, b: bundle<e: vector<uint<8>, 2>, f: vector<uint<5>, 2>>>
    aggregateconstant [[[1, 2], [3, 4]], [[5, 6], [7, 8]]] : !firrtl.bundle<a: vector<bundle<c: uint<8>, d: uint<5>>, 2>, b: vector<bundle<e: uint<8>, f: uint<5>>, 2>>
    // CHECK{LITERAL}: aggregateconstant [[[1, 3], [5, 7], [9, 11]], [[2, 4], [6, 8], [10, 12]]] : !firrtl.bundle<a: vector<vector<uint<8>, 2>, 3>, b: vector<vector<uint<8>, 2>, 3>>
    aggregateconstant [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]] : !firrtl.vector<vector<bundle<a: uint<8>, b: uint<8>>, 2>, 3>
  }

  // CHECK-LABEL: @TestBundleCreate
  module @TestBundleCreate() {
    // CHECK: %0 = bundlecreate  : () -> !firrtl.bundle<>
    %be = bundlecreate : () -> !firrtl.bundle<>

    // CHECK: %c0_ui8 = constant 0 : !firrtl.uint<8>
    // CHECK: %c1_ui4 = constant 1 : !firrtl.uint<4>
    // %1 = bundlecreate %c0_ui8, %c1_ui4 : (!firrtl.uint<8>, !firrtl.uint<4>) -> !firrtl.bundle<a: uint<8>, b: uint<4>>
    %c0 = constant 0 : !firrtl.uint<8>
    %c1 = constant 1 : !firrtl.uint<4>
    %bc = bundlecreate %c0, %c1 : (!firrtl.uint<8>, !firrtl.uint<4>) -> !firrtl.bundle<a: uint<8>, b: uint<4>>

    // %2 = aggregateconstant [1, 2, 3, 4] : !firrtl.vector<uint<8>, 4>
    // %3 = aggregateconstant [5, 6] : !firrtl.vector<uint<4>, 2>
    // %4 = bundlecreate %2, %3 : (!firrtl.vector<uint<8>, 4>, !firrtl.vector<uint<4>, 2>) -> !firrtl.bundle<a: vector<uint<8>, 4>, b: vector<uint<4>, 2>>
    %v0 = aggregateconstant [1, 2, 3, 4] : !firrtl.vector<uint<8>, 4>
    %v1 = aggregateconstant [5, 6] : !firrtl.vector<uint<4>, 2>
    %bv = bundlecreate %v0, %v1 : (!firrtl.vector<uint<8>, 4>, !firrtl.vector<uint<4>, 2>) -> !firrtl.bundle<a: vector<uint<8>, 4>, b: vector<uint<4>, 2>>

    // %5 = aggregateconstant [[1, 3], [2, 4]] : !firrtl.bundle<a: vector<uint<8>, 2>, b: vector<uint<5>, 2>>
    // %6 = aggregateconstant [[1, 3], [2, 4]] : !firrtl.bundle<a: vector<uint<8>, 2>, b: vector<uint<5>, 2>>
    // %7 = bundlecreate %5, %6 : (!firrtl.bundle<a: vector<uint<8>, 2>, b: vector<uint<5>, 2>>, !firrtl.bundle<a: vector<uint<8>, 2>, b: vector<uint<5>, 2>>) -> !firrtl.bundle<a: bundle<a: vector<uint<8>, 2>, b: vector<uint<5>, 2>>, b: bundle<a: vector<uint<8>, 2>, b: vector<uint<5>, 2>>>
    %vb0 = aggregateconstant [[1, 2], [3, 4]] : !firrtl.vector<bundle<a: uint<8>, b: uint<5>>, 2>
    %vb1 = aggregateconstant [[1, 2], [3, 4]] : !firrtl.vector<bundle<a: uint<8>, b: uint<5>>, 2>
    %bvb = bundlecreate %vb0, %vb1 : (!firrtl.vector<bundle<a: uint<8>, b: uint<5>>, 2>, !firrtl.vector<bundle<a: uint<8>, b: uint<5>>, 2>) -> !firrtl.bundle<a: vector<bundle<a: uint<8>, b: uint<5>>, 2>, b: vector<bundle<a: uint<8>, b: uint<5>>, 2>>
  }

  // CHECK-LABEL: @TestVectorCreate
  module @TestVectorCreate() {
    // CHECK: %0 = vectorcreate  : () -> !firrtl.vector<uint<8>, 0>
    %v0 = vectorcreate : () -> !firrtl.vector<uint<8>, 0>

    // CHECK: %c1_ui8 = constant 1 : !firrtl.uint<8>
    // CHECK: %1 = vectorcreate %c1_ui8, %c1_ui8 : (!firrtl.uint<8>, !firrtl.uint<8>) -> !firrtl.vector<uint<8>, 2>
    %c0 = constant 1 : !firrtl.uint<8>
    %v1 = vectorcreate %c0, %c0: (!firrtl.uint<8>, !firrtl.uint<8>) -> !firrtl.vector<uint<8>, 2>

    // CHECK: %2 = bundlecreate %c1_ui8 : (!firrtl.uint<8>) -> !firrtl.bundle<a: uint<8>>
    %b0 = bundlecreate %c0 : (!firrtl.uint<8>) -> !firrtl.bundle<a: uint<8>>

    // CHECK: %3 = subfield %2[a] : !firrtl.bundle<a: uint<8>>
    // CHECK: %4 = vectorcreate %3 : (!firrtl.uint<8>) -> !firrtl.vector<uint<8>, 1>
    // CHECK: %5 = bundlecreate %4 : (!firrtl.vector<uint<8>, 1>) -> !firrtl.bundle<a: vector<uint<8>, 1>>
    %v2 = vectorcreate %b0 : (!firrtl.bundle<a: uint<8>>) -> !firrtl.vector<bundle<a: uint<8>>, 1>

    // CHECK: %6 = bundlecreate %1 : (!firrtl.vector<uint<8>, 2>) -> !firrtl.bundle<a: vector<uint<8>, 2>>
    %b1 = bundlecreate %v1 : (!firrtl.vector<uint<8>, 2>) -> !firrtl.bundle<a: vector<uint<8>, 2>>

    // CHECK: %7 = subfield %6[a] : !firrtl.bundle<a: vector<uint<8>, 2>>
    // CHECK: %8 = vectorcreate %7 : (!firrtl.vector<uint<8>, 2>) -> !firrtl.vector<vector<uint<8>, 2>, 1>
    // CHECK: %9 = bundlecreate %8 : (!firrtl.vector<vector<uint<8>, 2>, 1>) -> !firrtl.bundle<a: vector<vector<uint<8>, 2>, 1>>
    %v3 = vectorcreate %b1 : (!firrtl.bundle<a: vector<uint<8>, 2>>) -> !firrtl.vector<bundle<a: vector<uint<8>, 2>>, 1>
  }

    // CHECK-LABEL @TestVBAggregate
  module @TestVBAggregate() {
    // CHECK: %0 = aggregateconstant [1, 2] : !firrtl.bundle<a: uint<8>, b: uint<5>>
    // CHECK: %1 = subfield %0[b] : !firrtl.bundle<a: uint<8>, b: uint<5>>
    // CHECK: %2 = subfield %0[a] : !firrtl.bundle<a: uint<8>, b: uint<5>>
    %0 = aggregateconstant [1, 2] : !firrtl.bundle<a: uint<8>, b: uint<5>>

    // CHECK: %3 = aggregateconstant [3, 4] : !firrtl.bundle<a: uint<8>, b: uint<5>>
    // CHECK: %4 = subfield %3[b] : !firrtl.bundle<a: uint<8>, b: uint<5>>
    // CHECK: %5 = subfield %3[a] : !firrtl.bundle<a: uint<8>, b: uint<5>>
    %1 = aggregateconstant [3, 4] : !firrtl.bundle<a: uint<8>, b: uint<5>>

    // CHECK: %6 = vectorcreate %2, %5 : (!firrtl.uint<8>, !firrtl.uint<8>) -> !firrtl.vector<uint<8>, 2>
    // CHECK: %7 = vectorcreate %1, %4 : (!firrtl.uint<5>, !firrtl.uint<5>) -> !firrtl.vector<uint<5>, 2>
    // CHECK: %8 = bundlecreate %6, %7 : (!firrtl.vector<uint<8>, 2>, !firrtl.vector<uint<5>, 2>) -> !firrtl.bundle<a: vector<uint<8>, 2>, b: vector<uint<5>, 2>>
    %2 = vectorcreate %0, %1 : (!firrtl.bundle<a: uint<8>, b: uint<5>>, !firrtl.bundle<a: uint<8>, b: uint<5>>) -> !firrtl.vector<bundle<a: uint<8>, b: uint<5>>, 2>
  }

  module @TestVVBAggregate() {
    // CHECK{LITERAL}: %0 = aggregateconstant [[1, 3], [2, 4]] : !firrtl.bundle<a: vector<uint<8>, 2>, b: vector<uint<5>, 2>>
    // CHECK: %1 = subfield %0[b] : !firrtl.bundle<a: vector<uint<8>, 2>, b: vector<uint<5>, 2>>
    // CHECK: %2 = subfield %0[a] : !firrtl.bundle<a: vector<uint<8>, 2>, b: vector<uint<5>, 2>>
    %0 = aggregateconstant [[1, 2], [3, 4]] : !firrtl.vector<bundle<a: uint<8>, b: uint<5>>, 2>
    
    // CHECK{LITERAL}: %3 = aggregateconstant [[5, 7], [6, 8]] : !firrtl.bundle<a: vector<uint<8>, 2>, b: vector<uint<5>, 2>>
    // CHECK: %4 = subfield %3[b] : !firrtl.bundle<a: vector<uint<8>, 2>, b: vector<uint<5>, 2>>
    // CHECK: %5 = subfield %3[a] : !firrtl.bundle<a: vector<uint<8>, 2>, b: vector<uint<5>, 2>>
    %1 = aggregateconstant [[5, 6], [7, 8]] : !firrtl.vector<bundle<a: uint<8>, b: uint<5>>, 2>

    // CHECK: %6 = vectorcreate %2, %5 : (!firrtl.vector<uint<8>, 2>, !firrtl.vector<uint<8>, 2>) -> !firrtl.vector<vector<uint<8>, 2>, 2>
    // CHECK: %7 = vectorcreate %1, %4 : (!firrtl.vector<uint<5>, 2>, !firrtl.vector<uint<5>, 2>) -> !firrtl.vector<vector<uint<5>, 2>, 2>
    // CHECK: %8 = bundlecreate %6, %7 : (!firrtl.vector<vector<uint<8>, 2>, 2>, !firrtl.vector<vector<uint<5>, 2>, 2>) -> !firrtl.bundle<a: vector<vector<uint<8>, 2>, 2>, b: vector<vector<uint<5>, 2>, 2>>
    %2 = vectorcreate %0, %1 : (!firrtl.vector<bundle<a: uint<8>, b: uint<5>>, 2>, !firrtl.vector<bundle<a: uint<8>, b: uint<5>>, 2>) -> !firrtl.vector<vector<bundle<a: uint<8>, b: uint<5>>, 2>, 2>
  }

  //===--------------------------------------------------------------------===//
  // Declaration Tests
  //===--------------------------------------------------------------------===//

  // CHECK-LABEL: @TestWire
  module @TestWire() {
    // CHECK: %0 = wire : !firrtl.uint<8>
    %0 = wire : !firrtl.uint<8>
    // CHECK: %1 = wire : !firrtl.bundle<>
    %1 = wire : !firrtl.bundle<>
    // CHECK: %2 = wire : !firrtl.bundle<a: uint<8>>
    %2 = wire : !firrtl.bundle<a: uint<8>>
    // CHECK: %3 = wire : !firrtl.vector<uint<8>, 0>
    %3 = wire : !firrtl.vector<uint<8>, 0>
    // CHECK: %4 = wire : !firrtl.vector<uint<8>, 2>
    %4 = wire : !firrtl.vector<uint<8>, 2>
    // CHECK: %5 = wire : !firrtl.bundle<a: vector<uint<8>, 2>
    %5 = wire : !firrtl.bundle<a: vector<uint<8>, 2>>
    // CHECK %6 = wire : !firrtl.bundle<a: vector<uint<8>, 2>>
    %6 = wire : !firrtl.vector<bundle<a: uint<8>>, 2>
    // CHECK: %7 = wire : !firrtl.bundle<a: bundle<b: vector<uint<8>, 2>>>
    %7 = wire : !firrtl.bundle<a: vector<bundle<b: uint<8>>, 2>>
  }

  // CHECK-LABEL: @TestNode
  module @TestNode() {
    // CHECK: %w = wire : !firrtl.bundle<a: vector<uint<8>, 2>>
    %w = wire : !firrtl.vector<bundle<a: uint<8>>, 2>
    // CHECK: %n = node %w : !firrtl.bundle<a: vector<uint<8>, 2>>
    %n = node %w : !firrtl.vector<bundle<a: uint<8>>, 2>
  }

  // CHECK-LABEL @TestNodeMaterializedFromExplodedBundle
  module @TestNodeMaterializedFromExplodedBundle() {
    // CHECK: %w = wire : !firrtl.bundle<a: vector<uint<8>, 2>>
    // CHECK: %0 = subfield %w[a] : !firrtl.bundle<a: vector<uint<8>, 2>>
    // CHECK: %1 = subindex %0[0] : !firrtl.vector<uint<8>, 2>
    // CHECK: %2 = bundlecreate %1 : (!firrtl.uint<8>) -> !firrtl.bundle<a: uint<8>>
    // CHECK: %m = node %2 : !firrtl.bundle<a: uint<8>>
    %w = wire : !firrtl.vector<bundle<a: uint<8>>, 2>
    %b = subindex %w[0] : !firrtl.vector<bundle<a: uint<8>>, 2>
    %m = node %b : !firrtl.bundle<a: uint<8>>
  }
  
  // CHECK-LABEL: @TestReg
  module @TestReg(in %clock: !firrtl.clock) {
    // CHECK: %r = reg %clock : !firrtl.clock, !firrtl.bundle<a: vector<uint<8>, 2>>
    %r = reg %clock : !firrtl.clock, !firrtl.vector<bundle<a: uint<8>>, 2>
  }

  // CHECK-LABEL: @TestRegReset
  module @TestRegReset(in %clock: !firrtl.clock) {
    // CHECK{LITERAL}: %0 = aggregateconstant [[1, 3], [2, 4]] : !firrtl.bundle<a: vector<uint<4>, 2>, b: vector<uint<8>, 2>>
     %rval = aggregateconstant [[1, 2], [3, 4]] : !firrtl.vector<bundle<a: uint<4>, b: uint<8>>, 2> 
    // CHECK: %c0_ui1 = constant 0 : !firrtl.uint<1>
    %rsig = constant 0 : !firrtl.uint<1>
    // CHECK: %r = regreset %clock, %c0_ui1, %0 : !firrtl.clock, !firrtl.uint<1>, !firrtl.bundle<a: vector<uint<4>, 2>, b: vector<uint<8>, 2>>, !firrtl.bundle<a: vector<uint<4>, 2>, b: vector<uint<8>, 2>>
    %r = regreset %clock, %rsig, %rval : !firrtl.clock, !firrtl.uint<1>, !firrtl.vector<bundle<a: uint<4>, b: uint<8>>, 2>, !firrtl.vector<bundle<a: uint<4>, b: uint<8>>, 2>
  }

  // CHECK-LABEL: @TestRegResetMaterializedFromExplodedBundle
  module @TestRegResetMaterializedFromExplodedBundle(in %clock: !firrtl.clock) {
    // CHECK{LITERAL}: %0 = aggregateconstant [[1, 3], [2, 4]] : !firrtl.bundle<a: vector<uint<4>, 2>, b: vector<uint<8>, 2>>
    %storage = aggregateconstant [[1, 2], [3, 4]] : !firrtl.vector<bundle<a: uint<4>, b: uint<8>>, 2> 
    // CHECK: %1 = subfield %0[b] : !firrtl.bundle<a: vector<uint<4>, 2>, b: vector<uint<8>, 2>>
    // CHECK: %2 = subindex %1[0] : !firrtl.vector<uint<8>, 2>
    // CHECK: %3 = subfield %0[a] : !firrtl.bundle<a: vector<uint<4>, 2>, b: vector<uint<8>, 2>>
    // CHECK: %4 = subindex %3[0] : !firrtl.vector<uint<4>, 2>
    // CHECK: %5 = bundlecreate %4, %2 : (!firrtl.uint<4>, !firrtl.uint<8>) -> !firrtl.bundle<a: uint<4>, b: uint<8>>
    %rval = subindex %storage[0] : !firrtl.vector<bundle<a: uint<4>, b: uint<8>>, 2>
    // CHECK: %c0_ui1 = constant 0 : !firrtl.uint<1>
    %rsig = constant 0 : !firrtl.uint<1>
    // CHECK: %r = regreset %clock, %c0_ui1, %5 : !firrtl.clock, !firrtl.uint<1>, !firrtl.bundle<a: uint<4>, b: uint<8>>, !firrtl.bundle<a: uint<4>, b: uint<8>>
    %r = regreset %clock, %rsig, %rval : !firrtl.clock, !firrtl.uint<1>, !firrtl.bundle<a: uint<4>, b: uint<8>>, !firrtl.bundle<a: uint<4>, b: uint<8>>
  }

  // CHECK-LABEL: @TestRegResetMaterializedFromDeepExplodedBundle
  module @TestRegResetMaterializedFromDeepExplodedBundle(in %clock: !firrtl.clock) {
    // CHECK{LITERAL}: %0 = aggregateconstant [[1, 4], [[2, 3], [5, 6]]] : !firrtl.bundle<a: vector<uint<4>, 2>, b: bundle<c: vector<uint<8>, 2>, d: vector<uint<16>, 2>>>
    // CHECK: %1 = subfield %0[b] : !firrtl.bundle<a: vector<uint<4>, 2>, b: bundle<c: vector<uint<8>, 2>, d: vector<uint<16>, 2>>>
    // CHECK: %2 = subfield %1[d] : !firrtl.bundle<c: vector<uint<8>, 2>, d: vector<uint<16>, 2>>
    // CHECK: %3 = subindex %2[1] : !firrtl.vector<uint<16>, 2>
    // CHECK: %4 = subfield %1[c] : !firrtl.bundle<c: vector<uint<8>, 2>, d: vector<uint<16>, 2>>
    // CHECK: %5 = subindex %4[1] : !firrtl.vector<uint<8>, 2>
    // CHECK: %6 = subfield %0[a] : !firrtl.bundle<a: vector<uint<4>, 2>, b: bundle<c: vector<uint<8>, 2>, d: vector<uint<16>, 2>>>
    // CHECK: %7 = subindex %6[1] : !firrtl.vector<uint<4>, 2>
    // CHECK: %8 = bundlecreate %5, %3 : (!firrtl.uint<8>, !firrtl.uint<16>) -> !firrtl.bundle<c: uint<8>, d: uint<16>>
    // CHECK: %9 = bundlecreate %7, %8 : (!firrtl.uint<4>, !firrtl.bundle<c: uint<8>, d: uint<16>>) -> !firrtl.bundle<a: uint<4>, b: bundle<c: uint<8>, d: uint<16>>>
    // CHECK: %c0_ui1 = constant 0 : !firrtl.uint<1>
    // CHECK: %reg = regreset %clock, %c0_ui1, %9 : !firrtl.clock, !firrtl.uint<1>, !firrtl.bundle<a: uint<4>, b: bundle<c: uint<8>, d: uint<16>>>, !firrtl.bundle<a: uint<4>, b: bundle<c: uint<8>, d: uint<16>>>
    %reset_value_storage = aggregateconstant [[1, [2, 3]], [4, [5, 6]]] : !firrtl.vector<bundle<a: uint<4>, b: bundle<c: uint<8>, d: uint<16>>>, 2> 
    %reset_value = subindex %reset_value_storage[1] : !firrtl.vector<bundle<a: uint<4>, b: bundle<c: uint<8>, d: uint<16>>>, 2>
    %reset = constant 0 : !firrtl.uint<1>
    %reg = regreset %clock, %reset, %reset_value : !firrtl.clock, !firrtl.uint<1>, !firrtl.bundle<a: uint<4>, b: bundle<c: uint<8>, d: uint<16>>>, !firrtl.bundle<a: uint<4>, b: bundle<c: uint<8>, d: uint<16>>>
  }

  // CHECK-LABEL: @TestInstance
  module @TestInstance() {
    // CHECK: %myinst_port = instance myinst @VB(in port: !firrtl.bundle<a: vector<uint<8>, 4>>)
    %myinst_port = instance myinst @VB(in port: !firrtl.vector<bundle<a: uint<8>>, 4>)
  }

  //===--------------------------------------------------------------------===//
  // Connect Tests
  //===--------------------------------------------------------------------===//

  // CHECK-LABEL: @TestBasicConnects
  module @TestBasicConnects() {
    // CHECK: %c1_ui1 = constant 1 : !firrtl.uint<1>
    // CHECK: %w0 = wire : !firrtl.uint<1>
    // CHECK: strictconnect %w0, %c1_ui1 : !firrtl.uint<1>
    %c1 = constant 1 : !firrtl.uint<1>
    %w0 = wire : !firrtl.uint<1>
    strictconnect %w0, %c1 : !firrtl.uint<1>

    // CHECK: %w1 = wire : !firrtl.bundle<a: bundle<>>
    // CHECK: %w2 = wire : !firrtl.bundle<a: bundle<>>
    // CHECK: strictconnect %w1, %w2 : !firrtl.bundle<a: bundle<>>
    %w1 = wire : !firrtl.bundle<a: bundle<>>
    %w2 = wire : !firrtl.bundle<a: bundle<>>
    strictconnect %w1, %w2 : !firrtl.bundle<a: bundle<>>

    // CHECK: %w3 = wire : !firrtl.bundle<a flip: uint<1>>
    // CHECK: %w4 = wire : !firrtl.bundle<a flip: uint<1>>
    // CHECK: connect %w3, %w4 : !firrtl.bundle<a flip: uint<1>>
    %w3 = wire : !firrtl.bundle<a flip: uint<1>>
    %w4 = wire : !firrtl.bundle<a flip: uint<1>>
    connect %w3, %w4 : !firrtl.bundle<a flip: uint<1>>, !firrtl.bundle<a flip: uint<1>>

    // CHECK: %w5 = wire : !firrtl.bundle<a flip: uint<1>, b: uint<1>>
    // CHECK: %w6 = wire : !firrtl.bundle<a flip: uint<1>, b: uint<1>>
    // CHECK: connect %w5, %w6 : !firrtl.bundle<a flip: uint<1>, b: uint<1>>
    %w5 = wire : !firrtl.bundle<a flip: uint<1>, b: uint<1>>
    %w6 = wire : !firrtl.bundle<a flip: uint<1>, b: uint<1>>
    connect %w5, %w6 : !firrtl.bundle<a flip: uint<1>, b: uint<1>>, !firrtl.bundle<a flip: uint<1>, b: uint<1>>
  
    // CHECK: %w7 = wire : !firrtl.bundle<a: bundle<b flip: uint<8>>>
    // CHECK: %w8 = wire : !firrtl.bundle<a: bundle<b flip: uint<8>>>
    // CHECK: connect %w7, %w8 : !firrtl.bundle<a: bundle<b flip: uint<8>>>
    %w7 = wire : !firrtl.bundle<a: bundle<b flip: uint<8>>>
    %w8 = wire : !firrtl.bundle<a: bundle<b flip: uint<8>>>
    connect %w7, %w8 : !firrtl.bundle<a: bundle<b flip: uint<8>>>, !firrtl.bundle<a: bundle<b flip: uint<8>>>
  
    // Test some deeper connections.
    // (access-path caching causes subfield/subindex op movement)
  
    // CHECK: %w9 = wire : !firrtl.bundle<a flip: bundle<b flip: uint<8>>>
    // CHECK: %0 = subfield %w9[a] : !firrtl.bundle<a flip: bundle<b flip: uint<8>>>
    // CHECK: %1 = subfield %0[b] : !firrtl.bundle<b flip: uint<8>>
    %w9  = wire : !firrtl.bundle<a flip: bundle<b flip: uint<8>>>
  
    // CHECK: %w10 = wire : !firrtl.bundle<a flip: bundle<b flip: uint<8>>>
    // CHECK: %2 = subfield %w10[a] : !firrtl.bundle<a flip: bundle<b flip: uint<8>>>
    // CHECK: %3 = subfield %2[b] : !firrtl.bundle<b flip: uint<8>>
    %w10 = wire : !firrtl.bundle<a flip: bundle<b flip: uint<8>>>

    // CHECK: connect %w9, %w10 : !firrtl.bundle<a flip: bundle<b flip: uint<8>>>
    connect %w9, %w10 : !firrtl.bundle<a flip: bundle<b flip: uint<8>>>, !firrtl.bundle<a flip: bundle<b flip: uint<8>>>

    %w9_a = subfield %w9[a] : !firrtl.bundle<a flip: bundle<b flip: uint<8>>>
    %w10_a = subfield %w10[a] : !firrtl.bundle<a flip: bundle<b flip: uint<8>>>
    
    // CHECK: connect %0, %2 : !firrtl.bundle<b flip: uint<8>>
    connect %w9_a, %w10_a: !firrtl.bundle<b flip: uint<8>>, !firrtl.bundle<b flip: uint<8>>
  
    %w9_a_b = subfield %w9_a[b] : !firrtl.bundle<b flip: uint<8>>
    %w10_a_b = subfield %w10_a[b] : !firrtl.bundle<b flip: uint<8>>

    // CHECK: strictconnect %1, %3 : !firrtl.uint<8>
    strictconnect %w9_a_b, %w10_a_b : !firrtl.uint<8>
  }

  //===--------------------------------------------------------------------===//
  // Path Tests
  //===--------------------------------------------------------------------===//

  // CHECK-LABEL: @TestSubfield
  module  @TestSubfield() {
    // CHECK: %b1 = wire : !firrtl.bundle<a: uint<1>>
    // CHECK: %0 = subfield %b1[a] : !firrtl.bundle<a: uint<1>>
    // CHECK: %b2 = wire : !firrtl.bundle<b: uint<1>>
    // CHECK: %1 = subfield %b2[b] : !firrtl.bundle<b: uint<1>>
    // CHECK: strictconnect %0, %1 : !firrtl.uint<1>
    %b1 = wire : !firrtl.bundle<a: uint<1>>
    %b2 = wire : !firrtl.bundle<b: uint<1>>
    %a = subfield %b1[a] : !firrtl.bundle<a: uint<1>>
    %b = subfield %b2[b] : !firrtl.bundle<b: uint<1>>
    strictconnect %a, %b : !firrtl.uint<1>
  }

  // CHECK-LABEL: @TestSubindex
  module @TestSubindex(
    // CHECK-SAME: in %port: !firrtl.bundle<a flip: vector<uint<8>, 4>>
    in %port: !firrtl.vector<bundle<a flip: uint<8>>, 4>) {

    // Basic test that a path is rewritten following a vb->bv conversion
  
    // CHECK: %0 = subfield %port[a] : !firrtl.bundle<a flip: vector<uint<8>, 4>>
    // CHECK: %1 = subindex %0[3] : !firrtl.vector<uint<8>, 4>
    // CHECK: %c7_ui8 = constant 7 : !firrtl.uint<8>
    // CHECK: connect %1, %c7_ui8 : !firrtl.uint<8>, !firrtl.uint<8>
    %bundle = subindex %port[3] : !firrtl.vector<bundle<a flip: uint<8>>, 4>
    %field  = subfield %bundle[a] : !firrtl.bundle<a flip: uint<8>>
    %value  = constant 7 : !firrtl.uint<8>
    connect %field, %value : !firrtl.uint<8>, !firrtl.uint<8>

    // Connect two exploded bundles.
  
    // CHECK: %v1 = wire : !firrtl.bundle<a: vector<vector<uint<8>, 8>, 2>>
    // CHECK: %2 = subfield %v1[a] : !firrtl.bundle<a: vector<vector<uint<8>, 8>, 2>>
    // CHECK: %3 = subindex %2[0] : !firrtl.vector<vector<uint<8>, 8>, 2>
    %v1 = wire : !firrtl.vector<bundle<a: vector<uint<8>, 8>>, 2>
  
    // CHECK: %v2 = wire : !firrtl.bundle<a: vector<vector<uint<8>, 8>, 2>>
    // CHECK: %4 = subfield %v2[a] : !firrtl.bundle<a: vector<vector<uint<8>, 8>, 2>>
    // CHECK: %5 = subindex %4[0] : !firrtl.vector<vector<uint<8>, 8>, 2>
    %v2 = wire : !firrtl.vector<bundle<a: vector<uint<8>, 8>>, 2>
  
    %b1 = subindex %v1[0] : !firrtl.vector<bundle<a: vector<uint<8>, 8>>, 2>
    %b2 = subindex %v2[0] : !firrtl.vector<bundle<a: vector<uint<8>, 8>>, 2>
  
    // CHECK: strictconnect %3, %5 : !firrtl.vector<uint<8>, 8>
    strictconnect %b1, %b2 : !firrtl.bundle<a: vector<uint<8>, 8>>
  }

  // CHECK-LABEL: TestSubaccess
  module @TestSubaccess() {
    // CHECK: %c0_ui8 = constant 0 : !firrtl.uint<8>
    %0 = constant 0 : !firrtl.uint<8>
    // CHECK: %0 = vectorcreate %c0_ui8, %c0_ui8 : (!firrtl.uint<8>, !firrtl.uint<8>) -> !firrtl.vector<uint<8>, 2>
    %v = vectorcreate %0, %0 : (!firrtl.uint<8>, !firrtl.uint<8>) -> !firrtl.vector<uint<8>, 2>
    // CHECK: %dst = wire : !firrtl.uint<8>
    %dst = wire : !firrtl.uint<8>
    // CHECK: %1 = subaccess %0[%c0_ui8] : !firrtl.vector<uint<8>, 2>, !firrtl.uint<8>
    %src = subaccess %v[%0] : !firrtl.vector<uint<8>, 2>, !firrtl.uint<8>
    // CHECK: strictconnect %dst, %1 : !firrtl.uint<8>
    strictconnect %dst, %src : !firrtl.uint<8>
  }

  // CHECK-LABEL: TestSubaccess2
  module @TestSubaccess2() {
    // CHECK: %c0_ui8 = constant 0 : !firrtl.uint<8>
    %0 = constant 0 : !firrtl.uint<8>
    // CHECK: %0 = vectorcreate %c0_ui8, %c0_ui8 : (!firrtl.uint<8>, !firrtl.uint<8>) -> !firrtl.vector<uint<8>, 2>
    %v = vectorcreate %0, %0 : (!firrtl.uint<8>, !firrtl.uint<8>) -> !firrtl.vector<uint<8>, 2>
    // CHECK: %c0_ui8_0 = constant 0 : !firrtl.uint<8>
    %i = constant 0 : !firrtl.uint<8>
    // CHECK: %dst = wire : !firrtl.uint<8>
    %dst = wire : !firrtl.uint<8>
    // CHECK: %1 = subaccess %0[%c0_ui8_0] : !firrtl.vector<uint<8>, 2>, !firrtl.uint<8>
    %src = subaccess %v[%i] : !firrtl.vector<uint<8>, 2>, !firrtl.uint<8>
    // CHECK: strictconnect %dst, %1 : !firrtl.uint<8>
    strictconnect %dst, %src : !firrtl.uint<8>
  }

  // CHECK-LABEL: @TestPathCaching()
  module @TestPathCaching() {
    // CHECK: %w = wire : !firrtl.bundle<a: bundle<b: uint<8>>>
    // CHECK: %0 = subfield %w[a] : !firrtl.bundle<a: bundle<b: uint<8>>>
    // CHECK: %n1 = node %0 : !firrtl.bundle<b: uint<8>>
    // CHECK: %n2 = node %0 :  !firrtl.bundle<b: uint<8>>
    %w = wire : !firrtl.bundle<a: bundle<b: uint<8>>>
    %a1 = subfield %w[a] : !firrtl.bundle<a: bundle<b: uint<8>>>
    %n1 = node %a1 : !firrtl.bundle<b: uint<8>>
    %a2 = subfield %w[a] : !firrtl.bundle<a: bundle<b: uint<8>>>
    %n2 = node %a2 : !firrtl.bundle<b: uint<8>>
  }

  //===--------------------------------------------------------------------===//
  // Operand Explosion Tests
  //===--------------------------------------------------------------------===//

  // CHECK-LABEL: @TestLhsExploded
  module @TestLhsExploded() {
    // CHECK: %lhs_storage = wire : !firrtl.bundle<a: vector<uint<1>, 2>, b: bundle<c: vector<uint<1>, 2>, d: vector<uint<1>, 2>>>
    // CHECK: %0 = subfield %lhs_storage[b] : !firrtl.bundle<a: vector<uint<1>, 2>, b: bundle<c: vector<uint<1>, 2>, d: vector<uint<1>, 2>>>
    // CHECK: %1 = subfield %0[d] : !firrtl.bundle<c: vector<uint<1>, 2>, d: vector<uint<1>, 2>>
    // CHECK: %2 = subindex %1[0] : !firrtl.vector<uint<1>, 2>
    // CHECK: %3 = subfield %0[c] : !firrtl.bundle<c: vector<uint<1>, 2>, d: vector<uint<1>, 2>>
    // CHECK: %4 = subindex %3[0] : !firrtl.vector<uint<1>, 2>
    // CHECK: %5 = subfield %lhs_storage[a] : !firrtl.bundle<a: vector<uint<1>, 2>, b: bundle<c: vector<uint<1>, 2>, d: vector<uint<1>, 2>>>
    // CHECK: %6 = subindex %5[0] : !firrtl.vector<uint<1>, 2>
    // CHECK: %rhs = wire : !firrtl.bundle<a: uint<1>, b: bundle<c: uint<1>, d: uint<1>>>
    // CHECK: %7 = subfield %rhs[b] : !firrtl.bundle<a: uint<1>, b: bundle<c: uint<1>, d: uint<1>>>
    // CHECK: %8 = subfield %7[d] : !firrtl.bundle<c: uint<1>, d: uint<1>>
    // CHECK: %9 = subfield %7[c] : !firrtl.bundle<c: uint<1>, d: uint<1>>
    // CHECK: %10 = subfield %rhs[a] : !firrtl.bundle<a: uint<1>, b: bundle<c: uint<1>, d: uint<1>>>
    // CHECK: strictconnect %6, %10 : !firrtl.uint<1>
    // CHECK: strictconnect %4, %9 : !firrtl.uint<1>
    // CHECK: strictconnect %2, %8 : !firrtl.uint<1>
    %lhs_storage  = wire : !firrtl.vector<bundle<a: uint<1>, b: bundle<c: uint<1>, d: uint<1>>>, 2>
    %lhs = subindex %lhs_storage[0] : !firrtl.vector<bundle<a: uint<1>, b: bundle<c: uint<1>, d: uint<1>>>, 2>
    %rhs = wire : !firrtl.bundle<a: uint<1>, b: bundle<c: uint<1>, d: uint<1>>>
    strictconnect %lhs, %rhs : !firrtl.bundle<a: uint<1>, b: bundle<c: uint<1>, d: uint<1>>>
  }

  // CHECK-LABEL: @TestLhsExplodedWhenLhsHasFlips
  module @TestLhsExplodedWhenLhsHasFlips() {
    // CHECK: %lhs_storage = wire : !firrtl.bundle<a flip: vector<uint<1>, 2>, b: vector<uint<1>, 2>>
    // CHECK: %0 = subfield %lhs_storage[b] : !firrtl.bundle<a flip: vector<uint<1>, 2>, b: vector<uint<1>, 2>>
    // CHECK: %1 = subindex %0[0] : !firrtl.vector<uint<1>, 2>
    // CHECK: %2 = subfield %lhs_storage[a] : !firrtl.bundle<a flip: vector<uint<1>, 2>, b: vector<uint<1>, 2>>
    // CHECK: %3 = subindex %2[0] : !firrtl.vector<uint<1>, 2>
    // CHECK: %rhs = wire : !firrtl.bundle<a flip: uint<1>, b: uint<1>>
    // CHECK: %4 = subfield %rhs[b] : !firrtl.bundle<a flip: uint<1>, b: uint<1>>
    // CHECK: %5 = subfield %rhs[a] : !firrtl.bundle<a flip: uint<1>, b: uint<1>>
    // CHECK: connect %5, %3 : !firrtl.uint<1>
    // CHECK: connect %1, %4 : !firrtl.uint<1>
    %lhs_storage = wire : !firrtl.vector<bundle<a flip: uint<1>, b: uint<1>>, 2>
    %lhs = subindex %lhs_storage[0] : !firrtl.vector<bundle<a flip: uint<1>, b: uint<1>>, 2>
    %rhs = wire : !firrtl.bundle<a flip: uint<1>, b: uint<1>>
    connect %lhs, %rhs : !firrtl.bundle<a flip: uint<1>, b: uint<1>>, !firrtl.bundle<a flip: uint<1>, b: uint<1>>
  }

  // CHECK-LABEL: @TestRhsExploded
  module @TestRhsExploded() {
    // CHECK: %lhs = wire : !firrtl.bundle<a: uint<1>, b: bundle<c: uint<1>, d: uint<1>>>
    // CHECK: %rhs_storage = wire : !firrtl.bundle<a: vector<uint<1>, 2>, b: bundle<c: vector<uint<1>, 2>, d: vector<uint<1>, 2>>>
    // CHECK: %0 = subfield %rhs_storage[b] : !firrtl.bundle<a: vector<uint<1>, 2>, b: bundle<c: vector<uint<1>, 2>, d: vector<uint<1>, 2>>>
    // CHECK: %1 = subfield %0[d] : !firrtl.bundle<c: vector<uint<1>, 2>, d: vector<uint<1>, 2>>
    // CHECK: %2 = subindex %1[0] : !firrtl.vector<uint<1>, 2>
    // CHECK: %3 = subfield %0[c] : !firrtl.bundle<c: vector<uint<1>, 2>, d: vector<uint<1>, 2>>
    // CHECK: %4 = subindex %3[0] : !firrtl.vector<uint<1>, 2>
    // CHECK: %5 = subfield %rhs_storage[a] : !firrtl.bundle<a: vector<uint<1>, 2>, b: bundle<c: vector<uint<1>, 2>, d: vector<uint<1>, 2>>>
    // CHECK: %6 = subindex %5[0] : !firrtl.vector<uint<1>, 2>
    // CHECK: %7 = bundlecreate %4, %2 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.bundle<c: uint<1>, d: uint<1>>
    // CHECK: %8 = bundlecreate %6, %7 : (!firrtl.uint<1>, !firrtl.bundle<c: uint<1>, d: uint<1>>) -> !firrtl.bundle<a: uint<1>, b: bundle<c: uint<1>, d: uint<1>>>
    // CHECK: strictconnect %lhs, %8 : !firrtl.bundle<a: uint<1>, b: bundle<c: uint<1>, d: uint<1>>>
    %lhs = wire : !firrtl.bundle<a: uint<1>, b: bundle<c: uint<1>, d: uint<1>>>
    %rhs_storage = wire : !firrtl.vector<bundle<a: uint<1>, b: bundle<c: uint<1>, d: uint<1>>>, 2>
    %rhs = subindex %rhs_storage[0] : !firrtl.vector<bundle<a: uint<1>, b: bundle<c: uint<1>, d: uint<1>>>, 2>
    strictconnect %lhs, %rhs : !firrtl.bundle<a: uint<1>, b: bundle<c: uint<1>, d: uint<1>>>
  }

  // CHECK-LABEL: @TestRhsExplodedWhenLhsHasFlips
  module @TestRhsExplodedWhenLhsHasFlips() {
    // CHECK: %lhs = wire : !firrtl.bundle<a: uint<1>, b: bundle<c: uint<1>, d flip: uint<1>>>
    // CHECK: %0 = subfield %lhs[b] : !firrtl.bundle<a: uint<1>, b: bundle<c: uint<1>, d flip: uint<1>>>
    // CHECK: %1 = subfield %0[d] : !firrtl.bundle<c: uint<1>, d flip: uint<1>>
    // CHECK: %2 = subfield %0[c] : !firrtl.bundle<c: uint<1>, d flip: uint<1>>
    // CHECK: %3 = subfield %lhs[a] : !firrtl.bundle<a: uint<1>, b: bundle<c: uint<1>, d flip: uint<1>>>
    // CHECK: %rhs_storage = wire : !firrtl.bundle<a: vector<uint<1>, 2>, b: bundle<c: vector<uint<1>, 2>, d flip: vector<uint<1>, 2>>>
    // CHECK: %4 = subfield %rhs_storage[b] : !firrtl.bundle<a: vector<uint<1>, 2>, b: bundle<c: vector<uint<1>, 2>, d flip: vector<uint<1>, 2>>>
    // CHECK: %5 = subfield %4[d] : !firrtl.bundle<c: vector<uint<1>, 2>, d flip: vector<uint<1>, 2>>
    // CHECK: %6 = subindex %5[0] : !firrtl.vector<uint<1>, 2>
    // CHECK: %7 = subfield %4[c] : !firrtl.bundle<c: vector<uint<1>, 2>, d flip: vector<uint<1>, 2>>
    // CHECK: %8 = subindex %7[0] : !firrtl.vector<uint<1>, 2>
    // CHECK: %9 = subfield %rhs_storage[a] : !firrtl.bundle<a: vector<uint<1>, 2>, b: bundle<c: vector<uint<1>, 2>, d flip: vector<uint<1>, 2>>>
    // CHECK: %10 = subindex %9[0] : !firrtl.vector<uint<1>, 2>
    // CHECK: connect %3, %10 : !firrtl.uint<1>
    // CHECK: connect %2, %8 : !firrtl.uint<1>
    // CHECK: connect %6, %1 : !firrtl.uint<1>
    %lhs = wire : !firrtl.bundle<a: uint<1>, b: bundle<c: uint<1>, d flip: uint<1>>>
    %rhs_storage = wire : !firrtl.vector<bundle<a: uint<1>, b: bundle<c: uint<1>, d flip: uint<1>>>, 2>
    %rhs = subindex %rhs_storage[0] : !firrtl.vector<bundle<a: uint<1>, b: bundle<c: uint<1>, d flip: uint<1>>>, 2>
    connect %lhs, %rhs : !firrtl.bundle<a: uint<1>, b: bundle<c: uint<1>, d flip: uint<1>>>, !firrtl.bundle<a: uint<1>, b: bundle<c: uint<1>, d flip: uint<1>>>
  }

  // CHECK-LABEL: @TestBothSidesExploded
  module @TestBothSidesExploded() {
    // CHECK: %v1 = wire : !firrtl.bundle<a: vector<uint<1>, 2>, b: bundle<c: vector<uint<1>, 2>, d: vector<uint<1>, 2>>>
    // CHECK: %0 = subfield %v1[b] : !firrtl.bundle<a: vector<uint<1>, 2>, b: bundle<c: vector<uint<1>, 2>, d: vector<uint<1>, 2>>>
    // CHECK: %1 = subfield %0[d] : !firrtl.bundle<c: vector<uint<1>, 2>, d: vector<uint<1>, 2>>
    // CHECK: %2 = subindex %1[0] : !firrtl.vector<uint<1>, 2>
    // CHECK: %3 = subfield %0[c] : !firrtl.bundle<c: vector<uint<1>, 2>, d: vector<uint<1>, 2>>
    // CHECK: %4 = subindex %3[0] : !firrtl.vector<uint<1>, 2>
    // CHECK: %5 = subfield %v1[a] : !firrtl.bundle<a: vector<uint<1>, 2>, b: bundle<c: vector<uint<1>, 2>, d: vector<uint<1>, 2>>>
    // CHECK: %6 = subindex %5[0] : !firrtl.vector<uint<1>, 2>
    // CHECK: %v2 = wire : !firrtl.bundle<a: vector<uint<1>, 2>, b: bundle<c: vector<uint<1>, 2>, d: vector<uint<1>, 2>>>
    // CHECK: %7 = subfield %v2[b] : !firrtl.bundle<a: vector<uint<1>, 2>, b: bundle<c: vector<uint<1>, 2>, d: vector<uint<1>, 2>>>
    // CHECK: %8 = subfield %7[d] : !firrtl.bundle<c: vector<uint<1>, 2>, d: vector<uint<1>, 2>>
    // CHECK: %9 = subindex %8[0] : !firrtl.vector<uint<1>, 2>
    // CHECK: %10 = subfield %7[c] : !firrtl.bundle<c: vector<uint<1>, 2>, d: vector<uint<1>, 2>>
    // CHECK: %11 = subindex %10[0] : !firrtl.vector<uint<1>, 2>
    // CHECK: %12 = subfield %v2[a] : !firrtl.bundle<a: vector<uint<1>, 2>, b: bundle<c: vector<uint<1>, 2>, d: vector<uint<1>, 2>>>
    // CHECK: %13 = subindex %12[0] : !firrtl.vector<uint<1>, 2>
    // CHECK: strictconnect %6, %13 : !firrtl.uint<1>
    // CHECK: strictconnect %4, %11 : !firrtl.uint<1>
    // CHECK: strictconnect %2, %9 : !firrtl.uint<1>
    %v1 = wire : !firrtl.vector<bundle<a: uint<1>, b: bundle<c: uint<1>, d: uint<1>>>, 2>
    %v2 = wire : !firrtl.vector<bundle<a: uint<1>, b: bundle<c: uint<1>, d: uint<1>>>, 2>
    %b2 = subindex %v1[0] : !firrtl.vector<bundle<a: uint<1>, b: bundle<c: uint<1>, d: uint<1>>>, 2>
    %b3 = subindex %v2[0] : !firrtl.vector<bundle<a: uint<1>, b: bundle<c: uint<1>, d: uint<1>>>, 2>
    strictconnect %b2, %b3 : !firrtl.bundle<a: uint<1>, b: bundle<c: uint<1>, d: uint<1>>>
  }

  // module @TestExplodedNode 
  module @TestExplodedNode() {
    // CHECK: %storage = wire : !firrtl.bundle<a: vector<uint<1>, 2>, b: vector<uint<1>, 2>>
    // CHECK: %0 = subfield %storage[b] : !firrtl.bundle<a: vector<uint<1>, 2>, b: vector<uint<1>, 2>>
    // CHECK: %1 = subindex %0[0] : !firrtl.vector<uint<1>, 2>
    // CHECK: %2 = subfield %storage[a] : !firrtl.bundle<a: vector<uint<1>, 2>, b: vector<uint<1>, 2>>
    // CHECK: %3 = subindex %2[0] : !firrtl.vector<uint<1>, 2>
    // CHECK: %4 = bundlecreate %3, %1 : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.bundle<a: uint<1>, b: uint<1>>
    // CHECK: %node = node %4 : !firrtl.bundle<a: uint<1>, b: uint<1>>    
    %storage = wire : !firrtl.vector<bundle<a: uint<1>, b: uint<1>>, 2>
    %bundle = subindex %storage[0] : !firrtl.vector<bundle<a: uint<1>, b: uint<1>>, 2>
    %node = node %bundle : !firrtl.bundle<a: uint<1>, b: uint<1>>
  }

  //===--------------------------------------------------------------------===//
  // When Tests
  //===--------------------------------------------------------------------===//
  
  /// CHECK-LABEL: @TestWhen()
  module @TestWhen() {
    // CHECK: %w = wire : !firrtl.bundle<a: uint<8>>
    // CHECK: %0 = subfield %w[a] : !firrtl.bundle<a: uint<8>>
    // CHECK: %c1_ui1 = constant 1 : !firrtl.uint<1>
    // CHECK: when %c1_ui1 : !firrtl.uint<1> {
    // CHECK:   %n2 = node %0 : !firrtl.uint<8>
    // CHECK: }
    // CHECK: %n3 = node %0 : !firrtl.uint<8>
    // CHECK: when %c1_ui1 : !firrtl.uint<1> {
    // CHECK:   %w2 = wire : !firrtl.bundle<a: vector<uint<8>, 2>>
    // CHECK: }
    %w = wire : !firrtl.bundle<a: uint<8>>
    %a = subfield %w[a] : !firrtl.bundle<a: uint<8>>
    %p = constant 1 : !firrtl.uint<1>
    when %p : !firrtl.uint<1> {
      %n2 = node %a : !firrtl.uint<8>
    }
    %n3 = node %a : !firrtl.uint<8>
    when %p : !firrtl.uint<1> {
      %w2 = wire : !firrtl.vector<bundle<a: uint<8>>, 2>
    }
  }

  // CHECK-LABEL: @TestWhenWithSubaccess
  module @TestWhenWithSubaccess() {
    // CHECK: %0 = aggregateconstant [123] : !firrtl.vector<uint<8>, 1>
    // CHECK: %c0_ui8 = constant 0 : !firrtl.uint<8>
    // CHECK: %c1_ui1 = constant 1 : !firrtl.uint<1>
    // CHECK: when %c1_ui1 : !firrtl.uint<1> {
    // CHECK:   %2 = subaccess %0[%c0_ui8] : !firrtl.vector<uint<8>, 1>, !firrtl.uint<8>
    // CHECK:   %nod_0 = node %2 {name = "nod"} : !firrtl.uint<8>
    // CHECK: }
    // CHECK: %1 = subaccess %0[%c0_ui8] : !firrtl.vector<uint<8>, 1>, !firrtl.uint<8>
    // CHECK: %nod = node %1 : !firrtl.uint<8>
    %vec = aggregateconstant [123] :  !firrtl.vector<uint<8>, 1>
    %idx = constant 0 : !firrtl.uint<8>
    %cnd = constant 1 : !firrtl.uint<1>
    when %cnd : !firrtl.uint<1> {
      %val = subaccess %vec[%idx] : !firrtl.vector<uint<8>, 1>, !firrtl.uint<8>
      %nod = node %val : !firrtl.uint<8>
    }
    %val = subaccess %vec[%idx] : !firrtl.vector<uint<8>, 1>, !firrtl.uint<8>
    %nod = node %val : !firrtl.uint<8>
  }

  //===--------------------------------------------------------------------===//
  // Misc Tests
  //===--------------------------------------------------------------------===//

  // CHECK-LABEL: @TestDoubleSlicing()
  module @TestDoubleSlicing() {
    // CHECK: %w = wire : !firrtl.bundle<a: vector<vector<uint<8>, 2>, 4>, v: vector<vector<vector<uint<7>, 3>, 2>, 4>>
    %w = wire : !firrtl.vector<vector<bundle<a: uint<8>, v: vector<uint<7>, 3>>, 2>, 4>

    // CHECK: %0 = subfield %w[v] : !firrtl.bundle<a: vector<vector<uint<8>, 2>, 4>, v: vector<vector<vector<uint<7>, 3>, 2>, 4>>
    // CHECK: %1 = subindex %0[0] : !firrtl.vector<vector<vector<uint<7>, 3>, 2>, 4>
    // CHECK: %2 = subindex %1[0] : !firrtl.vector<vector<uint<7>, 3>, 2>
    // CHECK: %3 = subindex %2[2] : !firrtl.vector<uint<7>, 3>
    // CHECK: %4 = subfield %w[a] : !firrtl.bundle<a: vector<vector<uint<8>, 2>, 4>, v: vector<vector<vector<uint<7>, 3>, 2>, 4>>
    // CHECK: %5 = subindex %4[0] : !firrtl.vector<vector<uint<8>, 2>, 4>
    // CHECK: %6 = subindex %5[0] : !firrtl.vector<uint<8>, 2>

    // CHECK: %7 = bundlecreate %5, %1 : (!firrtl.vector<uint<8>, 2>, !firrtl.vector<vector<uint<7>, 3>, 2>) -> !firrtl.bundle<a: vector<uint<8>, 2>, v: vector<vector<uint<7>, 3>, 2>>
    // CHECK: %n_0 = node %7 : !firrtl.bundle<a: vector<uint<8>, 2>, v: vector<vector<uint<7>, 3>, 2>>
    %w_0 = subindex %w[0] : !firrtl.vector<vector<bundle<a: uint<8>, v: vector<uint<7>, 3>>, 2>, 4>
    %n_0 = node %w_0 : !firrtl.vector<bundle<a: uint<8>, v: vector<uint<7>, 3>>, 2>

    // CHECK: %8 = bundlecreate %6, %2 : (!firrtl.uint<8>, !firrtl.vector<uint<7>, 3>) -> !firrtl.bundle<a: uint<8>, v: vector<uint<7>, 3>>
    // CHECK: %n_0_1 = node %8 : !firrtl.bundle<a: uint<8>, v: vector<uint<7>, 3>>
    %w_0_1 = subindex %w_0[0] : !firrtl.vector<bundle<a: uint<8>, v: vector<uint<7>, 3>>, 2>
    %n_0_1 = node %w_0_1 : !firrtl.bundle<a: uint<8>, v: vector<uint<7>, 3>>

    // CHECK: %n_0_1_b = node %2 : !firrtl.vector<uint<7>, 3>
    %w_0_1_b = subfield %w_0_1[v] : !firrtl.bundle<a: uint<8>, v: vector<uint<7>, 3>>
    %n_0_1_b = node %w_0_1_b : !firrtl.vector<uint<7>, 3>
  
    // CHECK: %n_0_1_b_2 = node %3 : !firrtl.uint<7>
    %w_0_1_b_2 = subindex %w_0_1_b[2] : !firrtl.vector<uint<7>, 3>
    %n_0_1_b_2 = node %w_0_1_b_2 : !firrtl.uint<7>
  }

  // connect with flip, rhs is a rematerialized bundle, Do we preserve the flip?
  // CHECK-LABEL: @VBF
  module @VBF(
    // CHECK-SAME: in %i: !firrtl.bundle<a: vector<uint<8>, 2>, b flip: vector<uint<8>, 2>>
    in  %i : !firrtl.vector<bundle<a: uint<8>, b flip: uint<8>>, 2>,
    // CHECK-SAME: out %o: !firrtl.bundle<a: uint<8>, b flip: uint<8>>
    out %o : !firrtl.bundle<a: uint<8>, b flip: uint<8>>) {
    // CHECK: %0 = subfield %o[b] : !firrtl.bundle<a: uint<8>, b flip: uint<8>>
    // CHECK: %1 = subfield %o[a] : !firrtl.bundle<a: uint<8>, b flip: uint<8>>
    // CHECK: %2 = subfield %i[b] : !firrtl.bundle<a: vector<uint<8>, 2>, b flip: vector<uint<8>, 2>>
    // CHECK: %3 = subindex %2[0] : !firrtl.vector<uint<8>, 2>
    // CHECK: %4 = subfield %i[a] : !firrtl.bundle<a: vector<uint<8>, 2>, b flip: vector<uint<8>, 2>>
    // CHECK: %5 = subindex %4[0] : !firrtl.vector<uint<8>, 2>
    // CHECK: connect %1, %5 : !firrtl.uint<8>
    // CHECK: connect %3, %0 : !firrtl.uint<8>
    %0 = subindex %i[0] : !firrtl.vector<bundle<a: uint<8>, b flip: uint<8>>, 2>
    connect %o, %0 : !firrtl.bundle<a: uint<8>, b flip: uint<8>>, !firrtl.bundle<a: uint<8>, b flip: uint<8>>
  }

  // connect lhs is an exploded bundle with flip, Do we connect in the right direction?
  // CHECK-LABEL: VBF2
  module @VBF2(
    // CHECK-SAME: in %i: !firrtl.bundle<a: uint<8>, b flip: uint<8>>
    in  %i : !firrtl.bundle<a: uint<8>, b flip: uint<8>>,
    // CHECK-SAME: out %o: !firrtl.bundle<a: vector<uint<8>, 2>, b flip: vector<uint<8>, 2>>
    out %o : !firrtl.vector<bundle<a: uint<8>, b flip: uint<8>>, 2>) {
    // CHECK: %0 = subfield %i[b] : !firrtl.bundle<a: uint<8>, b flip: uint<8>>
    // CHECK: %1 = subfield %i[a] : !firrtl.bundle<a: uint<8>, b flip: uint<8>>
    // CHECK: %2 = subfield %o[b] : !firrtl.bundle<a: vector<uint<8>, 2>, b flip: vector<uint<8>, 2>>
    // CHECK: %3 = subindex %2[0] : !firrtl.vector<uint<8>, 2>
    // CHECK: %4 = subfield %o[a] : !firrtl.bundle<a: vector<uint<8>, 2>, b flip: vector<uint<8>, 2>>
    // CHECK: %5 = subindex %4[0] : !firrtl.vector<uint<8>, 2>
    %0 = subindex %o[0] : !firrtl.vector<bundle<a: uint<8>, b flip: uint<8>>, 2>
    // CHECK: connect %5, %1 : !firrtl.uint<8>
    // CHECK: connect %0, %3 : !firrtl.uint<8>
    connect %0, %i : !firrtl.bundle<a: uint<8>, b flip: uint<8>>, !firrtl.bundle<a: uint<8>, b flip: uint<8>>
  }

  // CHECK-LABEL: TestBundleCreate_VB
  module @TestBundleCreate_VB(out %out : !firrtl.vector<bundle<a: uint<8>, b: uint<16>>, 2>) {
    // CHECK: %c1_ui8 = constant 1 : !firrtl.uint<8>
    %0 = constant 1 : !firrtl.uint<8>
  
    // CHECK: %c2_ui16 = constant 2 : !firrtl.uint<16>
    %1 = constant 2 : !firrtl.uint<16>
  
    // CHECK: %0 = bundlecreate %c1_ui8, %c2_ui16 : (!firrtl.uint<8>, !firrtl.uint<16>) -> !firrtl.bundle<a: uint<8>, b: uint<16>>
    // CHECK: %1 = subfield %0[b] : !firrtl.bundle<a: uint<8>, b: uint<16>>
    // CHECK: %2 = subfield %0[a] : !firrtl.bundle<a: uint<8>, b: uint<16>>
    %bundle1 = bundlecreate %0, %1 : (!firrtl.uint<8>, !firrtl.uint<16>) -> !firrtl.bundle<a: uint<8>, b: uint<16>>

    // CHECK: %3 = bundlecreate %c1_ui8, %c2_ui16 : (!firrtl.uint<8>, !firrtl.uint<16>) -> !firrtl.bundle<a: uint<8>, b: uint<16>>
    // CHECK: %4 = subfield %3[b] : !firrtl.bundle<a: uint<8>, b: uint<16>>
    // CHECK: %5 = subfield %3[a] : !firrtl.bundle<a: uint<8>, b: uint<16>>
    %bundle2 = bundlecreate %0, %1 : (!firrtl.uint<8>, !firrtl.uint<16>) -> !firrtl.bundle<a: uint<8>, b: uint<16>>
    
    // CHECK: %6 = vectorcreate %2, %5 : (!firrtl.uint<8>, !firrtl.uint<8>) -> !firrtl.vector<uint<8>, 2>
    // CHECK: %7 = vectorcreate %1, %4 : (!firrtl.uint<16>, !firrtl.uint<16>) -> !firrtl.vector<uint<16>, 2>
    // CHECK: %8 = bundlecreate %6, %7 : (!firrtl.vector<uint<8>, 2>, !firrtl.vector<uint<16>, 2>) -> !firrtl.bundle<a: vector<uint<8>, 2>, b: vector<uint<16>, 2>>
    %vector  = vectorcreate %bundle1, %bundle2 : (!firrtl.bundle<a: uint<8>, b: uint<16>>, !firrtl.bundle<a: uint<8>, b: uint<16>>) -> !firrtl.vector<bundle<a: uint<8>, b: uint<16>>, 2>
    
    // CHECK connect %out, %8 : !firrtl.bundle<a: vector<uint<8>, 2>, b: vector<uint<16>, 2>>, !firrtl.bundle<a: vector<uint<8>, 2>, b: vector<uint<16>, 2>>
    connect %out, %vector : !firrtl.vector<bundle<a: uint<8>, b: uint<16>>, 2>, !firrtl.vector<bundle<a: uint<8>, b: uint<16>>, 2>
  }

  //===--------------------------------------------------------------------===//
  // Ref Type Tests
  //===--------------------------------------------------------------------===//

  // CHECK-LABEL @RefSender
  module @RefSender(out %port: !firrtl.probe<vector<bundle<a: uint<4>, b: uint<8>>, 2>>) {
   // CHECK: %w = wire : !firrtl.bundle<a: vector<uint<4>, 2>, b: vector<uint<8>, 2>>
    // CHECK: %0 = ref.send %w : !firrtl.bundle<a: vector<uint<4>, 2>, b: vector<uint<8>, 2>>
    // CHECK: ref.define %port, %0 : !firrtl.probe<bundle<a: vector<uint<4>, 2>, b: vector<uint<8>, 2>>>
    %w = wire : !firrtl.vector<bundle<a: uint<4>, b: uint<8>>, 2>
    %ref = ref.send %w : !firrtl.vector<bundle<a: uint<4>, b: uint<8>>, 2>
    ref.define %port, %ref : !firrtl.probe<vector<bundle<a: uint<4>, b: uint<8>>, 2>>
  }

  module @RefResolver() {
    // CHECK: %sender_port = instance sender @RefSender(out port: !firrtl.probe<bundle<a: vector<uint<4>, 2>, b: vector<uint<8>, 2>>>)
    // CHECK: %0 = ref.sub %sender_port[1] : !firrtl.probe<bundle<a: vector<uint<4>, 2>, b: vector<uint<8>, 2>>>
    // CHECK: %1 = ref.sub %0[1] : !firrtl.probe<vector<uint<8>, 2>>
    // CHECK: %2 = ref.sub %sender_port[0] : !firrtl.probe<bundle<a: vector<uint<4>, 2>, b: vector<uint<8>, 2>>>
    // CHECK: %3 = ref.sub %2[1] : !firrtl.probe<vector<uint<4>, 2>>
    // CHECK: %4 = ref.resolve %3 : !firrtl.probe<uint<4>>
    // CHECK: %5 = ref.resolve %1 : !firrtl.probe<uint<8>>
    // CHECK: %6 = bundlecreate %4, %5 : (!firrtl.uint<4>, !firrtl.uint<8>) -> !firrtl.bundle<a: uint<4>, b: uint<8>>
    // CHECK: %w = wire : !firrtl.bundle<a: uint<4>, b: uint<8>>
    // CHECK: strictconnect %w, %6 : !firrtl.bundle<a: uint<4>, b: uint<8>>
    %vector_ref = instance sender @RefSender(out port: !firrtl.probe<vector<bundle<a: uint<4>, b: uint<8>>, 2>>)
    %bundle_ref = ref.sub     %vector_ref[1] : !firrtl.probe<vector<bundle<a: uint<4>, b: uint<8>>, 2>>
    %bundle_val = ref.resolve %bundle_ref    : !firrtl.probe<bundle<a: uint<4>, b: uint<8>>>
    %w = wire: !firrtl.bundle<a: uint<4>, b: uint<8>>
    strictconnect %w, %bundle_val : !firrtl.bundle<a: uint<4>, b: uint<8>>
  }

  //===--------------------------------------------------------------------===//
  // Annotation Tests
  //===--------------------------------------------------------------------===//

  // CHECK-LABEL: module @Annotations(
  module @Annotations(
    // CHECK-SAME: in %in: !firrtl.bundle<f: vector<uint<8>, 4>, g: vector<uint<8>, 4>>
    in %in: !firrtl.vector<bundle<f: uint<8>, g: uint<8>>, 4> [
      // CHECK-SAME: {circt.fieldID = 2 : i64, class = "f"}
      // CHECK-SAME: {circt.fieldID = 7 : i64, class = "f"}
      {class = "f", circt.fieldID = 1 : i64}
    ]
    ) {

    // CHECK: %w = wire {
    %w = wire {
      annotations =  [
        // CHECK-SAME: {circt.fieldID = 0 : i64, class = "0"}
        {circt.fieldID = 0 : i64, class = "0"},
        // CHECK-SAME: {circt.fieldID = 2 : i64, class = "1"}
        // CHECK-SAME: {circt.fieldID = 7 : i64, class = "1"}
        {circt.fieldID = 1 : i64, class = "1"},
        // CHECK-SAME: {circt.fieldID = 2 : i64, class = "2"}
        {circt.fieldID = 2 : i64, class = "2"},
        // CHECK-SAME: {circt.fieldID = 7 : i64, class = "3"}
        {circt.fieldID = 3 : i64, class = "3"},
        // CHECK-SAME: {circt.fieldID = 3 : i64, class = "4"}
        // CHECK-SAME: {circt.fieldID = 8 : i64, class = "4"}
        {circt.fieldID = 4 : i64, class = "4"},
        // CHECK-SAME: {circt.fieldID = 3 : i64, class = "5"}
        {circt.fieldID = 5 : i64, class = "5"},
        // CHECK-SAME: {circt.fieldID = 8 : i64, class = "6"}
        {circt.fieldID = 6 : i64, class = "6"}
    ]} : 
    // CHECK-SAME: !firrtl.bundle<a: vector<uint<8>, 4>, b: vector<uint<8>, 4>>
    !firrtl.vector<bundle<a: uint<8>, b: uint<8>>, 4>
    
    // Targeting the bundle of the data field should explode and retarget to the
    // first element of the field vector.
    // CHECK: mem
    // CHECK-SAME{LITERAL}: portAnnotations = [[{circt.fieldID = 6 : i64, class = "mem0"}]]
    %bar_r = mem Undefined  {depth = 16 : i64, name = "bar", portAnnotations = [[{circt.fieldID = 5 : i64, class = "mem0"}]], portNames = ["r"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: vector<bundle<a: uint<8>>, 5>> 
  }
}

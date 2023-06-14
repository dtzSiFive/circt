// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl-inliner))' -allow-unregistered-dialect %s | FileCheck %s

// Test that an external module as the main module works.
firrtl.circuit "main_extmodule" {
  extmodule @main_extmodule()
  module private @unused () { }
}
// CHECK-LABEL: circuit "main_extmodule" {
// CHECK-NEXT:   extmodule @main_extmodule()
// CHECK-NEXT: }

// Test that unused modules are deleted.
firrtl.circuit "delete_dead_modules" {
firrtl.module @delete_dead_modules () {
  instance used @used()
  instance used @used_ext()
}
firrtl.module private @unused () { }
firrtl.module private @used () { }
firrtl.extmodule private @unused_ext ()
firrtl.extmodule private @used_ext ()
}
// CHECK-LABEL: circuit "delete_dead_modules" {
// CHECK-NEXT:   module @delete_dead_modules() {
// CHECK-NEXT:     instance used @used()
// CHECK-NEXT:     instance used @used_ext
// CHECK-NEXT:   }
// CHECK-NEXT:   module private @used() {
// CHECK-NEXT:   }
// CHECK-NEXT:   extmodule private @used_ext()
// CHECK-NEXT: }


// Test basic inlining
firrtl.circuit "inlining" {
firrtl.module @inlining() {
  instance test1 @test1()
}
firrtl.module private @test1()
  attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
  %test_wire = wire : !firrtl.uint<2>
  instance test2 @test2()
}
firrtl.module private @test2()
  attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
  %test_wire = wire : !firrtl.uint<2>
}
}
// CHECK-LABEL: circuit "inlining" {
// CHECK-NEXT:   module @inlining() {
// CHECK-NEXT:     %test1_test_wire = wire : !firrtl.uint<2>
// CHECK-NEXT:     %test1_test2_test_wire = wire : !firrtl.uint<2>
// CHECK-NEXT:   }
// CHECK-NEXT: }


// Test basic flattening:
//   1. All instances under the flattened module are inlined.
//   2. The flatten annotation is removed.
firrtl.circuit "flattening" {
firrtl.module @flattening()
  attributes {annotations = [{class = "firrtl.transforms.FlattenAnnotation"}]} {
  instance test1 @test1()
}
firrtl.module private @test1() {
  %test_wire = wire : !firrtl.uint<2>
  instance test2 @test2()
}
firrtl.module private @test2() {
  %test_wire = wire : !firrtl.uint<2>
}
}
// CHECK-LABEL: circuit "flattening"
// CHECK-NEXT:   module @flattening()
// CHECK-NOT:      annotations
// CHECK-NEXT:     %test1_test_wire = wire : !firrtl.uint<2>
// CHECK-NEXT:     %test1_test2_test_wire = wire : !firrtl.uint<2>
// CHECK-NOT:    module private @test1
// CHECK-NOT:    module private @test2


// Test that inlining and flattening compose well.
firrtl.circuit "compose" {
firrtl.module @compose() {
  instance test1 @test1()
  instance test2 @test2()
  instance test3 @test3()
}
firrtl.module private @test1() attributes {annotations =
        [{class = "firrtl.transforms.FlattenAnnotation"},
         {class = "firrtl.passes.InlineAnnotation"}]} {
  %test_wire = wire : !firrtl.uint<2>
  instance test2 @test2()
  instance test3 @test3()
}
firrtl.module private @test2() attributes {annotations =
        [{class = "firrtl.passes.InlineAnnotation"}]} {
  %test_wire = wire : !firrtl.uint<2>
  instance test3 @test3()
}
firrtl.module private @test3() {
  %test_wire = wire : !firrtl.uint<2>
}
}
// CHECK-LABEL: circuit "compose" {
// CHECK-NEXT:   module @compose() {
// CHECK-NEXT:     %test1_test_wire = wire  : !firrtl.uint<2>
// CHECK-NEXT:     %test1_test2_test_wire = wire  : !firrtl.uint<2>
// CHECK-NEXT:     %test1_test2_test3_test_wire = wire  : !firrtl.uint<2>
// CHECK-NEXT:     %test1_test3_test_wire = wire  : !firrtl.uint<2>
// CHECK-NEXT:     %test2_test_wire = wire  : !firrtl.uint<2>
// CHECK-NEXT:     instance test2_test3 @test3()
// CHECK-NEXT:     instance test3 @test3()
// CHECK-NEXT:   }
// CHECK-NEXT:   module private @test3() {
// CHECK-NEXT:     %test_wire = wire  : !firrtl.uint<2>
// CHECK-NEXT:   }
// CHECK-NEXT: }

// Test behavior inlining a flattened module into multiple parents
firrtl.circuit "TestInliningFlatten" {
firrtl.module @TestInliningFlatten() {
  instance test1 @test1()
  instance test2 @test2()
}
firrtl.module private @test1() {
  %test_wire = wire : !firrtl.uint<2>
  instance fi @flatinline()
}
firrtl.module private @test2() {
  %test_wire = wire : !firrtl.uint<2>
  instance fi @flatinline()
}
firrtl.module private @flatinline() attributes {annotations =
        [{class = "firrtl.transforms.FlattenAnnotation"},
         {class = "firrtl.passes.InlineAnnotation"}]} {
  %test_wire = wire : !firrtl.uint<2>
  instance leaf @leaf()
}
firrtl.module private @leaf() {
  %test_wire = wire : !firrtl.uint<2>
}
}
// CHECK-LABEL: circuit "TestInliningFlatten"
// CHECK-NEXT:    module @TestInliningFlatten
// inlining a flattened module should not contain 'instance's:
// CHECK:       module private @test1()
// CHECK-NOT:     instance
// inlining a flattened module should not contain 'instance's:
// CHECK:       module private @test2()
// CHECK-NOT:     instance
// These should be removed
// CHECK-NOT:   @flatinline
// CHECK-NOT:   @leaf

// Test behavior retaining public modules but not their annotations
firrtl.circuit "TestPubAnno" {
firrtl.module @TestPubAnno() {
  instance fi @flatinline()
}
firrtl.module @flatinline() attributes {annotations =
        [{class = "firrtl.transforms.FlattenAnnotation"},
         {class = "firrtl.passes.InlineAnnotation"}]} {
  %test_wire = wire : !firrtl.uint<2>
  instance leaf @leaf()
}
firrtl.module private @leaf() {
  %test_wire = wire : !firrtl.uint<2>
}
}
// CHECK-LABEL: circuit "TestPubAnno"
// CHECK-NEXT:    module @TestPubAnno
// CHECK-NOT: annotation
// This is preserved, public
// CHECK:         module @flatinline
// CHECK-NOT: annotation
// CHECK-NOT: @leaf

// This is testing that connects are properly replaced when inlining. This is
// also testing that the deep clone and remapping values is working correctly.
firrtl.circuit "TestConnections" {
firrtl.module @InlineMe0(in %in0: !firrtl.uint<4>, in %in1: !firrtl.uint<4>,
                         out %out0: !firrtl.uint<4>, out %out1: !firrtl.uint<4>)
        attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
  %0 = and %in0, %in1 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  connect %out0, %0 : !firrtl.uint<4>, !firrtl.uint<4>
  %1 = and %in0, %in1 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
  connect %out1, %1 : !firrtl.uint<4>, !firrtl.uint<4>
}
firrtl.module @InlineMe1(in %in0: !firrtl.uint<4>, in %in1: !firrtl.uint<4>,
                   out %out0: !firrtl.uint<4>, out %out1: !firrtl.uint<4>)
        attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
  %a_in0, %a_in1, %a_out0, %a_out1 = instance a @InlineMe0(in in0: !firrtl.uint<4>, in in1: !firrtl.uint<4>, out out0: !firrtl.uint<4>, out out1: !firrtl.uint<4>)
  connect %a_in0, %in0 : !firrtl.uint<4>, !firrtl.uint<4>
  connect %a_in1, %in1 : !firrtl.uint<4>, !firrtl.uint<4>
  connect %out0, %a_out0 : !firrtl.uint<4>, !firrtl.uint<4>
  connect %out1, %a_out1 : !firrtl.uint<4>, !firrtl.uint<4>
}
firrtl.module @TestConnections(in %in0: !firrtl.uint<4>, in %in1: !firrtl.uint<4>,
                   out %out0: !firrtl.uint<4>, out %out1: !firrtl.uint<4>) {
  %b_in0, %b_in1, %b_out0, %b_out1 = instance b @InlineMe1(in in0: !firrtl.uint<4>, in in1: !firrtl.uint<4>, out out0: !firrtl.uint<4>, out out1: !firrtl.uint<4>)
  connect %b_in0, %in0 : !firrtl.uint<4>, !firrtl.uint<4>
  connect %b_in1, %in1 : !firrtl.uint<4>, !firrtl.uint<4>
  connect %out0, %b_out0 : !firrtl.uint<4>, !firrtl.uint<4>
  connect %out1, %b_out1 : !firrtl.uint<4>, !firrtl.uint<4>
}
}
// CHECK-LABEL: module @TestConnections(in %in0: !firrtl.uint<4>, in %in1: !firrtl.uint<4>, out %out0: !firrtl.uint<4>, out %out1: !firrtl.uint<4>) {
// CHECK-NEXT:   %b_in0 = wire  : !firrtl.uint<4>
// CHECK-NEXT:   %b_in1 = wire  : !firrtl.uint<4>
// CHECK-NEXT:   %b_out0 = wire  : !firrtl.uint<4>
// CHECK-NEXT:   %b_out1 = wire  : !firrtl.uint<4>
// CHECK-NEXT:   %b_a_in0 = wire  : !firrtl.uint<4>
// CHECK-NEXT:   %b_a_in1 = wire  : !firrtl.uint<4>
// CHECK-NEXT:   %b_a_out0 = wire  : !firrtl.uint<4>
// CHECK-NEXT:   %b_a_out1 = wire  : !firrtl.uint<4>
// CHECK-NEXT:   %0 = and %b_a_in0, %b_a_in1 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
// CHECK-NEXT:   connect %b_a_out0, %0 : !firrtl.uint<4>, !firrtl.uint<4>
// CHECK-NEXT:   %1 = and %b_a_in0, %b_a_in1 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<4>
// CHECK-NEXT:   connect %b_a_out1, %1 : !firrtl.uint<4>, !firrtl.uint<4>
// CHECK-NEXT:   connect %b_a_in0, %b_in0 : !firrtl.uint<4>, !firrtl.uint<4>
// CHECK-NEXT:   connect %b_a_in1, %b_in1 : !firrtl.uint<4>, !firrtl.uint<4>
// CHECK-NEXT:   connect %b_out0, %b_a_out0 : !firrtl.uint<4>, !firrtl.uint<4>
// CHECK-NEXT:   connect %b_out1, %b_a_out1 : !firrtl.uint<4>, !firrtl.uint<4>
// CHECK-NEXT:   connect %b_in0, %in0 : !firrtl.uint<4>, !firrtl.uint<4>
// CHECK-NEXT:   connect %b_in1, %in1 : !firrtl.uint<4>, !firrtl.uint<4>
// CHECK-NEXT:   connect %out0, %b_out0 : !firrtl.uint<4>, !firrtl.uint<4>
// CHECK-NEXT:   connect %out1, %b_out1 : !firrtl.uint<4>, !firrtl.uint<4>
// CHECK-NEXT: }


// This is testing that bundles with flip types are handled properly by the inliner.
firrtl.circuit "TestBulkConnections" {
firrtl.module @InlineMe0(in %in0: !firrtl.bundle<a: uint<4>, b flip: uint<4>>,
                         out %out0: !firrtl.bundle<a: uint<4>, b flip: uint<4>>)
        attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
  connect %out0, %in0 : !firrtl.bundle<a: uint<4>, b flip: uint<4>>, !firrtl.bundle<a: uint<4>, b flip: uint<4>>
}
firrtl.module @TestBulkConnections(in %in0: !firrtl.bundle<a: uint<4>, b flip: uint<4>>,
                                   out %out0: !firrtl.bundle<a: uint<4>, b flip: uint<4>>) {
  %i_in0, %i_out0 = instance i @InlineMe0(in in0: !firrtl.bundle<a: uint<4>, b flip: uint<4>>, out out0: !firrtl.bundle<a: uint<4>, b flip: uint<4>>)
  connect %i_in0, %in0 : !firrtl.bundle<a: uint<4>, b flip: uint<4>>, !firrtl.bundle<a: uint<4>, b flip: uint<4>>
  connect %out0, %i_out0 : !firrtl.bundle<a: uint<4>, b flip: uint<4>>, !firrtl.bundle<a: uint<4>, b flip: uint<4>>
// CHECK: %i_in0 = wire  : !firrtl.bundle<a: uint<4>, b flip: uint<4>>
// CHECK: %i_out0 = wire  : !firrtl.bundle<a: uint<4>, b flip: uint<4>>
// CHECK: connect %i_out0, %i_in0 : !firrtl.bundle<a: uint<4>, b flip: uint<4>>, !firrtl.bundle<a: uint<4>, b flip: uint<4>>
// CHECK: connect %i_in0, %in0 : !firrtl.bundle<a: uint<4>, b flip: uint<4>>, !firrtl.bundle<a: uint<4>, b flip: uint<4>>
// CHECK: connect %out0, %i_out0 : !firrtl.bundle<a: uint<4>, b flip: uint<4>>, !firrtl.bundle<a: uint<4>, b flip: uint<4>>
}
}

// Test that all operations with names are renamed.
firrtl.circuit "renaming" {
firrtl.module @renaming() {
  %0, %1, %2 = instance myinst @declarations(in clock : !firrtl.clock, in u8 : !firrtl.uint<8>, in reset : !firrtl.asyncreset)
}
firrtl.module @declarations(in %clock : !firrtl.clock, in %u8 : !firrtl.uint<8>, in %reset : !firrtl.asyncreset) attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
  %c0_ui8 = constant 0 : !firrtl.uint<8>
  // CHECK: %cmem = chirrtl.combmem : !chirrtl.cmemory<uint<8>, 8>
  %cmem = chirrtl.combmem : !chirrtl.cmemory<uint<8>, 8>
  // CHECK: %mem_read = mem Undefined {depth = 1 : i64, name = "mem", portNames = ["read"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: sint<42>>
  %mem_read = mem Undefined {depth = 1 : i64, name = "mem", portNames = ["read"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: sint<42>>
  // CHECK: %memoryport_data, %memoryport_port = chirrtl.memoryport Read %cmem {name = "memoryport"} : (!chirrtl.cmemory<uint<8>, 8>) -> (!firrtl.uint<8>, !chirrtl.cmemoryport)
  %memoryport_data, %memoryport_port = chirrtl.memoryport Read %cmem {name = "memoryport"} : (!chirrtl.cmemory<uint<8>, 8>) -> (!firrtl.uint<8>, !chirrtl.cmemoryport)
  chirrtl.memoryport.access %memoryport_port[%u8], %clock : !chirrtl.cmemoryport, !firrtl.uint<8>, !firrtl.clock
  // CHECK: %myinst_node = node %myinst_u8  : !firrtl.uint<8>
  %node = node %u8 {name = "node"} : !firrtl.uint<8>
  // CHECK: %myinst_reg = reg %myinst_clock : !firrtl.clock, !firrtl.uint<8>
  %reg = reg %clock {name = "reg"} : !firrtl.clock, !firrtl.uint<8>
  // CHECK: %myinst_regreset = regreset %myinst_clock, %myinst_reset, %c0_ui8 : !firrtl.clock, !firrtl.asyncreset, !firrtl.uint<8>, !firrtl.uint<8>
  %regreset = regreset %clock, %reset, %c0_ui8 : !firrtl.clock, !firrtl.asyncreset, !firrtl.uint<8>, !firrtl.uint<8>
  // CHECK: %smem = chirrtl.seqmem Undefined : !chirrtl.cmemory<uint<8>, 8>
  %smem = chirrtl.seqmem Undefined : !chirrtl.cmemory<uint<8>, 8>
  // CHECK: %myinst_wire = wire  : !firrtl.uint<1>
  %wire = wire : !firrtl.uint<1>
  when %wire : !firrtl.uint<1> {
    // CHECK:  %myinst_inwhen = wire  : !firrtl.uint<1>
    %inwhen = wire : !firrtl.uint<1>
  }
}

// Test that non-module operations should not be deleted.
firrtl.circuit "PreserveUnknownOps" {
firrtl.module @PreserveUnknownOps() { }
// CHECK: sv.verbatim "hello"
sv.verbatim "hello"
}

}

// Test NLA handling during inlining for situations involving NLAs where the NLA
// begins at the main module.  There are four behaviors being tested:
//
//   1) @nla1: Targeting a module should be updated
//   2) @nla2: Targeting a component should be updated
//   3) @nla3: Targeting a module port should be updated
//   4) @nla4: Targeting an inlined module should be dropped
//   5) @nla5: NLAs targeting a component should promote to local annotations
//   6) @nla5: NLAs targeting a port should promote to local annotations
//
// CHECK-LABEL: circuit "NLAInlining"
firrtl.circuit "NLAInlining" {
  // CHECK-NEXT: hw.hierpath private @nla1 [@NLAInlining::@bar, @Bar]
  // CHECK-NEXT: hw.hierpath private @nla2 [@NLAInlining::@bar, @Bar::@a]
  // CHECK-NEXT: hw.hierpath private @nla3 [@NLAInlining::@bar, @Bar::@port]
  // CHECK-NOT:  hw.hierpath private @nla4
  // CHECK-NOT:  hw.hierpath private @nla5
  hw.hierpath private @nla1 [@NLAInlining::@foo, @Foo::@bar, @Bar]
  hw.hierpath private @nla2 [@NLAInlining::@foo, @Foo::@bar, @Bar::@a]
  hw.hierpath private @nla3 [@NLAInlining::@foo, @Foo::@bar, @Bar::@port]
  hw.hierpath private @nla4 [@NLAInlining::@foo, @Foo]
  hw.hierpath private @nla5 [@NLAInlining::@foo, @Foo::@b]
  hw.hierpath private @nla6 [@NLAInlining::@foo, @Foo::@port]
  // CHECK-NEXT: module private @Bar
  // CHECK-SAME: %port: {{.+}} sym @port [{circt.nonlocal = @nla3, class = "nla3"}]
  // CHECK-SAME: [{circt.nonlocal = @nla1, class = "nla1"}]
  module private @Bar(
    in %port: !firrtl.uint<1> sym @port [{circt.nonlocal = @nla3, class = "nla3"}]
  ) attributes {annotations = [{circt.nonlocal = @nla1, class = "nla1"}]} {
    %a = wire sym @a {annotations = [{circt.nonlocal = @nla2, class = "nla2"}]} : !firrtl.uint<1>
  }
  module private @Foo(in %port: !firrtl.uint<1> sym @port [{circt.nonlocal = @nla6, class = "nla6"}]) attributes {annotations = [
  {class = "firrtl.passes.InlineAnnotation"}, {circt.nonlocal = @nla4, class = "nla4"}]} {
    %bar_port = instance bar sym @bar @Bar(in port: !firrtl.uint<1>)
    %b = wire sym @b {annotations = [{circt.nonlocal = @nla5, class = "nla5"}]} : !firrtl.uint<1>
  }
  // CHECK: module @NLAInlining
  module @NLAInlining() {
    %foo_port = instance foo sym @foo @Foo(in port: !firrtl.uint<1>)
    // CHECK-NEXT: %foo_port = wire {{.+}} [{class = "nla6"}]
    // CHECK-NEXT: instance foo_bar {{.+}}
    // CHECK-NEXT: %foo_b = wire {{.+}} [{class = "nla5"}]
  }
}

// Test NLA handling during inlining for situations where the NLA does NOT start
// at the root.  This checks that the NLA, on either a component or a port, is
// properly copied for each new instantiation.
//
// CHECK-LABEL: circuit "NLAInliningNotMainRoot"
firrtl.circuit "NLAInliningNotMainRoot" {
  // CHECK-NEXT: hw.hierpath private @nla1 [@NLAInliningNotMainRoot::@baz, @Baz::@a]
  // CHECK-NEXT: hw.hierpath private @nla1_0 [@Foo::@baz, @Baz::@a]
  // CHECK-NEXT: hw.hierpath private @nla2 [@NLAInliningNotMainRoot::@baz, @Baz::@port]
  // CHECK-NEXT: hw.hierpath private @nla2_0 [@Foo::@baz, @Baz::@port]
  hw.hierpath private @nla1 [@Bar::@baz, @Baz::@a]
  hw.hierpath private @nla2 [@Bar::@baz, @Baz::@port]
  // CHECK: module private @Baz
  // CHECK-SAME: %port: {{.+}} [{circt.nonlocal = @nla2, class = "nla2"}, {circt.nonlocal = @nla2_0, class = "nla2"}]
  module private @Baz(
    in %port: !firrtl.uint<1> sym @port [{circt.nonlocal = @nla2, class = "nla2"}]
  ) {
    // CHECK-NEXT: wire {{.+}} [{circt.nonlocal = @nla1, class = "hello"}, {circt.nonlocal = @nla1_0, class = "hello"}]
    %a = wire sym @a {annotations = [{circt.nonlocal = @nla1, class = "hello"}]} : !firrtl.uint<1>
  }
  module private @Bar() attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
    %baz_port = instance baz sym @baz @Baz(in port: !firrtl.uint<1>)
  }
  // CHECK: module private @Foo
  module private @Foo() {
    // CHECK-NEXT: instance bar_baz {{.+}}
    instance bar @Bar()
  }
  // CHECK: module @NLAInliningNotMainRoot
  module @NLAInliningNotMainRoot() {
    instance foo @Foo()
    // CHECK: instance bar_baz {{.+}}
    instance bar @Bar()
    %baz_port = instance baz @Baz(in port: !firrtl.uint<1>)
  }
}

// Test NLA handling during flattening for situations where the root of an NLA
// is the flattened module or an ancestor of the flattened module.  This is
// testing the following conditions:
//
//   1) @nla1: Targeting a reference should be updated.
//   2) @nla1: Targeting a port should be updated.
//   3) @nla3: Targeting a module should be dropped.
//   4) @nla4: Targeting a reference should be promoted to local.
//
// CHECK-LABEL: circuit "NLAFlattening"
firrtl.circuit "NLAFlattening" {
  // CHECK-NEXT: hw.hierpath private @nla1 [@NLAFlattening::@foo, @Foo::@a]
  // CHECK-NEXT: hw.hierpath private @nla2 [@NLAFlattening::@foo, @Foo::@port]
  // CHECK-NOT:  hw.hierpath private @nla3
  // CHECK-NOT:  hw.hierpath private @nla4
  hw.hierpath private @nla1 [@NLAFlattening::@foo, @Foo::@bar, @Bar::@baz, @Baz::@a]
  hw.hierpath private @nla2 [@NLAFlattening::@foo, @Foo::@bar, @Bar::@baz, @Baz::@port]
  hw.hierpath private @nla3 [@NLAFlattening::@foo, @Foo::@bar, @Bar::@baz, @Baz]
  hw.hierpath private @nla4 [@Foo::@bar, @Bar::@b]
  module @Baz(
    in %port: !firrtl.uint<1> sym @port [{circt.nonlocal = @nla2, class = "nla2"}]
  ) attributes {annotations = [{circt.nonlocal = @nla3, class = "nla3"}]} {
    %a = wire sym @a {annotations = [{circt.nonlocal = @nla1, class = "nla1"}]} : !firrtl.uint<1>
  }
  module @Bar() {
    instance baz sym @baz @Baz(in port: !firrtl.uint<1>)
    %b = wire sym @b {annotations = [{circt.nonlocal = @nla4, class = "nla4"}]} : !firrtl.uint<1>
  }
  // CHECK: module @Foo
  module @Foo() attributes {annotations = [{class = "firrtl.transforms.FlattenAnnotation"}]} {
    instance bar sym @bar @Bar()
    // CHECK-NEXT: %bar_baz_port = wire sym @port {{.+}} [{circt.nonlocal = @nla2, class = "nla2"}]
    // CHECK-NEXT: %bar_baz_a = wire {{.+}} [{circt.nonlocal = @nla1, class = "nla1"}]
    // CHECK-NEXT: %bar_b = wire {{.+}} [{class = "nla4"}]
  }
  // CHECK: module @NLAFlattening
  module @NLAFlattening() {
    // CHECK-NEXT: instance foo {{.+}}
    instance foo sym @foo @Foo()
  }
}

// Test NLA handling during flattening for situations where the NLA root is a
// child of the flattened module.  This is testing the following situations:
//
//   1) @nla1: NLA component is made local and garbage collected.
//   2) @nla2: NLA port is made local and garbage collected.
//   3) @nla3: NLA component is made local, but not garbage collected.
//   4) @nla4: NLA port is made local, but not garbage collected.
//
// CHECK-LABEL: circuit "NLAFlatteningChildRoot"
firrtl.circuit "NLAFlatteningChildRoot" {
  // CHECK-NOT:  hw.hierpath private @nla1
  // CHECK-NOT:  hw.hierpath private @nla2
  // CHECK-NEXT: hw.hierpath private @nla3 [@Baz::@quz, @Quz::@b]
  // CHECK-NEXT: hw.hierpath private @nla4 [@Baz::@quz, @Quz::@Quz_port]
  hw.hierpath private @nla1 [@Bar::@qux, @Qux::@a]
  hw.hierpath private @nla2 [@Bar::@qux, @Qux::@Qux_port]
  hw.hierpath private @nla3 [@Baz::@quz, @Quz::@b]
  hw.hierpath private @nla4 [@Baz::@quz, @Quz::@Quz_port]
  // CHECK: module private @Quz
  // CHECK-SAME: in %port: {{.+}} [{circt.nonlocal = @nla4, class = "nla4"}]
  module private @Quz(
    in %port: !firrtl.uint<1> sym @Quz_port [{circt.nonlocal = @nla4, class = "nla4"}]
  ) {
    // CHECK-NEXT: wire {{.+}} [{circt.nonlocal = @nla3, class = "nla3"}]
    %b = wire sym @b {annotations = [{circt.nonlocal = @nla3, class = "nla3"}]} : !firrtl.uint<1>
  }
  module private @Qux(
    in %port: !firrtl.uint<1> sym @Qux_port [{circt.nonlocal = @nla2, class = "nla2"}]
  ) {
    %a = wire sym @a {annotations = [{circt.nonlocal = @nla1, class = "nla1"}]} : !firrtl.uint<1>
  }
  // CHECK: module private @Baz
  module private @Baz() {
    // CHECK-NEXT: instance {{.+}}
    instance quz sym @quz @Quz(in port: !firrtl.uint<1>)
  }
  module private @Bar() {
    instance qux sym @qux @Qux(in port: !firrtl.uint<1>)
  }
  // CHECK: module private @Foo
  module private @Foo() attributes {annotations = [{class = "firrtl.transforms.FlattenAnnotation"}]} {
    // CHECK-NEXT: %bar_qux_port = wire sym @Qux_port {{.+}} [{class = "nla2"}]
    // CHECK-NEXT: %bar_qux_a = wire {{.+}} [{class = "nla1"}]
    // CHECK-NEXT: %baz_quz_port = wire sym @Quz_port {{.+}} [{class = "nla4"}]
    // CHECK-NEXT: %baz_quz_b = wire {{.+}} [{class = "nla3"}]
    instance bar @Bar()
    instance baz @Baz()
  }
  module @NLAFlatteningChildRoot() {
    instance foo @Foo()
    instance baz @Baz()
  }
}

// Test that symbols are uniqued due to collisions.
//
//   1) An inlined symbol is uniqued.
//   2) An inlined symbol that participates in an NLA is uniqued
//
// CHECK-LABEL: CollidingSymbols
firrtl.circuit "CollidingSymbols" {
  // CHECK-NEXT: hw.hierpath private @nla1 [@CollidingSymbols::@[[FoobarSym:[_a-zA-Z0-9]+]], @Bar]
  hw.hierpath private @nla1 [@CollidingSymbols::@foo, @Foo::@bar, @Bar]
  module @Bar() attributes {annotations = [{circt.nonlocal = @nla1, class = "nla1"}]} {}
  module @Foo() attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
    %b = wire sym @b : !firrtl.uint<1>
    instance bar sym @bar @Bar()
  }
  // CHECK:      module @CollidingSymbols
  // CHECK-NEXT:   wire sym @[[inlinedSymbol:[_a-zA-Z0-9]+]]
  // CHECK-NEXT:   instance foo_bar sym @[[FoobarSym]]
  // CHECK-NOT:    wire sym @[[inlinedSymbol]]
  // CHECK-NOT:    wire sym @[[FoobarSym]]
  module @CollidingSymbols() {
    instance foo sym @foo @Foo()
    %collision_b = wire sym @b : !firrtl.uint<1>
    %collision_bar = wire sym @bar : !firrtl.uint<1>
  }
}

// Test that port symbols are uniqued due to a collision.
//
//   1) An inlined port is uniqued and the NLA is updated.
//
// CHECK-LABEL: CollidingSymbolsPort
firrtl.circuit "CollidingSymbolsPort" {
  // CHECK-NEXT: hw.hierpath private @nla1 [@CollidingSymbolsPort::@foo, @Foo::@[[BarbSym:[_a-zA-Z0-9]+]]]
  hw.hierpath private @nla1 [@CollidingSymbolsPort::@foo, @Foo::@bar, @Bar::@b]
  // CHECK-NOT: module private @Bar
  module private @Bar(
    in %b: !firrtl.uint<1> sym @b [{circt.nonlocal = @nla1, class = "nla1"}]
  ) attributes {annotations = [
    {class = "firrtl.passes.InlineAnnotation"}
  ]} {}
  // CHECK-NEXT: module private @Foo
  module private @Foo() {
    // CHECK-NEXT: wire sym @[[BarbSym]] {annotations = [{circt.nonlocal = @nla1, class = "nla1"}]}
    instance bar sym @bar @Bar(in b: !firrtl.uint<1>)
    // CHECK-NEXT: wire sym @b
    %colliding_b = wire sym @b : !firrtl.uint<1>
  }
  module @CollidingSymbolsPort() {
    instance foo sym @foo @Foo()
  }
}

// Test that colliding symbols that originate from the root of an inlined module
// are properly duplicated and renamed.
//
//   1) The symbol @baz becomes @baz_0 in the top module (as @baz is used)
//   2) The symbol @baz becomes @baz_1 in Foo (as @baz and @baz_0 are both used)
//
// CHECK-LABEL: circuit "CollidingSymbolsReTop"
firrtl.circuit "CollidingSymbolsReTop" {
  // CHECK-NOT:  #hw.innerNameRef<@CollidingSymbolsReTop::@baz>
  // CHECK-NOT:  #hw.innerNameRef<@Foo::@baz>
  // CHECK-NEXT: hw.hierpath private @nla1 [@CollidingSymbolsReTop::@[[TopbazSym:[_a-zA-Z0-9]+]], @Baz::@a]
  // CHECK-NEXT: hw.hierpath private @nla1_0 [@Foo::@[[FoobazSym:[_a-zA-Z0-9]+]], @Baz::@a]
  hw.hierpath private @nla1 [@Bar::@baz, @Baz::@a]
  // CHECK: module @Baz
  module @Baz() {
    // CHECK-NEXT: wire {{.+}} [{circt.nonlocal = @nla1, class = "hello"}, {circt.nonlocal = @nla1_0, class = "hello"}]
    %a = wire sym @a {annotations = [{circt.nonlocal = @nla1, class = "hello"}]} : !firrtl.uint<1>
  }
  module @Bar() attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
    instance baz sym @baz @Baz()
  }
  // CHECK: module @Foo
  module @Foo() {
    // CHECK-NEXT: instance bar_baz sym @[[FoobazSym]] {{.+}}
    instance bar @Bar()
    %colliding_baz = wire sym @baz : !firrtl.uint<1>
    %colliding_baz_0 = wire sym @baz_0 : !firrtl.uint<1>
  }
  // CHECK: module @CollidingSymbolsReTop
  module @CollidingSymbolsReTop() {
    instance foo @Foo()
    // CHECK: instance bar_baz sym @[[TopbazSym]]{{.+}}
    instance bar @Bar()
    instance baz @Baz()
    %colliding_baz = wire sym @baz : !firrtl.uint<1>
  }
}

// Test that when inlining two instances of a module and the port names collide,
// that the NLA is properly updated.  Specifically in this test case, the second
// instance inlined should be renamed, and it should *not* update the NLA.
// CHECK-LABEL: circuit "CollidingSymbolsNLAFixup"
firrtl.circuit "CollidingSymbolsNLAFixup" {
  // CHECK: hw.hierpath private @nla0 [@Foo::@bar, @Bar::@io]
  hw.hierpath private @nla0 [@Foo::@bar, @Bar::@baz0, @Baz::@io]

  // CHECK: hw.hierpath private @nla1 [@Foo::@bar, @Bar::@w]
  hw.hierpath private @nla1 [@Foo::@bar, @Bar::@baz0, @Baz::@w]

  module @Baz(out %io: !firrtl.uint<1> sym @io [{circt.nonlocal = @nla0, class = "test"}])
       attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
    %w = wire sym @w {annotations = [{circt.nonlocal = @nla1, class = "test"}]} : !firrtl.uint<1>
  }

  // CHECK: module @Bar()
  module @Bar() {
    // CHECK: %baz0_io = wire sym @io  {annotations = [{circt.nonlocal = @nla0, class = "test"}]}
    // CHECK: %baz0_w = wire sym @w  {annotations = [{circt.nonlocal = @nla1, class = "test"}]}
    %0 = instance baz0 sym @baz0 @Baz(out io : !firrtl.uint<1>)

    // CHECK: %baz1_io = wire sym @io_0
    // CHECK: %baz1_w = wire sym @w
    %1 = instance baz1 sym @baz1 @Baz(out io : !firrtl.uint<1>)
  }

  module @Foo() {
    instance bar sym @bar @Bar()
  }

  module @CollidingSymbolsNLAFixup() {
    instance system sym @system @Foo()
  }
}

// Test that anything with a "name" will be renamed, even things that FIRRTL
// Dialect doesn't understand.
//
// CHECK-LABEL: circuit "RenameAnything"
firrtl.circuit "RenameAnything" {
  module private @Foo() attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
    "some_unknown_dialect.op"() { name = "world" } : () -> ()
  }
  // CHECK-NEXT: module @RenameAnything
  module @RenameAnything() {
    // CHECK-NEXT: "some_unknown_dialect.op"(){{.+}}name = "hello_world"
    instance hello @Foo()
  }
}

// Test that when an op is inlined into two locations and an annotation on it
// becomes local, that the local annotation is only copied to the clone that
// corresponds to the original NLA path.
// CHECK-LABEL: circuit "AnnotationSplit0"
firrtl.circuit "AnnotationSplit0" {
hw.hierpath private @nla_5560 [@Bar0::@leaf, @Leaf::@w]
hw.hierpath private @nla_5561 [@Bar1::@leaf, @Leaf::@w]
firrtl.module @Leaf() attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
  %w = wire sym @w {annotations = [
    {circt.nonlocal = @nla_5560, class = "test0"},
    {circt.nonlocal = @nla_5561, class = "test1"}]} : !firrtl.uint<8>
}
// CHECK: module @Bar0
firrtl.module @Bar0() {
  // CHECK: %leaf_w = wire sym @w  {annotations = [{class = "test0"}]}
  instance leaf sym @leaf  @Leaf()
}
// CHECK: module @Bar1
firrtl.module @Bar1() {
  // CHECK: %leaf_w = wire sym @w  {annotations = [{class = "test1"}]}
  instance leaf sym @leaf  @Leaf()
}
firrtl.module @AnnotationSplit0() {
  instance bar0 @Bar0()
  instance bar1 @Bar1()
}
}

// Test that when an operation is inlined into two locations and an annotation
// on it should only be copied to a specific clone. This differs from the test
// above in that the annotation does not become a regular local annotation.
// CHECK-LABEL: circuit "AnnotationSplit1"
firrtl.circuit "AnnotationSplit1" {
hw.hierpath private @nla_5560 [@AnnotationSplit1::@bar0, @Bar0::@leaf, @Leaf::@w]
hw.hierpath private @nla_5561 [@AnnotationSplit1::@bar1, @Bar1::@leaf, @Leaf::@w]
firrtl.module @Leaf() attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
  %w = wire sym @w {annotations = [
    {circt.nonlocal = @nla_5560, class = "test0"},
    {circt.nonlocal = @nla_5561, class = "test1"}]} : !firrtl.uint<8>
}
// CHECK: module @Bar0
firrtl.module @Bar0() {
  // CHECK: %leaf_w = wire sym @w  {annotations = [{circt.nonlocal = @nla_5560, class = "test0"}]}
  instance leaf sym @leaf  @Leaf()
}
// CHECK: module @Bar1
firrtl.module @Bar1() {
  // CHECK: %leaf_w = wire sym @w  {annotations = [{circt.nonlocal = @nla_5561, class = "test1"}]}
  instance leaf sym @leaf  @Leaf()
}
firrtl.module @AnnotationSplit1() {
  instance bar0 sym @bar0 @Bar0()
  instance bar1 sym @bar1 @Bar1()
}
}

// Test Rename of InstanceOps.
// https://github.com/llvm/circt/issues/3307
firrtl.circuit "Inline"  {
  // CHECK: circuit "Inline"
  hw.hierpath private @nla_2 [@Inline::@bar, @Bar::@i]
  hw.hierpath private @nla_1 [@Inline::@foo, @Foo::@bar, @Bar::@i]
  // CHECK:   hw.hierpath private @nla_2 [@Inline::@bar, @Bar::@i]
  // CHECK:   hw.hierpath private @nla_1 [@Inline::@[[bar_0:.+]], @Bar::@i]
  module @Inline(in %i: !firrtl.uint<1>, out %o: !firrtl.uint<1>) {
    %foo_i, %foo_o = instance foo sym @foo  @Foo(in i: !firrtl.uint<1>, out o: !firrtl.uint<1>)
    // CHECK:  = instance foo_bar sym @[[bar_0]]  @Bar(in i: !firrtl.uint<1>, out o: !firrtl.uint<1>)
    %bar_i, %bar_o = instance bar sym @bar  @Bar(in i: !firrtl.uint<1>, out o: !firrtl.uint<1>)
    // CHECK:  = instance bar sym @bar  @Bar(in i: !firrtl.uint<1>, out o: !firrtl.uint<1>)
    strictconnect %foo_i, %bar_i : !firrtl.uint<1>
    strictconnect %bar_i, %i : !firrtl.uint<1>
    strictconnect %o, %foo_o : !firrtl.uint<1>
  }
  module private @Bar(in %i: !firrtl.uint<1> sym @i [{circt.nonlocal = @nla_1, class = "test_1"}, {circt.nonlocal = @nla_2, class = "test_2"}], out %o: !firrtl.uint<1>) {
    strictconnect %o, %i : !firrtl.uint<1>
  }
  module private @Foo(in %i: !firrtl.uint<1>, out %o: !firrtl.uint<1>) attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
    %bar_i, %bar_o = instance bar sym @bar  @Bar(in i: !firrtl.uint<1>, out o: !firrtl.uint<1>)
    strictconnect %bar_i, %i : !firrtl.uint<1>
    strictconnect %o, %bar_o : !firrtl.uint<1>
  }
}

firrtl.circuit "Inline2"  {
  // CHECK-LABEL circuit "Inline2"
  hw.hierpath private @nla_1 [@Inline2::@foo, @Foo::@bar, @Bar::@i]
  // CHECK:  hw.hierpath private @nla_1 [@Inline2::@[[bar_0:.+]], @Bar::@i]
  module @Inline2(in %i: !firrtl.uint<1>, out %o: !firrtl.uint<1>) {
    %foo_i, %foo_o = instance foo sym @foo  @Foo(in i: !firrtl.uint<1>, out o: !firrtl.uint<1>)
    %bar = wire sym @bar  : !firrtl.uint<1>
    strictconnect %foo_i, %bar : !firrtl.uint<1>
    strictconnect %bar, %i : !firrtl.uint<1>
    strictconnect %o, %foo_o : !firrtl.uint<1>
  }
  module private @Bar(in %i: !firrtl.uint<1> sym @i [{circt.nonlocal = @nla_1, class = "testing"}], out %o: !firrtl.uint<1>) {
    strictconnect %o, %i : !firrtl.uint<1>
  }
  module private @Foo(in %i: !firrtl.uint<1>, out %o: !firrtl.uint<1>) attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
    %bar_i, %bar_o = instance bar sym @bar  @Bar(in i: !firrtl.uint<1>, out o: !firrtl.uint<1>)
    // CHECK:  = instance foo_bar sym @[[bar_0]]  @Bar(in i: !firrtl.uint<1>, out o: !firrtl.uint<1>)
    strictconnect %bar_i, %i : !firrtl.uint<1>
    strictconnect %o, %bar_o : !firrtl.uint<1>
  }
}

// CHECK-LABEL: circuit "Issue3334"
firrtl.circuit "Issue3334" {
  // CHECK: hw.hierpath private @path_component_old
  // CHECK: hw.hierpath private @path_component_new
  hw.hierpath private @path_component_old [@Issue3334::@foo, @Foo::@bar1, @Bar::@b]
  hw.hierpath private @path_component_new [@Issue3334::@foo, @Foo::@bar1, @Bar]
  module private @Bar() attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
    %b = wire sym @b {annotations = [
      {circt.nonlocal = @path_component_old, "path_component_old"},
      {circt.nonlocal = @path_component_new, "path_component_new"}
    ]} : !firrtl.uint<1>
  }
  module private @Foo() {
    instance bar1 sym @bar1 @Bar()
    instance bar2 sym @bar2 @Bar()
  }
  module @Issue3334() {
    instance foo sym @foo @Foo()
  }
}

// CHECK-LABEL: circuit "Issue3334_flatten"
firrtl.circuit "Issue3334_flatten" {
  // CHECK: hw.hierpath private @path_component_old
  // CHECK: hw.hierpath private @path_component_new
  hw.hierpath private @path_component_old [@Issue3334_flatten::@foo, @Foo::@bar1, @Bar::@b]
  hw.hierpath private @path_component_new [@Issue3334_flatten::@foo, @Foo::@bar1, @Bar]
  module private @Bar() {
    %b = wire sym @b {annotations = [
      {circt.nonlocal = @path_component_old, "path_component_old"},
      {circt.nonlocal = @path_component_new, "path_component_new"}
    ]} : !firrtl.uint<1>
  }
  module private @Foo() attributes {annotations = [{class = "firrtl.transforms.FlattenAnnotation"}]} {
    instance bar1 sym @bar1 @Bar()
    instance bar2 sym @bar2 @Bar()
  }
  module @Issue3334_flatten() {
    instance foo sym @foo @Foo()
  }
}

firrtl.circuit "instNameRename"  {
  hw.hierpath private @nla_5560 [@instNameRename::@bar0, @Bar0::@w, @Bar2::@w, @Bar1]
  // CHECK:  hw.hierpath private @nla_5560 [@instNameRename::@[[w_1:.+]], @Bar2::@w, @Bar1]
  hw.hierpath private @nla_5560_1 [@instNameRename::@bar1, @Bar0::@w, @Bar2::@w, @Bar1]
  // CHECK:  hw.hierpath private @nla_5560_1 [@instNameRename::@[[w_2:.+]], @Bar2::@w, @Bar1]
  module @Bar1() {
    %w = wire   {annotations = [{circt.nonlocal = @nla_5560, class = "test0"}, {circt.nonlocal = @nla_5560_1, class = "test1"}]} : !firrtl.uint<8>
  }
  module @Bar2() {
    instance leaf sym @w  @Bar1()
  }
  module @Bar0() attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
    instance leaf sym @w  @Bar2()
  }
  module @instNameRename() {
    instance no sym @no  @Bar0()
    instance bar0 sym @bar0  @Bar0()
    instance bar1 sym @bar1  @Bar0()
    // CHECK:  instance bar0_leaf sym @[[w_1]]  @Bar2()
    // CHECK:  instance bar1_leaf sym @[[w_2]]  @Bar2()
    %w = wire sym @w   : !firrtl.uint<8>
  }
}

// This test checks for context sensitive Hierpath update.
// The following inlining causes 4 instances of @Baz being added to @Foo1,
// but only two of them should have valid HierPathOps.
firrtl.circuit "CollidingSymbolsMultiInline" {

  hw.hierpath private @nla1 [@Foo1::@bar1, @Foo2::@bar, @Foo::@bar, @Bar::@w, @Baz::@w]
  // CHECK: hw.hierpath private @nla1 [@Foo1::@w_0, @Baz::@w]
  hw.hierpath private @nla2 [@Foo1::@bar2, @Foo2::@bar1, @Foo::@bar, @Bar::@w, @Baz::@w]
  // CHECK:  hw.hierpath private @nla2 [@Foo1::@w_7, @Baz::@w]

  module @Baz(out %io: !firrtl.uint<1> )
        {
    %w = wire sym @w {annotations = [{circt.nonlocal = @nla1, class = "test"}, {circt.nonlocal = @nla2, class = "test"}]} : !firrtl.uint<1>
  }

  module @Bar() attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]}{
    %0 = instance baz0 sym @w    @Baz(out io : !firrtl.uint<1>)
  }

  module @Foo() attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]}{
    instance bar sym @bar @Bar()
    %w = wire sym @w : !firrtl.uint<1>
  }

  module @Foo2() attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]}{
    instance bar sym @bar @Foo()
    instance bar sym @bar1 @Foo()
    %w = wire sym @w : !firrtl.uint<1>
  }

  module @Foo1() {
    instance bar sym @bar1 @Foo2()
    instance bar sym @bar2 @Foo2()
    %w = wire sym @bar : !firrtl.uint<1>
    %w1 = wire sym @w : !firrtl.uint<1>
    // CHECK:  %bar_bar_bar_baz0_io = instance bar_bar_bar_baz0 sym @w_0  @Baz(out io: !firrtl.uint<1>)
    // CHECK:  %bar_bar_bar_baz0_io_0 = instance bar_bar_bar_baz0 sym @w_2  @Baz(out io: !firrtl.uint<1>)
    // CHECK:  %bar_bar_bar_baz0_io_2 = instance bar_bar_bar_baz0 sym @w_5  @Baz(out io: !firrtl.uint<1>)
    // CHECK:  %bar_bar_bar_baz0_io_4 = instance bar_bar_bar_baz0 sym @w_7  @Baz(out io: !firrtl.uint<1>)
  }

  module @CollidingSymbolsMultiInline() {
    instance system sym @system @Foo1()
  }
}

// -----

// Test proper hierarchical inlining of RefType
// CHECK-LABEL: circuit "Top" {
firrtl.circuit "Top" {
  module @XmrSrcMod(out %_a: !firrtl.probe<uint<1>>) attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]}{
    %zero = constant 0 : !firrtl.uint<1>
    %1 = ref.send %zero : !firrtl.uint<1>
    ref.define %_a, %1 : !firrtl.probe<uint<1>>
  }
  module @Bar(out %_a: !firrtl.probe<uint<1>>) attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]}{
    %xmr   = instance bar sym @barXMR @XmrSrcMod(out _a: !firrtl.probe<uint<1>>)
    ref.define %_a, %xmr   : !firrtl.probe<uint<1>>
    // CHECK:  %c0_ui1 = constant 0 : !firrtl.uint<1>
    // CHECK:  %0 = ref.send %c0_ui1 : !firrtl.uint<1>
    // CHECK:  ref.define %_a, %0 : !firrtl.probe<uint<1>>
  }
  module @Top() {
    %bar_a = instance bar sym @bar  @Bar(out _a: !firrtl.probe<uint<1>>)
    %a = wire : !firrtl.uint<1>
    %0 = ref.resolve %bar_a : !firrtl.probe<uint<1>>
    strictconnect %a, %0 : !firrtl.uint<1>
    // CHECK:  %c0_ui1 = constant 0 : !firrtl.uint<1>
    // CHECK:  %0 = ref.send %c0_ui1 : !firrtl.uint<1>
    // CHECK:  %a = wire   : !firrtl.uint<1>
    // CHECK:  %1 = ref.resolve %0 : !firrtl.probe<uint<1>>
    // CHECK:  strictconnect %a, %1 : !firrtl.uint<1>
  }
}

// -----

// Test proper inlining of RefSend to Ports of RefType
// CHECK-LABEL: circuit "Top" {
firrtl.circuit "Top" {
  module @XmrSrcMod(in %pa: !firrtl.uint<1>, out %_a: !firrtl.probe<uint<1>>) attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]}{
    %1 = ref.send %pa : !firrtl.uint<1>
    ref.define %_a, %1 : !firrtl.probe<uint<1>>
  }
  module @Bar(out %_a: !firrtl.probe<uint<1>>)  attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]}{
    %pa, %xmr   = instance bar sym @barXMR @XmrSrcMod(in pa: !firrtl.uint<1>, out _a: !firrtl.probe<uint<1>>)
    // CHECK:  %bar_pa = wire   : !firrtl.uint<1>
    // CHECK:  %0 = ref.send %bar_pa : !firrtl.uint<1>
    // CHECK:  ref.define %_a, %0 : !firrtl.probe<uint<1>>
    ref.define %_a, %xmr   : !firrtl.probe<uint<1>>
  }
  module @Top() {
    %bar_a = instance bar sym @bar  @Bar(out _a: !firrtl.probe<uint<1>>)
    %a = wire : !firrtl.uint<1>
    %0 = ref.resolve %bar_a : !firrtl.probe<uint<1>>
    strictconnect %a, %0 : !firrtl.uint<1>
    // CHECK:  %bar_bar_pa = wire   : !firrtl.uint<1>
    // CHECK:  %0 = ref.send %bar_bar_pa : !firrtl.uint<1>
    // CHECK:  %a = wire   : !firrtl.uint<1>
    // CHECK:  %1 = ref.resolve %0 : !firrtl.probe<uint<1>>
  }
}

// -----

// Test for multiple readers and multiple instances of RefType
// CHECK-LABEL: circuit "Top" {
firrtl.circuit "Top" {
  module @XmrSrcMod(out %_a: !firrtl.probe<uint<1>>) {
    %zero = constant 0 : !firrtl.uint<1>
    %1 = ref.send %zero : !firrtl.uint<1>
    ref.define %_a, %1 : !firrtl.probe<uint<1>>
  }
  module @Foo(out %_a: !firrtl.probe<uint<1>>) {
    %xmr   = instance bar sym @fooXMR @XmrSrcMod(out _a: !firrtl.probe<uint<1>>)
    ref.define %_a, %xmr   : !firrtl.probe<uint<1>>
    %0 = ref.resolve %xmr   : !firrtl.probe<uint<1>>
    %a = wire : !firrtl.uint<1>
    strictconnect %a, %0 : !firrtl.uint<1>
  }
  module @Bar(out %_a: !firrtl.probe<uint<1>>) {
    %xmr   = instance bar sym @barXMR @XmrSrcMod(out _a: !firrtl.probe<uint<1>>)
    ref.define %_a, %xmr   : !firrtl.probe<uint<1>>
    %0 = ref.resolve %xmr   : !firrtl.probe<uint<1>>
    %a = wire : !firrtl.uint<1>
    strictconnect %a, %0 : !firrtl.uint<1>
  }
  module @Top() attributes {annotations = [{class = "firrtl.transforms.FlattenAnnotation"}]}{
    %bar_a = instance bar sym @bar  @Bar(out _a: !firrtl.probe<uint<1>>)
    // CHECK:  %c0_ui1 = constant 0 : !firrtl.uint<1>
    // CHECK:  %0 = ref.send %c0_ui1 : !firrtl.uint<1>
    // CHECK:  %1 = ref.resolve %0 : !firrtl.probe<uint<1>>
    // CHECK:  %bar_a = wire   : !firrtl.uint<1>
    // CHECK:  strictconnect %bar_a, %1 : !firrtl.uint<1>
    %foo_a = instance foo sym @foo @Foo(out _a: !firrtl.probe<uint<1>>)
    // CHECK:  %c0_ui1_0 = constant 0 : !firrtl.uint<1>
    // CHECK:  %2 = ref.send %c0_ui1_0 : !firrtl.uint<1>
    // CHECK:  %3 = ref.resolve %2 : !firrtl.probe<uint<1>>
    // CHECK:  %foo_a = wire   : !firrtl.uint<1>
    // CHECK:  strictconnect %foo_a, %3 : !firrtl.uint<1>
    %xmr_a = instance xmr sym @xmr @XmrSrcMod(out _a: !firrtl.probe<uint<1>>)
    // CHECK:  %c0_ui1_1 = constant 0 : !firrtl.uint<1>
    // CHECK:  %4 = ref.send %c0_ui1_1 : !firrtl.uint<1>
    %a = wire : !firrtl.uint<1>
    %b = wire : !firrtl.uint<1>
    %c = wire : !firrtl.uint<1>
    %0 = ref.resolve %bar_a : !firrtl.probe<uint<1>>
    %1 = ref.resolve %foo_a : !firrtl.probe<uint<1>>
    %2 = ref.resolve %xmr_a : !firrtl.probe<uint<1>>
    // CHECK:  %5 = ref.resolve %0 : !firrtl.probe<uint<1>>
    // CHECK:  %6 = ref.resolve %2 : !firrtl.probe<uint<1>>
    // CHECK:  %7 = ref.resolve %4 : !firrtl.probe<uint<1>>
    strictconnect %a, %0 : !firrtl.uint<1>
    strictconnect %b, %1 : !firrtl.uint<1>
    strictconnect %c, %2 : !firrtl.uint<1>
    // CHECK:  strictconnect %a, %5 : !firrtl.uint<1>
    // CHECK:  strictconnect %b, %6 : !firrtl.uint<1>
    // CHECK:  strictconnect %c, %7 : !firrtl.uint<1>
  }
}

// -----

// Test for inlining module with RefType input port.
// CHECK-LABEL: circuit "Top" {
firrtl.circuit "Top" {
  module @XmrSrcMod(out %_a: !firrtl.probe<uint<1>>) {
    %zero = constant 0 : !firrtl.uint<1>
    %1 = ref.send %zero : !firrtl.uint<1>
    ref.define %_a, %1 : !firrtl.probe<uint<1>>
  }
  module @Top() {
    %xmr = instance xmr sym @TopXMR @XmrSrcMod(out _a: !firrtl.probe<uint<1>>)
    %a = wire : !firrtl.uint<1>
    %0 = ref.resolve %xmr : !firrtl.probe<uint<1>>
    strictconnect %a, %0 : !firrtl.uint<1>
    %c_a = instance child @Child(in  _a: !firrtl.probe<uint<1>>)
    ref.define %c_a, %xmr : !firrtl.probe<uint<1>>
    // CHECK:  %1 = ref.resolve %xmr__a : !firrtl.probe<uint<1>>
    // CHECK:  %child_child__a = instance child_child  @Child2(in _a: !firrtl.probe<uint<1>>)
    // CHECK:  ref.define %child_child__a, %xmr__a : !firrtl.probe<uint<1>>
  }
  module @Child(in  %_a: !firrtl.probe<uint<1>>)  attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]}{
    %0 = ref.resolve %_a : !firrtl.probe<uint<1>>
    %c_a = instance child @Child2(in  _a: !firrtl.probe<uint<1>>)
    ref.define %c_a, %_a : !firrtl.probe<uint<1>>
  }
  module @Child2(in  %_a: !firrtl.probe<uint<1>>) {
    %0 = ref.resolve %_a : !firrtl.probe<uint<1>>
  }
}

// -----

// Test for inlining module with RefType input port.
// CHECK-LABEL: circuit "Top" {
firrtl.circuit "Top" {
  module @XmrSrcMod(out %_a: !firrtl.probe<uint<1>>) {
    %zero = constant 0 : !firrtl.uint<1>
    %1 = ref.send %zero : !firrtl.uint<1>
    ref.define %_a, %1 : !firrtl.probe<uint<1>>
  }
  module @Top() {
    %xmr = instance xmr sym @TopXMR @XmrSrcMod(out _a: !firrtl.probe<uint<1>>)
    %a = wire : !firrtl.uint<1>
    %0 = ref.resolve %xmr : !firrtl.probe<uint<1>>
    strictconnect %a, %0 : !firrtl.uint<1>
    %c_a = instance child @Child(in  _a: !firrtl.probe<uint<1>>)
    ref.define %c_a, %xmr : !firrtl.probe<uint<1>>
    // CHECK:  %1 = ref.resolve %xmr__a : !firrtl.probe<uint<1>>
    // CHECK:  %2 = ref.resolve %xmr__a : !firrtl.probe<uint<1>>
  }
  module @Child(in  %_a: !firrtl.probe<uint<1>>)  attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]}{
    %0 = ref.resolve %_a : !firrtl.probe<uint<1>>
    %c_a = instance child @Child2(in  _a: !firrtl.probe<uint<1>>)
    ref.define %c_a, %_a : !firrtl.probe<uint<1>>
  }
  module @Child2(in  %_a: !firrtl.probe<uint<1>>)  attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]}{
    %0 = ref.resolve %_a : !firrtl.probe<uint<1>>
  }
}

// -----

// Test for recursive inlining of modules with RefType input port.
// CHECK-LABEL: circuit "Top" {
firrtl.circuit "Top" {
  module @XmrSrcMod(out %_a: !firrtl.probe<uint<1>>) {
    %zero = constant 0 : !firrtl.uint<1>
    %1 = ref.send %zero : !firrtl.uint<1>
    ref.define %_a, %1 : !firrtl.probe<uint<1>>
  }
  module @Top() {
    %xmr = instance xmr sym @TopXMR @XmrSrcMod(out _a: !firrtl.probe<uint<1>>)
    %a = wire : !firrtl.uint<1>
    %xmr2 = ref.send %a : !firrtl.uint<1>
    %c_a1, %c_a2  = instance child @Child(in  _a1: !firrtl.probe<uint<1>>, in  _a2: !firrtl.probe<uint<1>>)
    ref.define %c_a1, %xmr : !firrtl.probe<uint<1>>
    ref.define %c_a2, %xmr2 : !firrtl.probe<uint<1>>
    // CHECK:  %1 = ref.resolve %xmr__a : !firrtl.probe<uint<1>>
    // CHECK:  %2 = ref.resolve %xmr__a : !firrtl.probe<uint<1>>
    // CHECK:  %3 = ref.resolve %0 : !firrtl.probe<uint<1>>
    // CHECK:  %child_cw = wire   : !firrtl.uint<1>
    // CHECK:  strictconnect %child_cw, %3 : !firrtl.uint<1>
  }
  module @Child(in  %_a1: !firrtl.probe<uint<1>>, in  %_a2: !firrtl.probe<uint<1>>)  attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]}{
    %c_a = instance child @Child2(in  _a: !firrtl.probe<uint<1>>)
    // CHECK:  %0 = ref.resolve %_a1 : !firrtl.probe<uint<1>>
    // CHECK:  %1 = ref.resolve %_a1 : !firrtl.probe<uint<1>>
    ref.define %c_a, %_a1 : !firrtl.probe<uint<1>>
    %0 = ref.resolve %_a2 : !firrtl.probe<uint<1>>
    %cw = wire : !firrtl.uint<1>
    strictconnect %cw, %0 : !firrtl.uint<1>
  }
  module @Child2(in  %_a: !firrtl.probe<uint<1>>)   attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]}{
    %0 = ref.resolve %_a : !firrtl.probe<uint<1>>
    %c_a = instance child @Child3(in  _b: !firrtl.probe<uint<1>>)
    ref.define %c_a, %_a : !firrtl.probe<uint<1>>
  }
  module @Child3(in  %_b: !firrtl.probe<uint<1>>)   attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]}{
    %0 = ref.resolve %_b : !firrtl.probe<uint<1>>
  }
}

// -----

// Test for flatten annotation, and remove unused port wires
// CHECK-LABEL: circuit "Top"
firrtl.circuit "Top" {
  module @XmrSrcMod(out %_a: !firrtl.probe<uint<1>>) {
    %zero = constant 0 : !firrtl.uint<1>
    %1 = ref.send %zero : !firrtl.uint<1>
    ref.define %_a, %1 : !firrtl.probe<uint<1>>
  }
  module @Top() attributes {annotations = [{class = "firrtl.transforms.FlattenAnnotation"}]}{
    %xmr = instance xmr sym @TopXMR @XmrSrcMod(out _a: !firrtl.probe<uint<1>>)
    // CHECK:  %c0_ui1 = constant 0 : !firrtl.uint<1>
    // CHECK:  %0 = ref.send %c0_ui1 : !firrtl.uint<1>
    %a = wire : !firrtl.uint<1>
    %xmr2 = ref.send %a : !firrtl.uint<1>
    %0 = ref.resolve %xmr : !firrtl.probe<uint<1>>
    // CHECK:  %2 = ref.resolve %0 : !firrtl.probe<uint<1>>
    strictconnect %a, %0 : !firrtl.uint<1>
    %c_a1, %c_a2  = instance child @Child(in  _a1: !firrtl.probe<uint<1>>, in  _a2: !firrtl.probe<uint<1>>)
    // CHECK:  %3 = ref.resolve %0 : !firrtl.probe<uint<1>>
    // CHECK:  %4 = ref.resolve %0 : !firrtl.probe<uint<1>>
    // CHECK:  %5 = ref.resolve %1 : !firrtl.probe<uint<1>>
    // CHECK:  %child_cw = wire   : !firrtl.uint<1>
    // CHECK:  strictconnect %child_cw, %5 : !firrtl.uint<1>
    ref.define %c_a1, %xmr : !firrtl.probe<uint<1>>
    ref.define %c_a2, %xmr2 : !firrtl.probe<uint<1>>
  }
  module @Child(in  %_a1: !firrtl.probe<uint<1>>, in  %_a2: !firrtl.probe<uint<1>>)  {
    %c_a = instance child @Child2(in  _a: !firrtl.probe<uint<1>>)
    ref.define %c_a, %_a1 : !firrtl.probe<uint<1>>
    %0 = ref.resolve %_a2 : !firrtl.probe<uint<1>>
    %cw = wire : !firrtl.uint<1>
    strictconnect %cw, %0 : !firrtl.uint<1>
  }
  module @Child2(in  %_a: !firrtl.probe<uint<1>>){
    %0 = ref.resolve %_a : !firrtl.probe<uint<1>>
    %c_a = instance child @Child3(in  _b: !firrtl.probe<uint<1>>)
    ref.define %c_a, %_a : !firrtl.probe<uint<1>>
  }
  module @Child3(in  %_b: !firrtl.probe<uint<1>>){
    %0 = ref.resolve %_b : !firrtl.probe<uint<1>>
  }
}

// -----

// Test for U-Turn in ref ports. The inlined module defines and uses the RefPort.
// CHECK-LABEL: circuit "Top"
firrtl.circuit "Top" {
  module @Top() {
    %c_a, %c_o = instance child @Child(in  _a: !firrtl.probe<uint<1>>, out  o_a: !firrtl.probe<uint<1>>)
    ref.define %c_a, %c_o : !firrtl.probe<uint<1>>
    // CHECK:  %child_bar__a = instance child_bar sym @bar  @Bar(out _a: !firrtl.probe<uint<1>>)
    // CHECK:  %0 = ref.resolve %child_bar__a : !firrtl.probe<uint<1>>
  }
  module @Child(in  %_a: !firrtl.probe<uint<1>>, out  %o_a: !firrtl.probe<uint<1>>)   attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]}{
    %0 = ref.resolve %_a : !firrtl.probe<uint<1>>
    %bar_a = instance bar sym @bar  @Bar(out _a: !firrtl.probe<uint<1>>)
    ref.define %o_a, %bar_a : !firrtl.probe<uint<1>>
  }
  module @Bar(out %_a: !firrtl.probe<uint<1>>) {
    %pa, %xmr   = instance bar sym @barXMR @XmrSrcMod(in pa: !firrtl.uint<1>, out _a: !firrtl.probe<uint<1>>)
    ref.define %_a, %xmr   : !firrtl.probe<uint<1>>
  }
  module @XmrSrcMod(in %pa: !firrtl.uint<1>, out %_a: !firrtl.probe<uint<1>>) {
    %1 = ref.send %pa : !firrtl.uint<1>
    ref.define %_a, %1 : !firrtl.probe<uint<1>>
  }
}


// -----

// PR #4882 fixes a bug, which was producing invalid NLAs.
// error: 'hw.hierpath' op  module: "instNameRename" does not contain any instance with symbol: "w"
// Due to coincidental name collisions, renaming was not updating the actual hierpath.
firrtl.circuit "Bug4882Rename"  {
  hw.hierpath private @nla_5560 [@Bug4882Rename::@w, @Bar2::@x]
  module private @Bar2() attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]}{
    %x = wire sym @x  {annotations = [{circt.nonlocal = @nla_5560, class = "test0"}]} : !firrtl.uint<8>
  }
  module private @Bar1() attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]}{
    instance bar3 sym @w  @Bar3()
  }
  module private @Bar3()  {
    %w = wire sym @w1   : !firrtl.uint<8>
  }
  module @Bug4882Rename() {
  // CHECK-LABEL: module @Bug4882Rename() {
    instance no sym @no  @Bar1()
    // CHECK-NEXT: instance no_bar3 sym @w_0 @Bar3()
    instance bar2 sym @w  @Bar2()
    // CHECK-NEXT: %bar2_x = wire sym @x {annotations = [{class = "test0"}]}
  }
}

// -----

// Issue #4920, the recursive inlining should consider the correct retop for NLAs.

firrtl.circuit "DidNotContainSymbol" {
  hw.hierpath private @path [@Bar1::@w, @Bar3]
  module private @Bar2() attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
    instance no sym @no @Bar1()
  }
  module private @Bar1() attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
    instance bar3 sym @w @Bar3()
  }
  module private @Bar3() attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
    %w = wire sym @w {annotations = [{circt.nonlocal = @path , class = "test0"}]} : !firrtl.uint<8>
  }
  module @DidNotContainSymbol() {
    instance bar2 sym @w @Bar2()
  }
  // CHECK-LABEL: module @DidNotContainSymbol() {
  // CHECK-NEXT:     %bar2_no_bar3_w = wire sym @w_0 {annotations = [{class = "test0"}]} : !firrtl.uint<8>
  // CHECK-NEXT:  }
}

// -----

// Issue #4915, the NLAs should be updated with renamed extern module instance.

firrtl.circuit "SimTop" {
  hw.hierpath private @nla_61 [@Rob::@difftest_3, @DifftestLoadEvent]
  // CHECK: hw.hierpath private @nla_61 [@SimTop::@difftest_3_0, @DifftestLoadEvent]
  hw.hierpath private @nla_60 [@Rob::@difftest_2, @DifftestLoadEvent]
  // CHECK: hw.hierpath private @nla_60 [@SimTop::@difftest_2, @DifftestLoadEvent]
  extmodule private @DifftestIntWriteback()
  extmodule private @DifftestLoadEvent() attributes {annotations = [{circt.nonlocal = @nla_60, class = "B"}, {circt.nonlocal = @nla_61, class = "B"}]}
	// CHECK: extmodule private @DifftestLoadEvent() attributes {annotations = [{circt.nonlocal = @nla_60, class = "B"}, {circt.nonlocal = @nla_61, class = "B"}]}
  module private @Rob() attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
    instance difftest_2 sym @difftest_2 @DifftestLoadEvent()
    instance difftest_3 sym @difftest_3 @DifftestLoadEvent()
  }
  module private @CtrlBlock() attributes {annotations = [{class = "firrtl.passes.InlineAnnotation"}]} {
    instance rob @Rob()
  }
  module @SimTop() {
    instance difftest_3 sym @difftest_3 @DifftestIntWriteback()
    instance ctrlBlock @CtrlBlock()
    // CHECK:  instance difftest_3 sym @difftest_3 @DifftestIntWriteback()
    // CHECK:  instance ctrlBlock_rob_difftest_2 sym @difftest_2 @DifftestLoadEvent()
    // CHECK:  instance ctrlBlock_rob_difftest_3 sym @difftest_3_0 @DifftestLoadEvent()
  }
}

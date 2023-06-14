// RUN: circt-opt --pass-pipeline='builtin.module(firrtl.circuit(firrtl-dedup))' %s | FileCheck %s

// CHECK-LABEL: circuit "Empty"
firrtl.circuit "Empty" {
  // CHECK: module @Empty0
  module @Empty0(in %i0: !firrtl.uint<1>) { }
  // CHECK-NOT: module @Empty1
  module @Empty1(in %i1: !firrtl.uint<1>) { }
  // CHECK-NOT: module @Empty2
  module @Empty2(in %i2: !firrtl.uint<1>) { }
  module @Empty() {
    // CHECK: %e0_i0 = instance e0  @Empty0
    // CHECK: %e1_i0 = instance e1  @Empty0
    // CHECK: %e2_i0 = instance e2  @Empty0
    %e0_i0 = instance e0 @Empty0(in i0: !firrtl.uint<1>)
    %e1_i1 = instance e1 @Empty1(in i1: !firrtl.uint<1>)
    %e2_i2 = instance e2 @Empty2(in i2: !firrtl.uint<1>)
  }
}


// CHECK-LABEL: circuit "Simple"
firrtl.circuit "Simple" {
  // CHECK: module @Simple0
  module @Simple0() {
    %a = wire: !firrtl.bundle<a: uint<1>>
  }
  // CHECK-NOT: module @Simple1
  module @Simple1() {
    %b = wire: !firrtl.bundle<b: uint<1>>
  }
  module @Simple() {
    // CHECK: instance simple0 @Simple0
    // CHECK: instance simple1 @Simple0
    instance simple0 @Simple0()
    instance simple1 @Simple1()
  }
}

// CHECK-LABEL: circuit "PrimOps"
firrtl.circuit "PrimOps" {
  // CHECK: module @PrimOps0
  module @PrimOps0(in %a: !firrtl.bundle<a: uint<2>, b: uint<2>, c flip: uint<2>>) {
    %a_a = subfield %a[a] : !firrtl.bundle<a: uint<2>, b: uint<2>, c flip: uint<2>>
    %a_b = subfield %a[b] : !firrtl.bundle<a: uint<2>, b: uint<2>, c flip: uint<2>>
    %a_c = subfield %a[c] : !firrtl.bundle<a: uint<2>, b: uint<2>, c flip: uint<2>>
    %0 = xor %a_a, %a_b: (!firrtl.uint<2>, !firrtl.uint<2>) -> !firrtl.uint<2>
    connect %a_c, %a_b: !firrtl.uint<2>, !firrtl.uint<2>
  }
  // CHECK-NOT: module @PrimOps1
  module @PrimOps1(in %b: !firrtl.bundle<a: uint<2>, b: uint<2>, c flip: uint<2>>) {
    %b_a = subfield %b[a] : !firrtl.bundle<a: uint<2>, b: uint<2>, c flip: uint<2>>
    %b_b = subfield %b[b] : !firrtl.bundle<a: uint<2>, b: uint<2>, c flip: uint<2>>
    %b_c = subfield %b[c] : !firrtl.bundle<a: uint<2>, b: uint<2>, c flip: uint<2>>
    %0 = xor %b_a, %b_b: (!firrtl.uint<2>, !firrtl.uint<2>) -> !firrtl.uint<2>
    connect %b_c, %b_b: !firrtl.uint<2>, !firrtl.uint<2>
  }
  module @PrimOps() {
    // CHECK: instance primops0 @PrimOps0
    // CHECK: instance primops1 @PrimOps0
    %primops0 = instance primops0 @PrimOps0(in a: !firrtl.bundle<a: uint<2>, b: uint<2>, c flip: uint<2>>)
    %primops1 = instance primops1 @PrimOps1(in b: !firrtl.bundle<a: uint<2>, b: uint<2>, c flip: uint<2>>)
  }
}

// Check that when operations are recursively merged.
// CHECK-LABEL: circuit "WhenOps"
firrtl.circuit "WhenOps" {
  // CHECK: module @WhenOps0
  module @WhenOps0(in %p : !firrtl.uint<1>) {
    // CHECK: when %p : !firrtl.uint<1> {
    // CHECK:  %w = wire : !firrtl.uint<8>
    // CHECK: }
    when %p : !firrtl.uint<1> {
      %w = wire : !firrtl.uint<8>
    }
  }
  // CHECK-NOT: module @PrimOps1
  module @WhenOps1(in %p : !firrtl.uint<1>) {
    when %p : !firrtl.uint<1> {
      %w = wire : !firrtl.uint<8>
    }
  }
  module @WhenOps() {
    // CHECK: instance whenops0 @WhenOps0
    // CHECK: instance whenops1 @WhenOps0
    %whenops0 = instance whenops0 @WhenOps0(in p : !firrtl.uint<1>)
    %whenops1 = instance whenops1 @WhenOps1(in p : !firrtl.uint<1>)
  }
}

// CHECK-LABEL: circuit "Annotations"
firrtl.circuit "Annotations" {
  // CHECK: hw.hierpath private [[NLA0:@nla.*]] [@Annotations::@annotations1, @Annotations0]
  // CHECK: hw.hierpath private @annos_nla0 [@Annotations::@annotations0, @Annotations0::@c]
  hw.hierpath private @annos_nla0 [@Annotations::@annotations0, @Annotations0::@c]
  // CHECK: hw.hierpath private @annos_nla1 [@Annotations::@annotations1, @Annotations0::@c]
  hw.hierpath private @annos_nla1 [@Annotations::@annotations1, @Annotations1::@j]
  // CHECK: hw.hierpath private @annos_nla2 [@Annotations::@annotations0, @Annotations0]
  hw.hierpath private @annos_nla2 [@Annotations::@annotations0, @Annotations0]

  // CHECK: module @Annotations0() attributes {annotations = [{circt.nonlocal = [[NLA0]], class = "one"}]}
  module @Annotations0() {
    // Annotation from other module becomes non-local.
    // CHECK: %a = wire {annotations = [{circt.nonlocal = [[NLA0]], class = "one"}]}
    %a = wire : !firrtl.uint<1>

    // Annotation from this module becomes non-local.
    // CHECK: %b = wire {annotations = [{circt.nonlocal = @annos_nla2, class = "one"}]}
    %b = wire {annotations = [{class = "one"}]} : !firrtl.uint<1>

    // Two non-local annotations are unchanged, as they have enough context in the NLA already.
    // CHECK: %c = wire sym @c  {annotations = [{circt.nonlocal = @annos_nla0, class = "NonLocal"}, {circt.nonlocal = @annos_nla1, class = "NonLocal"}]}
    %c = wire sym @c {annotations = [{circt.nonlocal = @annos_nla0, class = "NonLocal"}]} : !firrtl.uint<1>

    // Same test as above but with the hiearchical path targeting the module.
    // CHECK: %d = wire {annotations = [{circt.nonlocal = @annos_nla2, class = "NonLocal"}, {circt.nonlocal = @annos_nla2, class = "NonLocal"}]}
    %d = wire {annotations = [{circt.nonlocal = @annos_nla2, class = "NonLocal"}]} : !firrtl.uint<1>

    // Same annotation on both ops should become non-local.
    // CHECK: %e = wire {annotations = [{circt.nonlocal = @annos_nla2, class = "both"}, {circt.nonlocal = [[NLA0]], class = "both"}]}
    %e = wire {annotations = [{class = "both"}]} : !firrtl.uint<1>

    // Dont touch on both ops should become local.
    // CHECK: %f = wire  {annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]}
    // CHECK %f = wire {annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}, {circt.nonlocal = @annos_nla2, class = "firrtl.transforms.DontTouchAnnotation"}]}
    %f = wire {annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]} : !firrtl.uint<1>

    // Subannotations should be handled correctly.
    // CHECK: %g = wire {annotations = [{circt.fieldID = 1 : i32, circt.nonlocal = @annos_nla2, class = "subanno"}]}
    %g = wire {annotations = [{circt.fieldID = 1 : i32, class = "subanno"}]} : !firrtl.bundle<a: uint<1>>
  }
  // CHECK-NOT: module @Annotations1
  module @Annotations1() attributes {annotations = [{class = "one"}]} {
    %h = wire {annotations = [{class = "one"}]} : !firrtl.uint<1>
    %i = wire : !firrtl.uint<1>
    %j = wire sym @j {annotations = [{circt.nonlocal = @annos_nla1, class = "NonLocal"}]} : !firrtl.uint<1>
    %k = wire {annotations = [{circt.nonlocal = @annos_nla2, class = "NonLocal"}]} : !firrtl.uint<1>
    %l = wire {annotations = [{class = "both"}]} : !firrtl.uint<1>
    %m = wire {annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]} : !firrtl.uint<1>
    %n = wire : !firrtl.bundle<a: uint<1>>
  }
  module @Annotations() {
    // CHECK: instance annotations0 sym @annotations0  @Annotations0()
    // CHECK: instance annotations1 sym @annotations1  @Annotations0()
    instance annotations0 sym @annotations0 @Annotations0()
    instance annotations1 sym @annotations1 @Annotations1()
  }
}

// Special handling of DontTouch.
// CHECK-LABEL: circuit "DontTouch"
firrtl.circuit "DontTouch" {
hw.hierpath private @nla0 [@DontTouch::@bar, @Bar::@auto]
hw.hierpath private @nla1 [@DontTouch::@baz, @Baz::@auto]
firrtl.module @DontTouch() {
  // CHECK: %bar_auto = instance bar sym @bar @Bar(out auto: !firrtl.bundle<a: uint<1>, b: uint<1>>)
  // CHECK: %baz_auto = instance baz sym @baz @Bar(out auto: !firrtl.bundle<a: uint<1>, b: uint<1>>)
  %bar_auto = instance bar sym @bar @Bar(out auto: !firrtl.bundle<a: uint<1>, b: uint<1>>)
  %baz_auto = instance baz sym @baz @Baz(out auto: !firrtl.bundle<a: uint<1>, b: uint<1>>)
}
// CHECK:      module private @Bar(
// CHECK-SAME:   out %auto: !firrtl.bundle<a: uint<1>, b: uint<1>> sym @auto
// CHECK-SAME:   [{circt.fieldID = 1 : i32, class = "firrtl.transforms.DontTouchAnnotation"},
// CHECK-SAME:    {circt.fieldID = 2 : i32, class = "firrtl.transforms.DontTouchAnnotation"}]) {
firrtl.module private @Bar(out %auto: !firrtl.bundle<a: uint<1>, b: uint<1>> sym @auto
  [{circt.nonlocal = @nla0, circt.fieldID = 1 : i32, class = "firrtl.transforms.DontTouchAnnotation"},
  {circt.fieldID = 2 : i32, class = "firrtl.transforms.DontTouchAnnotation"}]) { }
// CHECK-NOT: module private @Baz
firrtl.module private @Baz(out %auto: !firrtl.bundle<a: uint<1>, b: uint<1>> sym @auto
  [{circt.fieldID = 1 : i32, class = "firrtl.transforms.DontTouchAnnotation"},
  {circt.nonlocal = @nla1, circt.fieldID = 2 : i32, class = "firrtl.transforms.DontTouchAnnotation"}]) { }
}


// Check that module and memory port annotations are merged correctly.
// CHECK-LABEL: circuit "PortAnnotations"
firrtl.circuit "PortAnnotations" {
  // CHECK: hw.hierpath private [[NLA1:@nla.*]] [@PortAnnotations::@portannos1, @PortAnnotations0]
  // CHECK: hw.hierpath private [[NLA0:@nla.*]] [@PortAnnotations::@portannos0, @PortAnnotations0]
  // CHECK: module @PortAnnotations0(in %a: !firrtl.uint<1> [
  // CHECK-SAME: {circt.nonlocal = [[NLA0]], class = "port0"},
  // CHECK-SAME: {circt.nonlocal = [[NLA1]], class = "port1"}]) {
  module @PortAnnotations0(in %a : !firrtl.uint<1> [{class = "port0"}]) {
    // CHECK: %bar_r = mem
    // CHECK-SAME: portAnnotations =
    // CHECK-SAME:  {circt.nonlocal = [[NLA0]], class = "mem0"},
    // CHECK-SAME:  {circt.nonlocal = [[NLA1]], class = "mem1"}
    %bar_r = mem Undefined  {depth = 16 : i64, name = "bar", portAnnotations = [[{class = "mem0"}]], portNames = ["r"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<8>>
  }
  // CHECK-NOT: module @PortAnnotations1
  module @PortAnnotations1(in %b : !firrtl.uint<1> [{class = "port1"}])  {
    %bar_r = mem Undefined  {depth = 16 : i64, name = "bar", portAnnotations = [[{class = "mem1"}]], portNames = ["r"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<8>>
  }
  // CHECK: module @PortAnnotations
  module @PortAnnotations() {
    %portannos0_in = instance portannos0 @PortAnnotations0(in a: !firrtl.uint<1>)
    %portannos1_in = instance portannos1 @PortAnnotations1(in b: !firrtl.uint<1>)
  }
}

// Non-local annotations should have their path updated and bread crumbs should
// not be turned into non-local annotations. Note that this should not create
// totally new NLAs for the annotations, it should just update the existing
// ones.
// CHECK-LABEL: circuit "Breadcrumb"
firrtl.circuit "Breadcrumb" {
  // CHECK:  @breadcrumb_nla0 [@Breadcrumb::@breadcrumb0, @Breadcrumb0::@crumb0, @Crumb::@in]
  hw.hierpath private @breadcrumb_nla0 [@Breadcrumb::@breadcrumb0, @Breadcrumb0::@crumb0, @Crumb::@in]
  // CHECK:  @breadcrumb_nla1 [@Breadcrumb::@breadcrumb1, @Breadcrumb0::@crumb0, @Crumb::@in]
  hw.hierpath private @breadcrumb_nla1 [@Breadcrumb::@breadcrumb1, @Breadcrumb1::@crumb1, @Crumb::@in]
  // CHECK:  @breadcrumb_nla2 [@Breadcrumb::@breadcrumb0, @Breadcrumb0::@crumb0, @Crumb::@w]
  hw.hierpath private @breadcrumb_nla2 [@Breadcrumb::@breadcrumb0, @Breadcrumb0::@crumb0, @Crumb::@w]
  // CHECK:  @breadcrumb_nla3 [@Breadcrumb::@breadcrumb1, @Breadcrumb0::@crumb0, @Crumb::@w]
  hw.hierpath private @breadcrumb_nla3 [@Breadcrumb::@breadcrumb1, @Breadcrumb1::@crumb1, @Crumb::@w]
  module @Crumb(in %in: !firrtl.uint<1> sym @in [
      {circt.nonlocal = @breadcrumb_nla0, class = "port0"},
      {circt.nonlocal = @breadcrumb_nla1, class = "port1"}]) {
    %w = wire sym @w {annotations = [
      {circt.nonlocal = @breadcrumb_nla2, class = "wire0"},
      {circt.nonlocal = @breadcrumb_nla3, class = "wire1"}]}: !firrtl.uint<1>
  }
  // CHECK: module @Breadcrumb0()
  module @Breadcrumb0() {
    // CHECK: %crumb0_in = instance crumb0 sym @crumb0
    %crumb_in = instance crumb0 sym @crumb0 @Crumb(in in : !firrtl.uint<1>)
  }
  // CHECK-NOT: module @Breadcrumb1()
  module @Breadcrumb1() {
    %crumb_in = instance crumb1 sym @crumb1 @Crumb(in in : !firrtl.uint<1>)
  }
  // CHECK: module @Breadcrumb()
  module @Breadcrumb() {
    instance breadcrumb0 sym @breadcrumb0 @Breadcrumb0()
    instance breadcrumb1 sym @breadcrumb1 @Breadcrumb1()
  }
}

// Non-local annotations should be updated with additional context if the module
// at the root of the NLA is deduplicated.  The original NLA should be deleted,
// and the annotation should be cloned for each parent of the root module.
// CHECK-LABEL: circuit "Context"
firrtl.circuit "Context" {
  // CHECK: hw.hierpath private [[NLA3:@nla.*]] [@Context::@context1, @Context0::@c0, @ContextLeaf::@w]
  // CHECK: hw.hierpath private [[NLA1:@nla.*]] [@Context::@context1, @Context0::@c0, @ContextLeaf::@in]
  // CHECK: hw.hierpath private [[NLA2:@nla.*]] [@Context::@context0, @Context0::@c0, @ContextLeaf::@w]
  // CHECK: hw.hierpath private [[NLA0:@nla.*]] [@Context::@context0, @Context0::@c0, @ContextLeaf::@in]
  // CHECK-NOT: @context_nla0
  // CHECK-NOT: @context_nla1
  // CHECK-NOT: @context_nla2
  // CHECK-NOT: @context_nla3
  hw.hierpath private @context_nla0 [@Context0::@c0, @ContextLeaf::@in]
  hw.hierpath private @context_nla1 [@Context0::@c0, @ContextLeaf::@w]
  hw.hierpath private @context_nla2 [@Context1::@c1, @ContextLeaf::@in]
  hw.hierpath private @context_nla3 [@Context1::@c1, @ContextLeaf::@w]

  // CHECK: module @ContextLeaf(in %in: !firrtl.uint<1> sym @in [
  // CHECK-SAME: {circt.nonlocal = [[NLA0]], class = "port0"},
  // CHECK-SAME: {circt.nonlocal = [[NLA1]], class = "port1"}]
  module @ContextLeaf(in %in : !firrtl.uint<1> sym @in [
      {circt.nonlocal = @context_nla0, class = "port0"},
      {circt.nonlocal = @context_nla2, class = "port1"}
    ]) {

    // CHECK: %w = wire sym @w  {annotations = [
    // CHECK-SAME: {circt.nonlocal = [[NLA2]], class = "fake0"}
    // CHECK-SAME: {circt.nonlocal = [[NLA3]], class = "fake1"}
    %w = wire sym @w {annotations = [
      {circt.nonlocal = @context_nla1, class = "fake0"},
      {circt.nonlocal = @context_nla3, class = "fake1"}]}: !firrtl.uint<3>
  }
  module @Context0() {
    // CHECK: %leaf_in = instance leaf sym @c0
    %leaf_in = instance leaf sym @c0 @ContextLeaf(in in : !firrtl.uint<1>)
  }
  // CHECK-NOT: module @Context1()
  module @Context1() {
    %leaf_in = instance leaf sym @c1 @ContextLeaf(in in : !firrtl.uint<1>)
  }
  module @Context() {
    // CHECK: instance context0 sym @context0
    instance context0 @Context0()
    // CHECK: instance context1 sym @context1
    instance context1 @Context1()
  }
}


// When an annotation is already non-local, and is copied over to another
// module, and in further dedups force us to add more context to the
// hierarchical path, the target of the annotation should be updated to use the
// new NLA.
// CHECK-LABEL: circuit "Context"
firrtl.circuit "Context" {

  // CHECK-NOT: hw.hierpath private @nla0
  hw.hierpath private @nla0 [@Context0::@leaf0, @ContextLeaf0::@w0]
  // CHECK-NOT: hw.hierpath private @nla1
  hw.hierpath private @nla1 [@Context1::@leaf1, @ContextLeaf1::@w1]

  // CHECK: hw.hierpath private [[NLA0:@.+]] [@Context::@context1, @Context0::@leaf0, @ContextLeaf0::@w0]
  // CHECK: hw.hierpath private [[NLA1:@.+]] [@Context::@context0, @Context0::@leaf0, @ContextLeaf0::@w0]

  // CHECK: module @ContextLeaf0()
  module @ContextLeaf0() {
    // CHECK: %w0 = wire sym @w0  {annotations = [
    // CHECK-SAME: {circt.nonlocal = [[NLA1]], class = "fake0"}
    // CHECK-SAME: {circt.nonlocal = [[NLA0]], class = "fake1"}]}
    %w0 = wire sym @w0 {annotations = [
      {circt.nonlocal = @nla0, class = "fake0"}]}: !firrtl.uint<3>
  }

  module @ContextLeaf1() {
    %w1 = wire sym @w1 {annotations = [
      {circt.nonlocal = @nla1, class = "fake1"}]}: !firrtl.uint<3>
  }

  module @Context0() {
    instance leaf0 sym @leaf0 @ContextLeaf0()
  }

  module @Context1() {
    instance leaf1 sym @leaf1 @ContextLeaf1()
  }

  module @Context() {
    instance context0 @Context0()
    instance context1 @Context1()
  }
}


// This is a larger version of the above test using 3 modules.
// CHECK-LABEL: circuit "DuplicateNLAs"
firrtl.circuit "DuplicateNLAs" {
  // CHECK-NOT: hw.hierpath private @annos_nla_1 [@Mid_1::@core, @Core_1]
  // CHECK-NOT: hw.hierpath private @annos_nla_2 [@Mid_2::@core, @Core_2]
  // CHECK-NOT: hw.hierpath private @annos_nla_3 [@Mid_3::@core, @Core_3]
  hw.hierpath private @annos_nla_1 [@Mid_1::@core, @Core_1]
  hw.hierpath private @annos_nla_2 [@Mid_2::@core, @Core_2]
  hw.hierpath private @annos_nla_3 [@Mid_3::@core, @Core_3]

  // CHECK: hw.hierpath private [[NLA0:@.+]] [@DuplicateNLAs::@core_3, @Mid_1::@core, @Core_1]
  // CHECK: hw.hierpath private [[NLA1:@.+]] [@DuplicateNLAs::@core_2, @Mid_1::@core, @Core_1]
  // CHECK: hw.hierpath private [[NLA2:@.+]] [@DuplicateNLAs::@core_1, @Mid_1::@core, @Core_1]

  module @DuplicateNLAs() {
    instance core_1 sym @core_1 @Mid_1()
    instance core_2 sym @core_2 @Mid_2()
    instance core_3 sym @core_3 @Mid_3()
  }

  module private @Mid_1() {
    instance core sym @core @Core_1()
  }

  module private @Mid_2() {
    instance core sym @core @Core_2()
  }

  module private @Mid_3() {
    instance core sym @core @Core_3()
  }

  // CHECK: module private @Core_1() attributes {annotations = [
  // CHECK-SAME: {circt.nonlocal = [[NLA2]], class = "SomeAnno1"}
  // CHECK-SAME: {circt.nonlocal = [[NLA1]], class = "SomeAnno2"}
  // CHECK-SAME: {circt.nonlocal = [[NLA0]], class = "SomeAnno3"}
  module private @Core_1() attributes {
    annotations = [
      {circt.nonlocal = @annos_nla_1, class = "SomeAnno1"}
    ]
  } { }

  module private @Core_2() attributes {
    annotations = [
      {circt.nonlocal = @annos_nla_2, class = "SomeAnno2"}
    ]
  } { }

  module private @Core_3() attributes {
    annotations = [
      {circt.nonlocal = @annos_nla_3, class = "SomeAnno3"}
    ]
  } { }
}

// External modules should dedup and fixup any NLAs.
// CHECK: circuit "ExtModuleTest"
firrtl.circuit "ExtModuleTest" {
  // CHECK: hw.hierpath private @ext_nla [@ExtModuleTest::@e1, @ExtMod0]
  hw.hierpath private @ext_nla [@ExtModuleTest::@e1, @ExtMod1]
  // CHECK: extmodule @ExtMod0() attributes {annotations = [{circt.nonlocal = @ext_nla}], defname = "a"}
  extmodule @ExtMod0() attributes {defname = "a"}
  // CHECK-NOT: extmodule @ExtMod1()
  extmodule @ExtMod1() attributes {annotations = [{circt.nonlocal = @ext_nla}], defname = "a"}
  module @ExtModuleTest() {
    // CHECK: instance e0  @ExtMod0()
    instance e0 @ExtMod0()
    // CHECK: instance e1 sym @e1 @ExtMod0()
    instance e1 sym @e1 @ExtMod1()
  }
}

// External modules with NLAs on ports should be properly rewritten.
// https://github.com/llvm/circt/issues/2713
// CHECK-LABEL: circuit "Foo"
firrtl.circuit "Foo"  {
  // CHECK: hw.hierpath private @nla_1 [@Foo::@b, @A::@a]
  hw.hierpath private @nla_1 [@Foo::@b, @B::@b]
  // CHECK: extmodule @A(out a: !firrtl.clock sym @a [{circt.nonlocal = @nla_1}])
  extmodule @A(out a: !firrtl.clock)
  extmodule @B(out b: !firrtl.clock sym @b [{circt.nonlocal = @nla_1}])
  module @Foo() {
    %b0_out = instance a @A(out a: !firrtl.clock)
    // CHECK: instance b sym @b  @A(out a: !firrtl.clock)
    %b1_out = instance b sym @b @B(out b: !firrtl.clock)
  }
}

// Extmodules should properly hash port types and not dedup when they differ.
// CHECK-LABEL: circuit "Foo"
firrtl.circuit "Foo"  {
  // CHECK: extmodule @Bar
  extmodule @Bar(
    in clock: !firrtl.clock, out io: !firrtl.bundle<a: clock>)
  // CHECK: extmodule @Baz
  extmodule @Baz(
    in clock: !firrtl.clock, out io: !firrtl.bundle<a flip: uint<1>, b flip: uint<16>, c: uint<1>>)
  module @Foo() {
    %bar_clock, %bar_io = instance bar @Bar(
      in clock: !firrtl.clock, out io: !firrtl.bundle<a: clock>)
    %baz_clock, %baz_io = instance baz @Baz(
      in clock: !firrtl.clock, out io: !firrtl.bundle<a flip: uint<1>, b flip: uint<16>, c: uint<1>>)
  }
}

// As we dedup modules, the chain on NLAs should continuously grow.
// CHECK-LABEL: circuit "Chain"
firrtl.circuit "Chain" {
  // CHECK: hw.hierpath private [[NLA1:@nla.*]] [@Chain::@chainB1, @ChainB0::@chainA0, @ChainA0::@extchain0, @ExtChain0]
  // CHECK: hw.hierpath private [[NLA0:@nla.*]] [@Chain::@chainB0, @ChainB0::@chainA0, @ChainA0::@extchain0, @ExtChain0]
  // CHECK: module @ChainB0()
  module @ChainB0() {
    instance chainA0 @ChainA0()
  }
  // CHECK: extmodule @ExtChain0() attributes {annotations = [
  // CHECK-SAME:  {circt.nonlocal = [[NLA0]], class = "0"},
  // CHECK-SAME:  {circt.nonlocal = [[NLA1]], class = "1"}], defname = "ExtChain"}
  extmodule @ExtChain0() attributes {annotations = [{class = "0"}], defname = "ExtChain"}
  // CHECK-NOT: extmodule @ExtChain1()
  extmodule @ExtChain1() attributes {annotations = [{class = "1"}], defname = "ExtChain"}
  // CHECK: module @ChainA0()
  module @ChainA0()  {
    instance extchain0 @ExtChain0()
  }
  // CHECK-NOT: module @ChainB1()
  module @ChainB1() {
    instance chainA1 @ChainA1()
  }
  // CHECK-NOT: module @ChainA1()
  module @ChainA1()  {
    instance extchain1 @ExtChain1()
  }
  module @Chain() {
    // CHECK: instance chainB0 sym @chainB0 @ChainB0()
    instance chainB0 @ChainB0()
    // CHECK: instance chainB1 sym @chainB1 @ChainB0()
    instance chainB1 @ChainB1()
  }
}


// Check that we fixup subfields and connects, when an
// instance op starts returning a different bundle type.
// CHECK-LABEL: circuit "Bundle"
firrtl.circuit "Bundle" {
  // CHECK: module @Bundle0
  module @Bundle0(out %a: !firrtl.bundle<b: bundle<c flip: uint<1>, d: uint<1>>>) { }
  // CHECK-NOT: module @Bundle1
  module @Bundle1(out %e: !firrtl.bundle<f: bundle<g flip: uint<1>, h: uint<1>>>) { }
  module @Bundle() {
    // CHECK: instance bundle0  @Bundle0
    %a = instance bundle0 @Bundle0(out a: !firrtl.bundle<b: bundle<c flip: uint<1>, d: uint<1>>>)

    // CHECK: instance bundle1  @Bundle0
    // CHECK: %a = wire : !firrtl.bundle<f: bundle<g flip: uint<1>, h: uint<1>>>
    // CHECK: [[A_F:%.+]] = subfield %a[f]
    // CHECK: [[A_B:%.+]] = subfield %bundle1_a[b]
    // CHECK: [[A_F_G:%.+]] = subfield %0[g]
    // CHECK: [[A_B_C:%.+]] = subfield %1[c]
    // CHECK: strictconnect [[A_B_C]], [[A_F_G]]
    // CHECK: [[A_F_H:%.+]] = subfield [[A_F]][h]
    // CHECK: [[A_B_D:%.+]] = subfield [[A_B]][d]
    // CHECK: strictconnect [[A_F_H]], [[A_B_D]]
    %e = instance bundle1 @Bundle1(out e: !firrtl.bundle<f: bundle<g flip: uint<1>, h: uint<1>>>)

    // CHECK: [[B:%.+]] = subfield %bundle0_a[b]
    %b = subfield %a[b] : !firrtl.bundle<b: bundle<c flip: uint<1>, d: uint<1>>>

    // CHECK: [[F:%.+]] = subfield %a[f]
    %f = subfield %e[f] : !firrtl.bundle<f: bundle<g flip: uint<1>, h: uint<1>>>

    // Check that we properly fixup connects when the field names change.
    %w0 = wire : !firrtl.bundle<g flip: uint<1>, h: uint<1>>

    // CHECK: connect %w0, [[F]]
    connect %w0, %f : !firrtl.bundle<g flip: uint<1>, h: uint<1>>, !firrtl.bundle<g flip: uint<1>, h: uint<1>>
  }
}

// CHECK-LABEL: circuit "MuxBundle"
firrtl.circuit "MuxBundle" {
  module @Bar0(out %o: !firrtl.bundle<a: uint<1>>) {
    %invalid = invalidvalue : !firrtl.bundle<a: uint<1>>
    strictconnect %o, %invalid : !firrtl.bundle<a: uint<1>>
  }
  module @Bar1(out %o: !firrtl.bundle<b: uint<1>>) {
    %invalid = invalidvalue : !firrtl.bundle<b: uint<1>>
    strictconnect %o, %invalid : !firrtl.bundle<b: uint<1>>
  }
  module @MuxBundle(in %p: !firrtl.uint<1>, in %l: !firrtl.bundle<b: uint<1>>, out %o: !firrtl.bundle<b: uint<1>>) attributes {convention = #firrtl<convention scalarized>} {
    // CHECK: %bar0_o = instance bar0 @Bar0(out o: !firrtl.bundle<a: uint<1>>)
    %bar0_o = instance bar0 @Bar0(out o: !firrtl.bundle<a: uint<1>>)

    // CHECK: %bar1_o = instance bar1 @Bar0(out o: !firrtl.bundle<a: uint<1>>)
    // CHECK: [[WIRE:%.+]] = wire {name = "o"} : !firrtl.bundle<b: uint<1>>
    // CHECK: [[WIRE_B:%.+]] = subfield [[WIRE]][b]
    // CHECK: [[PORT_A:%.+]] = subfield %bar1_o[a]
    // CHECK: strictconnect [[WIRE_B]], [[PORT_A]]
    %bar1_o = instance bar1 @Bar1(out o: !firrtl.bundle<b: uint<1>>)

    // CHECK: %2 = mux(%p, [[WIRE]], %l)
    // CHECK: strictconnect %o, %2 : !firrtl.bundle<b: uint<1>>
    %0 = mux(%p, %bar1_o, %l) : (!firrtl.uint<1>, !firrtl.bundle<b: uint<1>>, !firrtl.bundle<b: uint<1>>) -> !firrtl.bundle<b: uint<1>>
    strictconnect %o, %0 : !firrtl.bundle<b: uint<1>>
  }
}

// Make sure flipped fields are handled properly. This should pass flow
// verification checking.
// CHECK-LABEL: circuit "Flip"
firrtl.circuit "Flip" {
  module @Flip0(out %io: !firrtl.bundle<foo flip: uint<1>, fuzz: uint<1>>) {
    %0 = subfield %io[foo] : !firrtl.bundle<foo flip: uint<1>, fuzz: uint<1>>
    %1 = subfield %io[fuzz] : !firrtl.bundle<foo flip: uint<1>, fuzz: uint<1>>
    connect %1, %0 : !firrtl.uint<1>, !firrtl.uint<1>
  }
  module @Flip1(out %io: !firrtl.bundle<bar flip: uint<1>, buzz: uint<1>>) {
    %0 = subfield %io[bar] : !firrtl.bundle<bar flip: uint<1>, buzz: uint<1>>
    %1 = subfield %io[buzz] : !firrtl.bundle<bar flip: uint<1>, buzz: uint<1>>
    connect %1, %0 : !firrtl.uint<1>, !firrtl.uint<1>
  }
  module @Flip(out %io: !firrtl.bundle<foo: bundle<foo flip: uint<1>, fuzz: uint<1>>, bar: bundle<bar flip: uint<1>, buzz: uint<1>>>) {
    %0 = subfield %io[bar] : !firrtl.bundle<foo: bundle<foo flip: uint<1>, fuzz: uint<1>>, bar: bundle<bar flip: uint<1>, buzz: uint<1>>>
    %1 = subfield %io[foo] : !firrtl.bundle<foo: bundle<foo flip: uint<1>, fuzz: uint<1>>, bar: bundle<bar flip: uint<1>, buzz: uint<1>>>
    %foo_io = instance foo  @Flip0(out io: !firrtl.bundle<foo flip: uint<1>, fuzz: uint<1>>)
    %bar_io = instance bar  @Flip1(out io: !firrtl.bundle<bar flip: uint<1>, buzz: uint<1>>)
    connect %1, %foo_io : !firrtl.bundle<foo flip: uint<1>, fuzz: uint<1>>, !firrtl.bundle<foo flip: uint<1>, fuzz: uint<1>>
    connect %0, %bar_io : !firrtl.bundle<bar flip: uint<1>, buzz: uint<1>>, !firrtl.bundle<bar flip: uint<1>, buzz: uint<1>>
  }
}

// This is checking that the fixup phase due to changing bundle names does not
// block the deduplication of parent modules.
// CHECK-LABEL: circuit "DelayedFixup"
firrtl.circuit "DelayedFixup"  {
  // CHECK: extmodule @Foo
  extmodule @Foo(out a: !firrtl.bundle<a: uint<1>>)
  // CHECK-NOT: extmodule @Bar
  extmodule @Bar(out b: !firrtl.bundle<b: uint<1>>)
  // CHECK: module @Parent0
  module @Parent0(out %a: !firrtl.bundle<a: uint<1>>, out %b: !firrtl.bundle<b: uint<1>>) {
    %foo_a = instance foo  @Foo(out a: !firrtl.bundle<a: uint<1>>)
    connect %a, %foo_a : !firrtl.bundle<a: uint<1>>, !firrtl.bundle<a: uint<1>>
    %bar_b = instance bar  @Bar(out b: !firrtl.bundle<b: uint<1>>)
    connect %b, %bar_b : !firrtl.bundle<b: uint<1>>, !firrtl.bundle<b: uint<1>>
  }
  // CHECK-NOT: module @Parent1
  module @Parent1(out %a: !firrtl.bundle<a: uint<1>>, out %b: !firrtl.bundle<b: uint<1>>) {
    %foo_a = instance foo  @Foo(out a: !firrtl.bundle<a: uint<1>>)
    connect %a, %foo_a : !firrtl.bundle<a: uint<1>>, !firrtl.bundle<a: uint<1>>
    %bar_b = instance bar  @Bar(out b: !firrtl.bundle<b: uint<1>>)
    connect %b, %bar_b : !firrtl.bundle<b: uint<1>>, !firrtl.bundle<b: uint<1>>
  }
  module @DelayedFixup() {
    // CHECK: instance parent0  @Parent0
    %parent0_a, %parent0_b = instance parent0  @Parent0(out a: !firrtl.bundle<a: uint<1>>, out b: !firrtl.bundle<b: uint<1>>)
    // CHECK: instance parent1  @Parent0
    %parent1_a, %parent1_b = instance parent1  @Parent1(out a: !firrtl.bundle<a: uint<1>>, out b: !firrtl.bundle<b: uint<1>>)
  }
}

// Don't attach empty annotations onto ops without annotations.
// CHECK-LABEL: circuit "NoEmptyAnnos"
firrtl.circuit "NoEmptyAnnos" {
  // CHECK-LABEL: @NoEmptyAnnos0()
  module @NoEmptyAnnos0() {
    // CHECK: %w = wire  : !firrtl.bundle<a: uint<1>>
    // CHECK: %0 = subfield %w[a] : !firrtl.bundle<a: uint<1>>
    %w = wire : !firrtl.bundle<a: uint<1>>
    %0 = subfield %w[a] : !firrtl.bundle<a: uint<1>>
  }
  module @NoEmptyAnnos1() {
    %w = wire : !firrtl.bundle<a: uint<1>>
    %0 = subfield %w[a] : !firrtl.bundle<a: uint<1>>
  }
  module @NoEmptyAnnos() {
    instance empty0 @NoEmptyAnnos0()
    instance empty1 @NoEmptyAnnos1()
  }
}


// Don't deduplicate modules with NoDedup.
// CHECK-LABEL: circuit "NoDedup"
firrtl.circuit "NoDedup" {
  module @Simple0() { }
  module @Simple1() attributes {annotations = [{class = "firrtl.transforms.NoDedupAnnotation"}]} { }
  // CHECK: module @NoDedup
  module @NoDedup() {
    instance simple0 @Simple0()
    instance simple1 @Simple1()
  }
}

// Don't deduplicate modules with input RefType ports.
// CHECK-LABEL:   circuit "InputRefTypePorts"
// CHECK-COUNT-3: module
firrtl.circuit "InputRefTypePorts" {
  module @Foo(in %a: !firrtl.probe<uint<1>>) {}
  module @Bar(in %a: !firrtl.probe<uint<1>>) {}
  module @InputRefTypePorts() {
    %foo_a = instance foo @Foo(in a: !firrtl.probe<uint<1>>)
    %bar_a = instance bar @Bar(in a: !firrtl.probe<uint<1>>)
  }
}

// Check that modules marked MustDedup have been deduped.
// CHECK-LABEL: circuit "MustDedup"
firrtl.circuit "MustDedup" attributes {annotations = [{
    // The annotation should be removed.
    // CHECK-NOT: class = "firrtl.transforms.MustDeduplicateAnnotation"
    class = "firrtl.transforms.MustDeduplicateAnnotation",
    modules = ["~MustDedup|Simple0", "~MustDedup|Simple1"]}]
   } {
  // CHECK: @Simple0
  module @Simple0() { }
  // CHECK-NOT: @Simple1
  module @Simple1() { }
  // CHECK: module @MustDedup
  module @MustDedup() {
    instance simple0 @Simple0()
    instance simple1 @Simple1()
  }
}

// Check that the following doesn't crash.
// https://github.com/llvm/circt/issues/3360
firrtl.circuit "Foo"  {
  module private @X() { }
  module private @Y() { }
  module @Foo() {
    instance x0 @X()
    instance y0 @Y()
    instance y1 @Y()
  }
}

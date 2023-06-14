// RUN: circt-opt -pass-pipeline='builtin.module(firrtl.circuit(firrtl-inject-dut-hier))' -split-input-file %s | FileCheck %s

// CHECK-LABEL: circuit "Top"
firrtl.circuit "Top" attributes {
    annotations = [{class = "sifive.enterprise.firrtl.InjectDUTHierarchyAnnotation", name = "Foo"}]
  } {
  // CHECK:      module private @Foo()
  //
  // CHECK:      module private @DUT
  // CHECK-SAME:   class = "sifive.enterprise.firrtl.MarkDUTAnnotation"
  //
  // CHECK-NEXT:   instance Foo {{.+}} @Foo()
  // CHECK-NEXT: }
  module private @DUT() attributes {annotations = [{class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}]} {}

  // CHECK:      module @Top
  // CHECK-NEXT:   instance dut @DUT
  module @Top() {
    instance dut @DUT()
  }
}

// -----

// CHECK-LABEL: circuit "NLARenaming"
firrtl.circuit "NLARenaming" attributes {
    annotations = [{class = "sifive.enterprise.firrtl.InjectDUTHierarchyAnnotation", name = "Foo"}]
  } {
  // An NLA that is rooted at the DUT moves to the wrapper.
  //
  // CHECK:      hw.hierpath private @nla_DUTRoot [@Foo::@sub, @Sub::@a]
  hw.hierpath private @nla_DUTRoot [@DUT::@sub, @Sub::@a]

  // NLAs that end at the DUT or a DUT port are unmodified.
  //
  // CHECK-NEXT: hw.hierpath private @[[nla_DUTLeafModule_clone:.+]] [@NLARenaming::@dut, @DUT]
  // CHECK-NEXT: hw.hierpath private @nla_DUTLeafModule [@NLARenaming::@dut, @DUT::@Foo, @Foo]
  // CHECK-NEXT: hw.hierpath private @nla_DUTLeafPort [@NLARenaming::@dut, @DUT::@in]
  hw.hierpath private @nla_DUTLeafModule [@NLARenaming::@dut, @DUT]
  hw.hierpath private @nla_DUTLeafPort [@NLARenaming::@dut, @DUT::@in]

  // NLAs that end inside the DUT get an extra level of hierarchy.
  //
  // CHECK-NEXT: hw.hierpath private @nla_DUTLeafWire [@NLARenaming::@dut, @DUT::@[[inst_sym:.+]], @Foo::@w]
  hw.hierpath private @nla_DUTLeafWire [@NLARenaming::@dut, @DUT::@w]

  // An NLA that passes through the DUT gets an extra level of hierarchy.
  //
  // CHECK-NEXT: hw.hierpath private @nla_DUTPassthrough [@NLARenaming::@dut, @DUT::@[[inst_sym:.+]], @Foo::@sub, @Sub]
  hw.hierpath private @nla_DUTPassthrough [@NLARenaming::@dut, @DUT::@sub, @Sub]
  module private @Sub() attributes {annotations = [{circt.nonlocal = @nla_DUTPassthrough, class = "nla_DUTPassthrough"}]} {
    %a = wire sym @a : !firrtl.uint<1>
  }

  // CHECK:      module private @Foo
  // CHECK:      module private @DUT
  // CHECK-SAME:   {circt.nonlocal = @[[nla_DUTLeafModule_clone]], class = "nla_DUTLeafModule"}
  // CHECK-NEXT    instance Foo sym @[[inst_sym]]
  module private @DUT(
    in %in: !firrtl.uint<1> sym @in [{circt.nonlocal = @nla_DUTLeafPort, class = "nla_DUTLeafPort"}]
  ) attributes {
    annotations = [
      {class = "sifive.enterprise.firrtl.MarkDUTAnnotation"},
      {circt.nonlocal = @nla_DUTLeafModule, class = "nla_DUTLeafModule"}]}
  {
    %w = wire sym @w {
      annotations = [
        {circt.nonlocal = @nla_DUTPassthrough, class = "nla_DUT_LeafWire"}]
    } : !firrtl.uint<1>
    instance sub sym @sub @Sub()
  }
  module @NLARenaming() {
    %dut_in = instance dut sym @dut @DUT(in in: !firrtl.uint<1>)
  }
}

// -----

// CHECK-LABEL: circuit "NLARenamingNewNLAs"
firrtl.circuit "NLARenamingNewNLAs" attributes {
    annotations = [{class = "sifive.enterprise.firrtl.InjectDUTHierarchyAnnotation", name = "Foo"}]
  } {
  // An NLA that is rooted at the DUT moves to the wrapper.
  //
  // CHECK:      hw.hierpath private @nla_DUTRoot [@Foo::@sub, @Sub]
  // CHECK:      hw.hierpath private @nla_DUTRootRef [@Foo::@sub, @Sub::@a]
  hw.hierpath private @nla_DUTRoot [@DUT::@sub, @Sub]
  hw.hierpath private @nla_DUTRootRef [@DUT::@sub, @Sub::@a]

  // NLAs that end at the DUT or a DUT port are unmodified.  These should not be
  // cloned unless they have users.
  //
  // CHECK-NEXT: hw.hierpath private @nla_DUTLeafModule[[_:.+]] [@NLARenamingNewNLAs::@dut, @DUT]
  // CHECK-NEXT: hw.hierpath private @nla_DUTLeafModule [@NLARenamingNewNLAs::@dut, @DUT::@Foo, @Foo]
  // CHECK-NEXT: hw.hierpath private @[[nla_DUTLeafPort_clone:.+]] [@NLARenamingNewNLAs::@dut, @DUT]
  // CHECK-NEXT: hw.hierpath private @nla_DUTLeafPort [@NLARenamingNewNLAs::@dut, @DUT::@Foo, @Foo]
  hw.hierpath private @nla_DUTLeafModule [@NLARenamingNewNLAs::@dut, @DUT]
  hw.hierpath private @nla_DUTLeafPort [@NLARenamingNewNLAs::@dut, @DUT]

  // NLAs that end at the DUT are moved to a cloned path.  NLAs that end inside
  // the DUT keep the old path symbol which gets the added hierarchy.
  //
  // CHECK-NEXT: hw.hierpath private @nla_DUTLeafWire [@NLARenamingNewNLAs::@dut, @DUT::@[[inst_sym:.+]], @Foo]
  hw.hierpath private @nla_DUTLeafWire [@NLARenamingNewNLAs::@dut, @DUT]

  // An NLA that passes through the DUT gets an extra level of hierarchy.
  //
  // CHECK-NEXT: hw.hierpath private @nla_DUTPassthrough [@NLARenamingNewNLAs::@dut, @DUT::@[[inst_sym]], @Foo::@sub, @Sub]
  hw.hierpath private @nla_DUTPassthrough [@NLARenamingNewNLAs::@dut, @DUT::@sub, @Sub]
  module private @Sub() attributes {annotations = [{circt.nonlocal = @nla_DUTPassthrough, class = "nla_DUTPassthrough"}]} {
    %a = wire sym @a : !firrtl.uint<1>
  }

  // CHECK:      module private @Foo
  // CHECK-NEXT:   %w = wire
  // CHECK-SAME:     {annotations = [{circt.nonlocal = @nla_DUTLeafWire, class = "nla_DUT_LeafWire"}]}

  // CHECK:      module private @DUT
  // CHECK-SAME:   in %in{{.+}} [{circt.nonlocal = @[[nla_DUTLeafPort_clone]], class = "nla_DUTLeafPort"}]
  // CHECK-NEXT    instance Foo sym @[[inst_sym]]
  module private @DUT(
    in %in: !firrtl.uint<1> [{circt.nonlocal = @nla_DUTLeafPort, class = "nla_DUTLeafPort"}]
  ) attributes {
    annotations = [
      {class = "sifive.enterprise.firrtl.MarkDUTAnnotation"},
      {circt.nonlocal = @nla_DUTLeafModule, class = "nla_DUTLeafModule"}]}
  {
    %w = wire {
      annotations = [
        {circt.nonlocal = @nla_DUTLeafWire, class = "nla_DUT_LeafWire"}]
    } : !firrtl.uint<1>
    instance sub sym @sub @Sub()
  }
  module @NLARenamingNewNLAs() {
    %dut_in = instance dut sym @dut @DUT(in in: !firrtl.uint<1>)
  }
}

// -----

// CHECK-LABEL: circuit "Refs"
firrtl.circuit "Refs" attributes {
    annotations = [{class = "sifive.enterprise.firrtl.InjectDUTHierarchyAnnotation", name = "Foo"}]
  } {

  module private @DUT(
    in %in: !firrtl.uint<1>, out %out: !firrtl.ref<uint<1>>
  ) attributes {
    annotations = [
      {class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}
    ]}
  {
    %ref = ref.send %in : !firrtl.uint<1>
    ref.define %out, %ref : !firrtl.ref<uint<1>>
  }
  module @Refs() {
    %dut_in, %dut_tap = instance dut sym @dut @DUT(in in: !firrtl.uint<1>, out out: !firrtl.ref<uint<1>>)
  }
}

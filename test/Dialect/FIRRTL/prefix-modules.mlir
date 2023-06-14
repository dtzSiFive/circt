// RUN: circt-opt --pass-pipeline="builtin.module(firrtl.circuit(firrtl-prefix-modules))" %s | FileCheck %s

// Check that the circuit is updated when the main module is updated.
// CHECK: circuit "T_Top"
firrtl.circuit "Top" {
  // CHECK: module @T_Top
  module @Top()
    attributes {annotations = [{
      class = "sifive.enterprise.firrtl.NestedPrefixModulesAnnotation",
      prefix = "T_",
      inclusive = true
    }]} {
  }
}


// Check that the circuit is not updated if the annotation is non-inclusive.
// CHECK: circuit "Top"
firrtl.circuit "Top" {
  // CHECK: module @Top
  module @Top()
    attributes {annotations = [{
      class = "sifive.enterprise.firrtl.NestedPrefixModulesAnnotation",
      prefix = "T_",
      inclusive = false
    }]} {
  }
}


// Check that basic module prefixing is working.
firrtl.circuit "Top" {
  // The annotation should be removed.
  // CHECK:  module @Top() {
  module @Top()
    attributes {annotations = [{
      class = "sifive.enterprise.firrtl.NestedPrefixModulesAnnotation",
      prefix = "T_",
      inclusive = false
    }]} {
    instance test @Zebra()
  }

  // CHECK: module @T_Zebra
  // CHECK-NOT: module @Zebra
  module @Zebra() { }
}


// Check that memories are renamed.
firrtl.circuit "Top" {
  module @Top()
    attributes {annotations = [{
      class = "sifive.enterprise.firrtl.NestedPrefixModulesAnnotation",
      prefix = "T_",
      inclusive = true
    }]} {
    // CHECK: mem
    // CHECK-SAME: name = "ram1"
    // CHECK-SAME: prefix = "T_"
    %ram1_r = mem Undefined {depth = 256 : i64, name = "ram1", portNames = ["r"], readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<8>, en: uint<1>, clk: clock, data flip: uint<1>>
    // CHECK: mem
    // CHECK-SAME: name = "ram2"
    // CHECK-SAME: prefix = "T_foo_"
    %ram2_r = mem Undefined {depth = 256 : i64, name = "ram2", portNames = ["r"], prefix = "foo_", readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<8>, en: uint<1>, clk: clock, data flip: uint<1>>
  }
}

// Check that memory modules are renamed.
// CHECK-LABEL: circuit "MemModule"
firrtl.circuit "MemModule" {
  // CHECK: memmodule @T_MWrite_ext
  memmodule @MWrite_ext(in W0_addr: !firrtl.uint<4>, in W0_en: !firrtl.uint<1>, in W0_clk: !firrtl.clock, in W0_data: !firrtl.uint<42>) attributes {dataWidth = 42 : ui32, depth = 12 : ui64, extraPorts = [], maskBits = 1 : ui32, numReadPorts = 0 : ui32, numReadWritePorts = 0 : ui32, numWritePorts = 1 : ui32, readLatency = 1 : ui32, writeLatency = 1 : ui32}
  module @MemModule()
    attributes {annotations = [{
      class = "sifive.enterprise.firrtl.NestedPrefixModulesAnnotation",
      prefix = "T_",
      inclusive = false
    }]} {
    // CHECK: instance MWrite_ext  @T_MWrite_ext
    %0:4 = instance MWrite_ext  @MWrite_ext(in W0_addr: !firrtl.uint<4>, in W0_en: !firrtl.uint<1>, in W0_clk: !firrtl.clock, in W0_data: !firrtl.uint<42>)
  }
}

// Check that external modules are not renamed.
// CHECK: circuit "T_Top"
firrtl.circuit "Top" {
  // CHECK: extmodule @ExternalModule
  extmodule @ExternalModule()

  // CHECK: module @T_Top
  module @Top()
    attributes {annotations = [{
      class = "sifive.enterprise.firrtl.NestedPrefixModulesAnnotation",
      prefix = "T_",
      inclusive = true
    }]} {

    instance ext @ExternalModule()
  }
}


// Check that the module is not cloned more than necessary.
firrtl.circuit "Top0" {
  module @Top0()
    attributes {annotations = [{
      class = "sifive.enterprise.firrtl.NestedPrefixModulesAnnotation",
      prefix = "T_",
      inclusive = false
    }]} {
    instance test @Zebra()
  }

  module @Top1()
    attributes {annotations = [{
      class = "sifive.enterprise.firrtl.NestedPrefixModulesAnnotation",
      prefix = "T_",
      inclusive = false
    }]} {
    instance test @Zebra()
  }

  // CHECK: module @T_Zebra
  // CHECK-NOT: module @T_Zebra
  // CHECK-NOT: module @Zebra
  module @Zebra() { }
}


// Complex nested test.
// CHECK: circuit "T_Top"
firrtl.circuit "Top" {
  // CHECK: module @T_Top
  module @Top()
    attributes {annotations = [{
      class = "sifive.enterprise.firrtl.NestedPrefixModulesAnnotation",
      prefix = "T_",
      inclusive = true
    }]} {

    // CHECK: instance test @T_Aardvark()
    instance test @Aardvark()

    // CHECK: instance test @T_Z_Zebra()
    instance test @Zebra()
  }

  // CHECK: module @T_Aardvark
  module @Aardvark()
    attributes {annotations = [{
      class = "sifive.enterprise.firrtl.NestedPrefixModulesAnnotation",
      prefix = "A_",
      inclusive = false
    }]} {

    // CHECK: instance test @T_A_Z_Zebra()
    instance test @Zebra()
  }

  // CHECK: module @T_Z_Zebra
  // CHECK: module @T_A_Z_Zebra
  module @Zebra()
    attributes {annotations = [{
      class = "sifive.enterprise.firrtl.NestedPrefixModulesAnnotation",
      prefix = "Z_",
      inclusive = true
    }]} {
  }
}


// Updates should be made to a Grand Central interface to add a "prefix" field.
// The annotatinos associated with the parent and companion should be
// unmodified.
// CHECK-LABEL: circuit "GCTInterfacePrefix"
// CHECK-SAME:    name = "MyView", prefix = "FOO_"
firrtl.circuit "GCTInterfacePrefix"
  attributes {annotations = [{
    class = "sifive.enterprise.grandcentral.AugmentedBundleType",
    defName = "MyInterface",
    elements = [],
    id = 0 : i64,
    name = "MyView"}]}  {
  // CHECK:      module @FOO_MyView_companion
  // CHECK-SAME:   name = "MyView"
  module @MyView_companion()
    attributes {annotations = [{
      class = "sifive.enterprise.grandcentral.ViewAnnotation.companion",
      id = 0 : i64,
      name = "MyView",
      type = "companion"}]} {}
  // CHECK:      module @FOO_DUT
  // CHECK-SAME:   name = "MyView"
  module @DUT()
    attributes {annotations = [
      {class = "sifive.enterprise.grandcentral.ViewAnnotation.parent",
       id = 0 : i64,
       name = "MyView",
       type = "parent"},
      {class = "sifive.enterprise.firrtl.NestedPrefixModulesAnnotation",
       prefix = "FOO_",
       inclusive = true}]} {
    instance MyView_companion  @MyView_companion()
  }
  module @GCTInterfacePrefix() {
    instance dut @DUT()
  }
}

// CHECK: circuit "T_NLATop"
firrtl.circuit "NLATop" {

  hw.hierpath private @nla [@NLATop::@test, @Aardvark::@test, @Zebra]
  hw.hierpath private @nla_1 [@NLATop::@test,@Aardvark::@test_1, @Zebra]
  // CHECK: hw.hierpath private @nla [@T_NLATop::@test, @T_Aardvark::@test, @T_A_Z_Zebra]
  // CHECK: hw.hierpath private @nla_1 [@T_NLATop::@test, @T_Aardvark::@test_1, @T_A_Z_Zebra]
  // CHECK: module @T_NLATop
  module @NLATop()
    attributes {annotations = [{
      class = "sifive.enterprise.firrtl.NestedPrefixModulesAnnotation",
      prefix = "T_",
      inclusive = true
    }]} {

    // CHECK:  instance test sym @test @T_Aardvark()
    instance test  sym @test @Aardvark()

    // CHECK: instance test2 @T_Z_Zebra()
    instance test2 @Zebra()
  }

  // CHECK: module @T_Aardvark
  module @Aardvark()
    attributes {annotations = [{
      class = "sifive.enterprise.firrtl.NestedPrefixModulesAnnotation",
      prefix = "A_",
      inclusive = false
    }]} {

    // CHECK:  instance test sym @test @T_A_Z_Zebra()
    instance test sym @test @Zebra()
    instance test1 sym @test_1 @Zebra()
  }

  // CHECK: module @T_Z_Zebra
  // CHECK: module @T_A_Z_Zebra
  module @Zebra()
    attributes {annotations = [{
      class = "sifive.enterprise.firrtl.NestedPrefixModulesAnnotation",
      prefix = "Z_",
      inclusive = true
    }]} {
  }
}

// Prefixes should be applied to Grand Central Data or Mem taps.  Check that a
// multiply instantiated Data/Mem tap is cloned ("duplicated" in Scala FIRRTL
// Compiler terminology) if needed.  (Note: multiply instantiated taps are
// completely untrodden territory for Grand Central.  However, the behavior here
// is the exact same as how normal modules are cloned.)
//
// CHECK-LABLE: circuit "GCTDataMemTapsPrefix"
firrtl.circuit "GCTDataMemTapsPrefix" {
  // CHECK:      extmodule @FOO_DataTap
  // CHECK-SAME:   defname = "FOO_DataTap"
  extmodule @DataTap()
    attributes {annotations = [{
      class = "sifive.enterprise.grandcentral.DataTapsAnnotation.blackbox"}],
      defname = "DataTap"}
  // The Mem tap should be prefixed with "FOO_" and cloned to create a copy
  // prefixed with "BAR_".
  //
  // CHECK:      extmodule @FOO_MemTap
  // CHECK-SAME:   defname = "FOO_MemTap"
  // CHECK:      extmodule @BAR_MemTap
  // CHECK-SAME:   defname = "BAR_MemTap"
  extmodule @MemTap(
    out mem: !firrtl.vector<uint<1>, 1>
      [{
        circt.fieldID = 1 : i32,
        class = "sifive.enterprise.grandcentral.MemTapAnnotation.port",
        id = 0 : i64,
        word = 0 : i64}])
    attributes {defname = "MemTap"}
  // Module DUT has a "FOO_" prefix.
  module @DUT()
    attributes {annotations = [
      {class = "sifive.enterprise.firrtl.NestedPrefixModulesAnnotation",
       prefix = "FOO_",
       inclusive = true}]} {
    // CHECK: instance d @FOO_DataTap
    instance d @DataTap()
    // CHECK: instance m @FOO_MemTap
    %a = instance m @MemTap(out mem: !firrtl.vector<uint<1>, 1>)
  }
  // Module DUT2 has a "BAR_" prefix.
  module @DUT2()
    attributes {annotations = [
      {class = "sifive.enterprise.firrtl.NestedPrefixModulesAnnotation",
       prefix = "BAR_",
       inclusive = true}]} {
    // CHECK: instance m @BAR_MemTap
    %a = instance m @MemTap(out mem: !firrtl.vector<uint<1>, 1>)
  }
  module @GCTDataMemTapsPrefix() {
    instance dut @DUT()
    instance dut @DUT2()
  }
}

// Test the NonLocalAnchor is properly updated.
// CHECK-LABEL: circuit "FixNLA" {
firrtl.circuit "FixNLA"   {
  hw.hierpath private @nla_1 [@FixNLA::@bar, @Bar::@baz, @Baz]
  // CHECK:   hw.hierpath private @nla_1 [@FixNLA::@bar, @Bar::@baz, @Baz]
  hw.hierpath private @nla_2 [@FixNLA::@foo, @Foo::@bar, @Bar::@baz, @Baz::@s1]
  // CHECK:   hw.hierpath private @nla_2 [@FixNLA::@foo, @X_Foo::@bar, @X_Bar::@baz, @X_Baz::@s1]
  hw.hierpath private @nla_3 [@FixNLA::@bar, @Bar::@baz, @Baz]
  // CHECK:   hw.hierpath private @nla_3 [@FixNLA::@bar, @Bar::@baz, @Baz]
  hw.hierpath private @nla_4 [@Foo::@bar, @Bar::@baz, @Baz]
  // CHECK:       hw.hierpath private @nla_4 [@X_Foo::@bar, @X_Bar::@baz, @X_Baz]
  // CHECK-LABEL: module @FixNLA()
  module @FixNLA() {
    instance foo sym @foo  @Foo()
    instance bar sym @bar  @Bar()
    // CHECK:   instance foo sym @foo @X_Foo()
    // CHECK:   instance bar sym @bar @Bar()
  }
  module @Foo() attributes {annotations = [{class = "sifive.enterprise.firrtl.NestedPrefixModulesAnnotation", inclusive = true, prefix = "X_"}]} {
    instance bar sym @bar  @Bar()
  }
  // CHECK-LABEL:   module @X_Foo()
  // CHECK:         instance bar sym @bar @X_Bar()

  // CHECK-LABEL:   module @Bar()
  module @Bar() {
    instance baz sym @baz @Baz()
    // CHECK:     instance baz sym @baz @Baz()
  }
  // CHECK-LABEL: module @X_Bar()
  // CHECK:       instance baz sym @baz @X_Baz()

  module @Baz() attributes {annotations = [{circt.nonlocal = @nla_1, class = "nla_1"}, {circt.nonlocal = @nla_3, class = "nla_3"}, {circt.nonlocal = @nla_4, class = "nla_4"}]} {
    %mem_MPORT_en = wire sym @s1  {annotations = [{circt.nonlocal = @nla_2, class = "nla_2"}]} : !firrtl.uint<1>
  }
  // CHECK-LABEL: module @X_Baz()
  // CHECK-SAME:  annotations = [{circt.nonlocal = @nla_4, class = "nla_4"}]
  // CHECK:       %mem_MPORT_en = wire sym @s1  {annotations = [{circt.nonlocal = @nla_2, class = "nla_2"}]} : !firrtl.uint<1>
  // CHECK:       module @Baz()
  // CHECK-SAME:  annotations = [{circt.nonlocal = @nla_1, class = "nla_1"}, {circt.nonlocal = @nla_3, class = "nla_3"}]
  // CHECK:       %mem_MPORT_en = wire sym @s1  : !firrtl.uint<1>
}

// Test that NonLocalAnchors are properly updated with memmodules.
firrtl.circuit "Test"   {
  // CHECK: hw.hierpath private @nla_1 [@Test::@foo1, @A_Foo1::@bar, @A_Bar]
  hw.hierpath private @nla_1 [@Test::@foo1, @Foo1::@bar, @Bar]
  // CHECK: hw.hierpath private @nla_2 [@Test::@foo2, @B_Foo2::@bar, @B_Bar]
  hw.hierpath private @nla_2 [@Test::@foo2, @Foo2::@bar, @Bar]

  module @Test() {
    instance foo1 sym @foo1 @Foo1()
    instance foo2 sym @foo2 @Foo2()
  }

  module @Foo1() attributes {annotations = [{class = "sifive.enterprise.firrtl.NestedPrefixModulesAnnotation", inclusive = true, prefix = "A_"}]} {
    instance bar sym @bar @Bar()
  }

  module @Foo2() attributes {annotations = [{class = "sifive.enterprise.firrtl.NestedPrefixModulesAnnotation", inclusive = true, prefix = "B_"}]} {
    instance bar sym @bar @Bar()
  }

  // CHECK: memmodule @A_Bar() attributes {annotations = [{circt.nonlocal = @nla_1, class = "test1"}]
  // CHECK: memmodule @B_Bar() attributes {annotations = [{circt.nonlocal = @nla_2, class = "test2"}]
  memmodule @Bar() attributes {annotations = [{circt.nonlocal = @nla_1, class = "test1"}, {circt.nonlocal = @nla_2, class = "test2"}], dataWidth = 1 : ui32, depth = 16 : ui64, extraPorts = [], maskBits = 0 : ui32, numReadPorts = 0 : ui32, numReadWritePorts = 0 : ui32, numWritePorts = 0 : ui32, readLatency = 0 : ui32,  writeLatency = 1 : ui32}
}

// Test that the MarkDUTAnnotation receives a prefix.
// CHECK-LABEL: circuit "Prefix_MarkDUTAnnotationGetsPrefix"
firrtl.circuit "MarkDUTAnnotationGetsPrefix" {
  // CHECK-NEXT: module @Prefix_MarkDUTAnnotationGetsPrefix
  // CHECK-SAME:   class = "sifive.enterprise.firrtl.MarkDUTAnnotation", prefix = "Prefix_"
  module @MarkDUTAnnotationGetsPrefix() attributes {
    annotations = [
     {
       class = "sifive.enterprise.firrtl.MarkDUTAnnotation"
     },
     {
       class = "sifive.enterprise.firrtl.NestedPrefixModulesAnnotation",
       prefix = "Prefix_",
       inclusive = true
     }
    ]
  } {}
}


// Test that inner name refs are properly adjusted.
firrtl.circuit "RewriteInnerNameRefs" {
  // CHECK-LABEL: module @Prefix_RewriteInnerNameRefs
  module @RewriteInnerNameRefs() attributes {
    annotations = [
     {
       class = "sifive.enterprise.firrtl.NestedPrefixModulesAnnotation",
       prefix = "Prefix_",
       inclusive = true
     }
    ]
  } {
    %wire = wire sym @wire : !firrtl.uint<1>
    instance nested @Nested()

    // CHECK: #hw.innerNameRef<@Prefix_RewriteInnerNameRefs::@wire>
    sv.verbatim "{{0}}" {symbols=[#hw.innerNameRef<@RewriteInnerNameRefs::@wire>]}

    // CHECK: #hw.innerNameRef<@Prefix_RewriteInnerNameRefs::@wire>
    // CHECK: #hw.innerNameRef<@Prefix_Nested::@wire>
    sv.verbatim "{{0}} {{1}}" {symbols=[
      #hw.innerNameRef<@RewriteInnerNameRefs::@wire>,
      #hw.innerNameRef<@Nested::@wire>
    ]}
  }

  module @Nested() {
    %wire = wire sym @wire : !firrtl.uint<1>
  }
}

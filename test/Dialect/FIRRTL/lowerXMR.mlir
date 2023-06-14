// RUN: circt-opt %s  --firrtl-lower-xmr -split-input-file |  FileCheck %s

// Test for same module lowering
// CHECK-LABEL: circuit "xmr"
firrtl.circuit "xmr" {
  // CHECK : #hw.innerNameRef<@xmr::@[[wSym]]>
  // CHECK-LABEL: module @xmr(out %o: !firrtl.uint<2>)
  module @xmr(out %o: !firrtl.uint<2>, in %2: !firrtl.probe<uint<0>>) {
    %w = wire : !firrtl.uint<2>
    %1 = ref.send %w : !firrtl.uint<2>
    %x = ref.resolve %1 : !firrtl.probe<uint<2>>
    %x2 = ref.resolve %2 : !firrtl.probe<uint<0>>
    // CHECK-NOT: ref.resolve
    strictconnect %o, %x : !firrtl.uint<2>
    // CHECK:      %w = wire sym @[[wSym:[a-zA-Z0-9_]+]] : !firrtl.uint<2>
    // CHECK-NEXT: %[[#xmr:]] = sv.xmr.ref @xmrPath : !hw.inout<i2>
    // CHECK-NEXT: %[[#cast:]] = builtin.unrealized_conversion_cast %[[#xmr]] : !hw.inout<i2> to !firrtl.uint<2>
    // CHECK:      strictconnect %o, %[[#cast]] : !firrtl.uint<2>
  }
}

// -----

// Test the correct xmr path is generated
// CHECK-LABEL: circuit "Top" {
firrtl.circuit "Top" {
  // CHECK:      hw.hierpath private @[[path:[a-zA-Z0-9_]+]]
  // CHECK-SAME:   [@Top::@bar, @Bar::@barXMR, @XmrSrcMod::@[[xmrSym:[a-zA-Z0-9_]+]]]
  module @XmrSrcMod(out %_a: !firrtl.probe<uint<1>>) {
    // CHECK: module @XmrSrcMod() {
    %zero = constant 0 : !firrtl.uint<1>
    // CHECK:  %c0_ui1 = constant 0 : !firrtl.uint<1>
    // CHECK:  %0 = node sym @[[xmrSym]] %c0_ui1  : !firrtl.uint<1>
    %1 = ref.send %zero : !firrtl.uint<1>
    ref.define %_a, %1 : !firrtl.probe<uint<1>>
  }
  module @Bar(out %_a: !firrtl.probe<uint<1>>) {
    %xmr   = instance bar sym @barXMR @XmrSrcMod(out _a: !firrtl.probe<uint<1>>)
    // CHECK:  instance bar sym @barXMR  @XmrSrcMod()
    ref.define %_a, %xmr   : !firrtl.probe<uint<1>>
  }
  module @Top() {
    %bar_a = instance bar sym @bar  @Bar(out _a: !firrtl.probe<uint<1>>)
    // CHECK:  instance bar sym @bar  @Bar()
    %a = wire : !firrtl.uint<1>
    %0 = ref.resolve %bar_a : !firrtl.probe<uint<1>>
    // CHECK:      %[[#xmr:]] = sv.xmr.ref @[[path]] : !hw.inout<i1>
    // CHECK-NEXT: %[[#cast:]] = builtin.unrealized_conversion_cast %[[#xmr]] : !hw.inout<i1> to !firrtl.uint<1>
    // CHECK-NEXT; strictconnect %a, %[[#cast]] : !firrtl.uint<1>
    strictconnect %a, %0 : !firrtl.uint<1>
  }
}

// -----

// Test 0-width xmrs are handled
// CHECK-LABEL: circuit "Top" {
firrtl.circuit "Top" {
  module @Top(in %bar_a : !firrtl.probe<uint<0>>, in %bar_b : !firrtl.probe<vector<uint<0>,10>>) {
    %a = wire : !firrtl.uint<0>
    %0 = ref.resolve %bar_a : !firrtl.probe<uint<0>>
    // CHECK:  %[[c0_ui0:.+]] = constant 0 : !firrtl.uint<0>
    strictconnect %a, %0 : !firrtl.uint<0>
    // CHECK:  strictconnect %a, %[[c0_ui0]] : !firrtl.uint<0>
    %b = wire : !firrtl.vector<uint<0>,10>
    %1 = ref.resolve %bar_b : !firrtl.probe<vector<uint<0>,10>>
    strictconnect %b, %1 : !firrtl.vector<uint<0>,10>
    // CHECK:	%[[c0_ui0_0:.+]] = constant 0 : !firrtl.uint<0>
    // CHECK:  %[[v2:.+]] = bitcast %[[c0_ui0_0]] : (!firrtl.uint<0>) -> !firrtl.vector<uint<0>, 10>
    // CHECK:  strictconnect %b, %[[v2]] : !firrtl.vector<uint<0>, 10>
  }
}

// -----

// Test the correct xmr path to port is generated
// CHECK-LABEL: circuit "Top" {
firrtl.circuit "Top" {
  // CHECK: hw.hierpath private @[[path:[a-zA-Z0-9_]+]] [@Top::@bar, @Bar::@barXMR, @XmrSrcMod::@[[xmrSym:[a-zA-Z0-9_]+]]]
  module @XmrSrcMod(in %pa: !firrtl.uint<1>, out %_a: !firrtl.probe<uint<1>>) {
    // CHECK: module @XmrSrcMod(in %pa: !firrtl.uint<1> sym @[[xmrSym]]) {
    %1 = ref.send %pa : !firrtl.uint<1>
    ref.define %_a, %1 : !firrtl.probe<uint<1>>
  }
  module @Bar(out %_a: !firrtl.probe<uint<1>>) {
    %pa, %xmr   = instance bar sym @barXMR @XmrSrcMod(in pa: !firrtl.uint<1>, out _a: !firrtl.probe<uint<1>>)
    // CHECK: %bar_pa = instance bar sym @barXMR  @XmrSrcMod(in pa: !firrtl.uint<1>)
    ref.define %_a, %xmr   : !firrtl.probe<uint<1>>
  }
  module @Top() {
    %bar_a = instance bar sym @bar  @Bar(out _a: !firrtl.probe<uint<1>>)
    // CHECK:  instance bar sym @bar  @Bar()
    %a = wire : !firrtl.uint<1>
    %0 = ref.resolve %bar_a : !firrtl.probe<uint<1>>
    // CHECK:      %[[#xmr:]] = sv.xmr.ref @[[path]] : !hw.inout<i1>
    // CHECK-NEXT: %[[#cast:]] = builtin.unrealized_conversion_cast %[[#xmr]] : !hw.inout<i1> to !firrtl.uint<1>
    strictconnect %a, %0 : !firrtl.uint<1>
    // CHECK-NEXT: strictconnect %a, %[[#cast]]
  }
}

// -----

// Test for multiple readers and multiple instances
// CHECK-LABEL: circuit "Top" {
firrtl.circuit "Top" {
  // CHECK-DAG: hw.hierpath private @[[path_0:[a-zA-Z0-9_]+]] [@Foo::@fooXMR, @XmrSrcMod::@[[xmrSym:[a-zA-Z0-9_]+]]]
  // CHECK-DAG: hw.hierpath private @[[path_1:[a-zA-Z0-9_]+]] [@Bar::@barXMR, @XmrSrcMod::@[[xmrSym]]]
  // CHECK-DAG: hw.hierpath private @[[path_2:[a-zA-Z0-9_]+]] [@Top::@bar, @Bar::@barXMR, @XmrSrcMod::@[[xmrSym]]]
  // CHECK-DAG: hw.hierpath private @[[path_3:[a-zA-Z0-9_]+]] [@Top::@foo, @Foo::@fooXMR, @XmrSrcMod::@[[xmrSym]]]
  // CHECK-DAG: hw.hierpath private @[[path_4:[a-zA-Z0-9_]+]] [@Top::@xmr, @XmrSrcMod::@[[xmrSym]]]
  module @XmrSrcMod(out %_a: !firrtl.probe<uint<1>>) {
    // CHECK: module @XmrSrcMod() {
    %zero = constant 0 : !firrtl.uint<1>
    // CHECK:   %c0_ui1 = constant 0
    // CHECK:  %0 = node sym @[[xmrSym]] %c0_ui1  : !firrtl.uint<1>
    %1 = ref.send %zero : !firrtl.uint<1>
    ref.define %_a, %1 : !firrtl.probe<uint<1>>
  }
  module @Foo(out %_a: !firrtl.probe<uint<1>>) {
    %xmr   = instance bar sym @fooXMR @XmrSrcMod(out _a: !firrtl.probe<uint<1>>)
    // CHECK:  instance bar sym @fooXMR  @XmrSrcMod()
    ref.define %_a, %xmr   : !firrtl.probe<uint<1>>
    %0 = ref.resolve %xmr   : !firrtl.probe<uint<1>>
    // CHECK:      %[[#xmr:]] = sv.xmr.ref @[[path_0]]
    // CHECK-NEXT: %[[#cast:]] = builtin.unrealized_conversion_cast %[[#xmr]]
    %a = wire : !firrtl.uint<1>
    strictconnect %a, %0 : !firrtl.uint<1>
    // CHECK:      strictconnect %a, %[[#cast]]
  }
  module @Bar(out %_a: !firrtl.probe<uint<1>>) {
    %xmr   = instance bar sym @barXMR @XmrSrcMod(out _a: !firrtl.probe<uint<1>>)
    // CHECK:  instance bar sym @barXMR  @XmrSrcMod()
    ref.define %_a, %xmr   : !firrtl.probe<uint<1>>
    %0 = ref.resolve %xmr   : !firrtl.probe<uint<1>>
    // CHECK:      %[[#xmr:]] = sv.xmr.ref @[[path_1]]
    // CHECK-NEXT: %[[#cast:]] = builtin.unrealized_conversion_cast %[[#xmr]]
    %a = wire : !firrtl.uint<1>
    strictconnect %a, %0 : !firrtl.uint<1>
    // CHECK:      strictconnect %a, %[[#cast]]
  }
  module @Top() {
    %bar_a = instance bar sym @bar  @Bar(out _a: !firrtl.probe<uint<1>>)
    %foo_a = instance foo sym @foo @Foo(out _a: !firrtl.probe<uint<1>>)
    %xmr_a = instance xmr sym @xmr @XmrSrcMod(out _a: !firrtl.probe<uint<1>>)
    // CHECK:  instance bar sym @bar  @Bar()
    // CHECK:  instance foo sym @foo  @Foo()
    // CHECK:  instance xmr sym @xmr  @XmrSrcMod()
    %a = wire : !firrtl.uint<1>
    %b = wire : !firrtl.uint<1>
    %c = wire : !firrtl.uint<1>
    %0 = ref.resolve %bar_a : !firrtl.probe<uint<1>>
    // CHECK:      %[[#xmr:]] = sv.xmr.ref @[[path_2]]
    // CHECK-NEXT: %[[#cast_2:]] = builtin.unrealized_conversion_cast %[[#xmr]]
    %1 = ref.resolve %foo_a : !firrtl.probe<uint<1>>
    // CHECK:      %[[#xmr:]] = sv.xmr.ref @[[path_3]]
    // CHECK-NEXT: %[[#cast_3:]] = builtin.unrealized_conversion_cast %[[#xmr]]
    %2 = ref.resolve %xmr_a : !firrtl.probe<uint<1>>
    // CHECK:      %[[#xmr:]] = sv.xmr.ref @[[path_4]]
    // CHECK-NEXT: %[[#cast_4:]] = builtin.unrealized_conversion_cast %[[#xmr]]
    strictconnect %a, %0 : !firrtl.uint<1>
    // CHECK-NEXT: strictconnect %a, %[[#cast_2]]
    strictconnect %b, %1 : !firrtl.uint<1>
    // CHECK-NEXT: strictconnect %b, %[[#cast_3]]
    strictconnect %c, %2 : !firrtl.uint<1>
    // CHECK-NEXT: strictconnect %c, %[[#cast_4]]
  }
}

// -----

// Check for downward reference
// CHECK-LABEL: circuit "Top" {
firrtl.circuit "Top" {
  // CHECK: hw.hierpath private @[[path:[a-zA-Z0-9_]+]] [@Top::@bar, @Bar::@barXMR, @XmrSrcMod::@[[xmrSym:[a-zA-Z0-9_]+]]]
  module @XmrSrcMod(out %_a: !firrtl.probe<uint<1>>) {
    // CHECK: module @XmrSrcMod() {
    %zero = constant 0 : !firrtl.uint<1>
    // CHECK:  %c0_ui1 = constant 0
    // CHECK:  %0 = node sym @[[xmrSym]] %c0_ui1  : !firrtl.uint<1>
    %1 = ref.send %zero : !firrtl.uint<1>
    ref.define %_a, %1 : !firrtl.probe<uint<1>>
  }
  module @Bar(out %_a: !firrtl.probe<uint<1>>) {
    %xmr   = instance bar sym @barXMR @XmrSrcMod(out _a: !firrtl.probe<uint<1>>)
    // CHECK:  instance bar sym @barXMR  @XmrSrcMod()
    ref.define %_a, %xmr   : !firrtl.probe<uint<1>>
  }
  module @Top() {
    %bar_a = instance bar sym @bar  @Bar(out _a: !firrtl.probe<uint<1>>)
    // CHECK:  instance bar sym @bar  @Bar()
    %a = wire : !firrtl.uint<1>
    %0 = ref.resolve %bar_a : !firrtl.probe<uint<1>>
    // CHECK:      %[[#xmr:]] = sv.xmr.ref @[[path]]
    // CHECK-NEXT: %[[#cast:]] = builtin.unrealized_conversion_cast %[[#xmr]]
    strictconnect %a, %0 : !firrtl.uint<1>
    // CHECK-NEXT: strictconnect %a, %[[#cast]]
    %c_a = instance child @Child(in  _a: !firrtl.probe<uint<1>>)
    ref.define %c_a, %bar_a : !firrtl.probe<uint<1>>
  }
  module @Child(in  %_a: !firrtl.probe<uint<1>>) {
    %0 = ref.resolve %_a : !firrtl.probe<uint<1>>
    // CHECK:      %[[#xmr:]] = sv.xmr.ref @[[path]]
    // CHECK-NEXT: %[[#cast:]] = builtin.unrealized_conversion_cast %[[#xmr]]
  }

}

// -----

// Check for downward reference to port
// CHECK-LABEL: circuit "Top" {
firrtl.circuit "Top" {
  // CHECK: hw.hierpath private @[[path:[a-zA-Z0-9_]+]] [@Top::@bar, @Bar::@barXMR, @XmrSrcMod::@[[xmrSym:[a-zA-Z0-9_]+]]]
  module @XmrSrcMod(in %pa: !firrtl.uint<1>, out %_a: !firrtl.probe<uint<1>>) {
    // CHECK: module @XmrSrcMod(in %pa: !firrtl.uint<1> sym @xmr_sym) {
    %1 = ref.send %pa : !firrtl.uint<1>
    ref.define %_a, %1 : !firrtl.probe<uint<1>>
  }
  module @Bar(out %_a: !firrtl.probe<uint<1>>) {
    %pa, %xmr   = instance bar sym @barXMR @XmrSrcMod(in pa: !firrtl.uint<1>, out _a: !firrtl.probe<uint<1>>)
    // CHECK: %bar_pa = instance bar sym @barXMR  @XmrSrcMod(in pa: !firrtl.uint<1>)
    ref.define %_a, %xmr   : !firrtl.probe<uint<1>>
  }
  module @Top() {
    %bar_a = instance bar sym @bar  @Bar(out _a: !firrtl.probe<uint<1>>)
    // CHECK:  instance bar sym @bar  @Bar()
    %a = wire : !firrtl.uint<1>
    %0 = ref.resolve %bar_a : !firrtl.probe<uint<1>>
    // CHECK:      %[[#xmr:]] = sv.xmr.ref @[[path]]
    // CHECK-NEXT: %[[#cast:]] = builtin.unrealized_conversion_cast %[[#xmr]]
    strictconnect %a, %0 : !firrtl.uint<1>
    // CHECK-NEXT: strictconnect %a, %[[#cast]]
    %c_a = instance child @Child(in  _a: !firrtl.probe<uint<1>>)
    ref.define %c_a, %bar_a : !firrtl.probe<uint<1>>
  }
  module @Child(in  %_a: !firrtl.probe<uint<1>>) {
    %0 = ref.resolve %_a : !firrtl.probe<uint<1>>
    // CHECK:      %[[#xmr:]] = sv.xmr.ref @[[path]]
    // CHECK-NEXT: %[[#cast:]] = builtin.unrealized_conversion_cast %[[#xmr]]
  }
}

// -----

// Test for multiple paths and downward reference.
// CHECK-LABEL: circuit "Top" {
firrtl.circuit "Top" {
  // CHECK: hw.hierpath private @[[path_0:[a-zA-Z0-9_]+]] [@Top::@foo, @Foo::@fooXMR, @XmrSrcMod::@[[xmrSym:[a-zA-Z0-9_]+]]]
  // CHECK: hw.hierpath private @[[path_1:[a-zA-Z0-9_]+]] [@Top::@xmr, @XmrSrcMod::@[[xmrSym]]]
  module @XmrSrcMod(out %_a: !firrtl.probe<uint<1>>) {
    %zero = constant 0 : !firrtl.uint<1>
    %1 = ref.send %zero : !firrtl.uint<1>
    ref.define %_a, %1 : !firrtl.probe<uint<1>>
    // CHECK: node sym @[[xmrSym]]
  }
  module @Foo(out %_a: !firrtl.probe<uint<1>>) {
    %xmr   = instance bar sym @fooXMR @XmrSrcMod(out _a: !firrtl.probe<uint<1>>)
    ref.define %_a, %xmr   : !firrtl.probe<uint<1>>
  }
  module @Top() {
    %foo_a = instance foo sym @foo @Foo(out _a: !firrtl.probe<uint<1>>)
    %xmr_a = instance xmr sym @xmr @XmrSrcMod(out _a: !firrtl.probe<uint<1>>)
    %c_a, %c_b = instance child @Child2p(in _a: !firrtl.probe<uint<1>>, in _b: !firrtl.probe<uint<1>> )
    // CHECK:  instance child  @Child2p()
    ref.define %c_a, %foo_a : !firrtl.probe<uint<1>>
    ref.define %c_b, %xmr_a : !firrtl.probe<uint<1>>
  }
  module @Child2p(in  %_a: !firrtl.probe<uint<1>>, in  %_b: !firrtl.probe<uint<1>>) {
    %0 = ref.resolve %_a : !firrtl.probe<uint<1>>
    // CHECK:      %[[#xmr:]] = sv.xmr.ref @[[path_0]]
    // CHECK-NEXT: %[[#cast:]] = builtin.unrealized_conversion_cast %[[#xmr]]
    %1 = ref.resolve %_b : !firrtl.probe<uint<1>>
    // CHECK:      %[[#xmr:]] = sv.xmr.ref @[[path_1]]
    // CHECK-NEXT: %[[#cast:]] = builtin.unrealized_conversion_cast %[[#xmr]]
  }
}

// -----

// Test for multiple children paths
// CHECK-LABEL: circuit "Top" {
firrtl.circuit "Top" {
  // CHECK: hw.hierpath private @[[path:[a-zA-Z0-9_]+]] [@Top::@xmr, @XmrSrcMod::@[[xmrSym:[a-zA-Z0-9_]+]]]
  module @XmrSrcMod(out %_a: !firrtl.probe<uint<1>>) {
    %zero = constant 0 : !firrtl.uint<1>
    %1 = ref.send %zero : !firrtl.uint<1>
    ref.define %_a, %1 : !firrtl.probe<uint<1>>
    // CHECK: node sym @[[xmrSym]]
  }
  module @Top() {
    %xmr_a = instance xmr sym @xmr @XmrSrcMod(out _a: !firrtl.probe<uint<1>>)
    %c_a = instance child @Child1(in _a: !firrtl.probe<uint<1>>)
    ref.define %c_a, %xmr_a : !firrtl.probe<uint<1>>
  }
  // CHECK-LABEL: module @Child1() {
  module @Child1(in  %_a: !firrtl.probe<uint<1>>) {
    %0 = ref.resolve %_a : !firrtl.probe<uint<1>>
    // CHECK:      %[[#xmr:]] = sv.xmr.ref @[[path]]
    // CHECK-NEXT: %[[#cast:]] = builtin.unrealized_conversion_cast %[[#xmr]]
    %c_a, %c_b = instance child @Child2(in _a: !firrtl.probe<uint<1>>, in _b: !firrtl.probe<uint<1>> )
    ref.define %c_a, %_a : !firrtl.probe<uint<1>>
    ref.define %c_b, %_a : !firrtl.probe<uint<1>>
    %c3 = instance child @Child3(in _a: !firrtl.probe<uint<1>>)
    ref.define %c3 , %_a : !firrtl.probe<uint<1>>
  }
  module @Child2(in  %_a: !firrtl.probe<uint<1>>, in  %_b: !firrtl.probe<uint<1>>) {
    %0 = ref.resolve %_a : !firrtl.probe<uint<1>>
    // CHECK:      %[[#xmr:]] = sv.xmr.ref @[[path]]
    // CHECK-NEXT: %[[#cast:]] = builtin.unrealized_conversion_cast %[[#xmr]]
    %1 = ref.resolve %_b : !firrtl.probe<uint<1>>
    // CHECK:      %[[#xmr:]] = sv.xmr.ref @[[path]]
    // CHECK-NEXT: %[[#cast:]] = builtin.unrealized_conversion_cast %[[#xmr]]
  }
  module @Child3(in  %_a: !firrtl.probe<uint<1>>) {
    %0 = ref.resolve %_a : !firrtl.probe<uint<1>>
    // CHECK:      %[[#xmr:]] = sv.xmr.ref @[[path]]
    // CHECK-NEXT: %[[#cast:]] = builtin.unrealized_conversion_cast %[[#xmr]]
    %1 = ref.resolve %_a : !firrtl.probe<uint<1>>
    // CHECK:      %[[#xmr:]] = sv.xmr.ref @[[path]]
    // CHECK-NEXT: %[[#cast:]] = builtin.unrealized_conversion_cast %[[#xmr]]
  }
}

// -----

// Test for multiple children paths
// CHECK-LABEL: circuit "Top" {
firrtl.circuit "Top" {
  // CHECK: hw.hierpath private @[[path:[a-zA-Z0-9_]+]] [@Top::@xmr, @XmrSrcMod::@[[xmrSym:[a-zA-Z0-9_]+]]]
  module @XmrSrcMod(out %_a: !firrtl.probe<uint<1>>) {
    %zero = constant 0 : !firrtl.uint<1>
    %1 = ref.send %zero : !firrtl.uint<1>
    ref.define %_a, %1 : !firrtl.probe<uint<1>>
    // CHECK: node sym @[[xmrSym]]
  }
  module @Top() {
    %xmr_a = instance xmr sym @xmr @XmrSrcMod(out _a: !firrtl.probe<uint<1>>)
    %c_a = instance child @Child1(in _a: !firrtl.probe<uint<1>>)
    ref.define %c_a, %xmr_a : !firrtl.probe<uint<1>>
  }
  // CHECK-LABEL: module @Child1() {
  module @Child1(in  %_a: !firrtl.probe<uint<1>>) {
    %0 = ref.resolve %_a : !firrtl.probe<uint<1>>
    // CHECK:      %[[#xmr:]] = sv.xmr.ref @[[path]]
    // CHECK-NEXT: %[[#cast:]] = builtin.unrealized_conversion_cast %[[#xmr]]
    %c_a, %c_b = instance child @Child2(in _a: !firrtl.probe<uint<1>>, in _b: !firrtl.probe<uint<1>> )
    ref.define %c_a, %_a : !firrtl.probe<uint<1>>
    ref.define %c_b, %_a : !firrtl.probe<uint<1>>
    %c3 = instance child @Child3(in _a: !firrtl.probe<uint<1>>)
    ref.define %c3 , %_a : !firrtl.probe<uint<1>>
  }
  module @Child2(in  %_a: !firrtl.probe<uint<1>>, in  %_b: !firrtl.probe<uint<1>>) {
    %0 = ref.resolve %_a : !firrtl.probe<uint<1>>
    // CHECK:      %[[#xmr:]] = sv.xmr.ref @[[path]]
    // CHECK-NEXT: %[[#cast:]] = builtin.unrealized_conversion_cast %[[#xmr]]
    %1 = ref.resolve %_b : !firrtl.probe<uint<1>>
    // CHECK:      %[[#xmr:]] = sv.xmr.ref @[[path]]
    // CHECK-NEXT: %[[#cast:]] = builtin.unrealized_conversion_cast %[[#xmr]]
  }
  module @Child3(in  %_a: !firrtl.probe<uint<1>>) {
    %0 = ref.resolve %_a : !firrtl.probe<uint<1>>
    // CHECK:      %[[#xmr:]] = sv.xmr.ref @[[path]]
    // CHECK-NEXT: %[[#cast:]] = builtin.unrealized_conversion_cast %[[#xmr]]
    %1 = ref.resolve %_a : !firrtl.probe<uint<1>>
    // CHECK:      %[[#xmr:]] = sv.xmr.ref @[[path]]
    // CHECK-NEXT: %[[#cast:]] = builtin.unrealized_conversion_cast %[[#xmr]]
  }
}

// -----

// Multiply instantiated Top works, because the reference port does not flow through it.
firrtl.circuit "Top" {
  // CHECK: hw.hierpath private @[[path:[a-zA-Z0-9_]+]] [@Dut::@xmr, @XmrSrcMod::@[[xmrSym:[a-zA-Z0-9_]+]]]
  module @XmrSrcMod(out %_a: !firrtl.probe<uint<1>>) {
    %zero = constant 0 : !firrtl.uint<1>
    %1 = ref.send %zero : !firrtl.uint<1>
    ref.define %_a, %1 : !firrtl.probe<uint<1>>
    // CHECK: node sym @[[xmrSym]]
  }
  module @Top() {
    instance d1 @Dut()
  }
  module @Top2() {
    instance d2 @Dut()
  }
  module @Dut() {
    %xmr_a = instance xmr sym @xmr @XmrSrcMod(out _a: !firrtl.probe<uint<1>>)
    %c_a = instance child @Child1(in _a: !firrtl.probe<uint<1>>)
    ref.define %c_a, %xmr_a : !firrtl.probe<uint<1>>
  }
  // CHECK-LABEL: module @Child1() {
  module @Child1(in  %_a: !firrtl.probe<uint<1>>) {
    %0 = ref.resolve %_a : !firrtl.probe<uint<1>>
    // CHECK:      %[[#xmr:]] = sv.xmr.ref @[[path]]
    // CHECK-NEXT: %[[#cast:]] = builtin.unrealized_conversion_cast %[[#xmr]]
    %c_a, %c_b = instance child @Child2(in _a: !firrtl.probe<uint<1>>, in _b: !firrtl.probe<uint<1>> )
    ref.define %c_a, %_a : !firrtl.probe<uint<1>>
    ref.define %c_b, %_a : !firrtl.probe<uint<1>>
    %c3 = instance child @Child3(in _a: !firrtl.probe<uint<1>>)
    ref.define %c3 , %_a : !firrtl.probe<uint<1>>
  }
  module @Child2(in  %_a: !firrtl.probe<uint<1>>, in  %_b: !firrtl.probe<uint<1>>) {
    %0 = ref.resolve %_a : !firrtl.probe<uint<1>>
    // CHECK:      %[[#xmr:]] = sv.xmr.ref @[[path]]
    // CHECK-NEXT: %[[#cast:]] = builtin.unrealized_conversion_cast %[[#xmr]]
    %1 = ref.resolve %_b : !firrtl.probe<uint<1>>
    // CHECK:      %[[#xmr:]] = sv.xmr.ref @[[path]]
    // CHECK-NEXT: %[[#cast:]] = builtin.unrealized_conversion_cast %[[#xmr]]
  }
  module @Child3(in  %_a: !firrtl.probe<uint<1>>) {
    %0 = ref.resolve %_a : !firrtl.probe<uint<1>>
    // CHECK:      %[[#xmr:]] = sv.xmr.ref @[[path]]
    // CHECK-NEXT: %[[#cast:]] = builtin.unrealized_
    %1 = ref.resolve %_a : !firrtl.probe<uint<1>>
    // CHECK:      %[[#xmr:]] = sv.xmr.ref @[[path]]
    // CHECK-NEXT: %[[#cast:]] = builtin.unrealized_
  }
}

// -----

firrtl.circuit "Top"  {
  // CHECK: hw.hierpath private @[[path:[a-zA-Z0-9_]+]] [@Top::@xmr_sym, @DUTModule::@[[xmrSym:[a-zA-Z0-9_]+]]]
  // CHECK-LABEL: module private @DUTModule
  // CHECK-SAME: (in %clock: !firrtl.clock, in %io_addr: !firrtl.uint<3>, in %io_dataIn: !firrtl.uint<8>, in %io_wen: !firrtl.uint<1>, out %io_dataOut: !firrtl.uint<8>)
  module private @DUTModule(in %clock: !firrtl.clock, in %io_addr: !firrtl.uint<3>, in %io_dataIn: !firrtl.uint<8>, in %io_wen: !firrtl.uint<1>, out %io_dataOut: !firrtl.uint<8>, out %_gen_memTap: !firrtl.probe<vector<uint<8>, 8>>) attributes {annotations = [{class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}]} {
    %c1_ui1 = constant 1 : !firrtl.uint<1>
    %rf_memTap, %rf_read, %rf_write = mem  Undefined  {depth = 8 : i64, name = "rf", portNames = ["memTap", "read", "write"], prefix = "foo_", readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.probe<vector<uint<8>, 8>>, !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data flip: uint<8>>, !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>
    // CHECK:  %rf_read, %rf_write = mem sym @xmr_sym  Undefined  {depth = 8 : i64, name = "rf", portNames = ["read", "write"], prefix = "foo_", readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data flip: uint<8>>, !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>
    %0 = subfield %rf_read[addr] : !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data flip: uint<8>>
    %1 = subfield %rf_read[en] : !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data flip: uint<8>>
    %2 = subfield %rf_read[clk] : !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data flip: uint<8>>
    %3 = subfield %rf_read[data] : !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data flip: uint<8>>
    %4 = subfield %rf_write[addr] : !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>
    %5 = subfield %rf_write[en] : !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>
    %6 = subfield %rf_write[clk] : !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>
    %7 = subfield %rf_write[data] : !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>
    %8 = subfield %rf_write[mask] : !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>
    strictconnect %0, %io_addr : !firrtl.uint<3>
    strictconnect %1, %c1_ui1 : !firrtl.uint<1>
    strictconnect %2, %clock : !firrtl.clock
    strictconnect %io_dataOut, %3 : !firrtl.uint<8>
    strictconnect %4, %io_addr : !firrtl.uint<3>
    strictconnect %5, %io_wen : !firrtl.uint<1>
    strictconnect %6, %clock : !firrtl.clock
    strictconnect %8, %c1_ui1 : !firrtl.uint<1>
    strictconnect %7, %io_dataIn : !firrtl.uint<8>
    ref.define %_gen_memTap, %rf_memTap : !firrtl.probe<vector<uint<8>, 8>>
  }
  // CHECK: module @Top
  module @Top(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %io_addr: !firrtl.uint<3>, in %io_dataIn: !firrtl.uint<8>, in %io_wen: !firrtl.uint<1>, out %io_dataOut: !firrtl.uint<8>) {
    %dut_clock, %dut_io_addr, %dut_io_dataIn, %dut_io_wen, %dut_io_dataOut, %dut__gen_memTap = instance dut  @DUTModule(in clock: !firrtl.clock, in io_addr: !firrtl.uint<3>, in io_dataIn: !firrtl.uint<8>, in io_wen: !firrtl.uint<1>, out io_dataOut: !firrtl.uint<8>, out _gen_memTap: !firrtl.probe<vector<uint<8>, 8>>)
    %0 = ref.resolve %dut__gen_memTap : !firrtl.probe<vector<uint<8>, 8>>
    // CHECK:      %[[#xmr:]] = sv.xmr.ref @[[path]] ".Memory"
    // CHECK-NEXT: %[[#cast:]] = builtin.unrealized_conversion_cast %[[#xmr]]
    strictconnect %dut_clock, %clock : !firrtl.clock
    %memTap_0 = wire   {annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]} : !firrtl.uint<8>
    %memTap_1 = wire   {annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]} : !firrtl.uint<8>
    %memTap_2 = wire   {annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]} : !firrtl.uint<8>
    %memTap_3 = wire   {annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]} : !firrtl.uint<8>
    %memTap_4 = wire   {annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]} : !firrtl.uint<8>
    %memTap_5 = wire   {annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]} : !firrtl.uint<8>
    %memTap_6 = wire   {annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]} : !firrtl.uint<8>
    %memTap_7 = wire   {annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]} : !firrtl.uint<8>
    strictconnect %io_dataOut, %dut_io_dataOut : !firrtl.uint<8>
    strictconnect %dut_io_wen, %io_wen : !firrtl.uint<1>
    strictconnect %dut_io_dataIn, %io_dataIn : !firrtl.uint<8>
    strictconnect %dut_io_addr, %io_addr : !firrtl.uint<3>
    %1 = subindex %0[0] : !firrtl.vector<uint<8>, 8>
    // CHECK: %[[#cast_0:]] = subindex %[[#cast]][0]
    strictconnect %memTap_0, %1 : !firrtl.uint<8>
    // CHECK:  strictconnect %memTap_0, %[[#cast_0]] : !firrtl.uint<8>
    %2 = subindex %0[1] : !firrtl.vector<uint<8>, 8>
    // CHECK: %[[#cast_1:]] = subindex %[[#cast]][1]
    strictconnect %memTap_1, %2 : !firrtl.uint<8>
    // CHECK:  strictconnect %memTap_1, %[[#cast_1]] : !firrtl.uint<8>
    %3 = subindex %0[2] : !firrtl.vector<uint<8>, 8>
    // CHECK: %[[#cast_2:]] = subindex %[[#cast]][2]
    strictconnect %memTap_2, %3 : !firrtl.uint<8>
    // CHECK:  strictconnect %memTap_2, %[[#cast_2]] : !firrtl.uint<8>
    %4 = subindex %0[3] : !firrtl.vector<uint<8>, 8>
    // CHECK: %[[#cast_3:]] = subindex %[[#cast]][3]
    strictconnect %memTap_3, %4 : !firrtl.uint<8>
    // CHECK:  strictconnect %memTap_3, %[[#cast_3]] : !firrtl.uint<8>
    %5 = subindex %0[4] : !firrtl.vector<uint<8>, 8>
    // CHECK: %[[#cast_4:]] = subindex %[[#cast]][4]
    strictconnect %memTap_4, %5 : !firrtl.uint<8>
    // CHECK:  strictconnect %memTap_4, %[[#cast_4]] : !firrtl.uint<8>
    %6 = subindex %0[5] : !firrtl.vector<uint<8>, 8>
    // CHECK: %[[#cast_5:]] = subindex %[[#cast]][5]
    strictconnect %memTap_5, %6 : !firrtl.uint<8>
    // CHECK:  strictconnect %memTap_5, %[[#cast_5]] : !firrtl.uint<8>
    %7 = subindex %0[6] : !firrtl.vector<uint<8>, 8>
    // CHECK: %[[#cast_6:]] = subindex %[[#cast]][6]
    strictconnect %memTap_6, %7 : !firrtl.uint<8>
    // CHECK:  strictconnect %memTap_6, %[[#cast_6]] : !firrtl.uint<8>
    %8 = subindex %0[7] : !firrtl.vector<uint<8>, 8>
    // CHECK: %[[#cast_7:]] = subindex %[[#cast]][7]
    strictconnect %memTap_7, %8 : !firrtl.uint<8>
    // CHECK:  strictconnect %memTap_7, %[[#cast_7]] : !firrtl.uint<8>
    }
}

// -----

firrtl.circuit "Top"  {
  // CHECK: hw.hierpath private @[[path:[a-zA-Z0-9_]+]] [@Top::@xmr_sym, @DUTModule::@[[xmrSym:[a-zA-Z0-9_]+]]]
  // CHECK-LABEL:  module private @DUTModule
  // CHECK-SAME: in %io_wen: !firrtl.uint<1>, out %io_dataOut: !firrtl.uint<8>)
  module private @DUTModule(in %clock: !firrtl.clock, in %io_addr: !firrtl.uint<3>, in %io_dataIn: !firrtl.uint<8>, in %io_wen: !firrtl.uint<1>, out %io_dataOut: !firrtl.uint<8>, out %_gen_memTap_0: !firrtl.probe<uint<8>>, out %_gen_memTap_1: !firrtl.probe<uint<8>>) attributes {annotations = [{class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}]} {
    %c1_ui1 = constant 1 : !firrtl.uint<1>
    %rf_memTap, %rf_read, %rf_write = mem  Undefined  {depth = 2 : i64, name = "rf", portNames = ["memTap", "read", "write"], prefix = "foo_", readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.probe<vector<uint<8>, 2>>, !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<8>>, !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>
    // CHECK:  %rf_read, %rf_write = mem sym @xmr_sym  Undefined  {depth = 2 : i64, name = "rf", portNames = ["read", "write"], prefix = "foo_", readLatency = 0 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<8>>, !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: uint<8>, mask: uint<1>>
    %9 = ref.sub %rf_memTap[0] : !firrtl.probe<vector<uint<8>, 2>>
    ref.define %_gen_memTap_0, %9 : !firrtl.probe<uint<8>>
    %10 = ref.sub %rf_memTap[1] : !firrtl.probe<vector<uint<8>, 2>>
    ref.define %_gen_memTap_1, %10 : !firrtl.probe<uint<8>>
  }
  module @Top(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %io_addr: !firrtl.uint<3>, in %io_dataIn: !firrtl.uint<8>, in %io_wen: !firrtl.uint<1>, out %io_dataOut: !firrtl.uint<8>) {
    %dut_clock, %dut_io_addr, %dut_io_dataIn, %dut_io_wen, %dut_io_dataOut, %dut__gen_memTap_0, %dut__gen_memTap_1 = instance dut  @DUTModule(in clock: !firrtl.clock, in io_addr: !firrtl.uint<3>, in io_dataIn: !firrtl.uint<8>, in io_wen: !firrtl.uint<1>, out io_dataOut: !firrtl.uint<8>, out _gen_memTap_0: !firrtl.probe<uint<8>>, out _gen_memTap_1: !firrtl.probe<uint<8>>)
    %0 = ref.resolve %dut__gen_memTap_0 : !firrtl.probe<uint<8>>
    // CHECK:      %[[#xmr:]] = sv.xmr.ref @[[path]] ".Memory[0]"
    // CHECK-NEXT: %[[#cast_0:]] = builtin.unrealized_conversion_cast %[[#xmr]]
    %1 = ref.resolve %dut__gen_memTap_1 : !firrtl.probe<uint<8>>
    // CHECK:      %[[#xmr:]] = sv.xmr.ref @[[path]] ".Memory[1]"
    // CHECK-NEXT: %[[#cast_1:]] = builtin.unrealized_conversion_cast %[[#xmr]]
    strictconnect %dut_clock, %clock : !firrtl.clock
    %memTap_0 = wire   {annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]} : !firrtl.uint<8>
    %memTap_1 = wire   {annotations = [{class = "firrtl.transforms.DontTouchAnnotation"}]} : !firrtl.uint<8>
    strictconnect %memTap_0, %0 : !firrtl.uint<8>
    // CHECK:      strictconnect %memTap_0, %[[#cast_0]]
    strictconnect %memTap_1, %1 : !firrtl.uint<8>
    // CHECK:      strictconnect %memTap_1, %[[#cast_1]]
  }
}

// -----

// Test lowering of internal path into a module
// CHECK-LABEL: circuit "Top" {
firrtl.circuit "Top" {
  // CHECK: hw.hierpath private @[[path:[a-zA-Z0-9_]+]] [@Top::@bar, @Bar::@[[xmrSym:[a-zA-Z0-9_]+]]]
  module @XmrSrcMod(out %_a: !firrtl.probe<uint<1>>) {
    // CHECK: module @XmrSrcMod() {
    // CHECK-NEXT: }
    %z = verbatim.expr "internal.path" : () -> !firrtl.uint<1>
    %1 = ref.send %z : !firrtl.uint<1>
    ref.define %_a, %1 : !firrtl.probe<uint<1>>
  }
  module @Bar(out %_a: !firrtl.probe<uint<1>>) {
    %xmr   = instance bar sym @barXMR @XmrSrcMod(out _a: !firrtl.probe<uint<1>>)
    // CHECK:  instance bar sym @barXMR  @XmrSrcMod()
    ref.define %_a, %xmr   : !firrtl.probe<uint<1>>
  }
  module @Top() {
    %bar_a = instance bar sym @bar  @Bar(out _a: !firrtl.probe<uint<1>>)
    // CHECK:  instance bar sym @bar  @Bar()
    %a = wire : !firrtl.uint<1>
    %0 = ref.resolve %bar_a : !firrtl.probe<uint<1>>
    // CHECK:      %[[#xmr:]] = sv.xmr.ref @[[path]] ".internal.path"
    // CHECK-NEXT: %[[#cast:]] = builtin.unrealized_conversion_cast %[[#xmr]]
    strictconnect %a, %0 : !firrtl.uint<1>
    // CHECK-NEXT: strictconnect %a, %[[#cast]]
  }
}

// -----

// Test lowering of internal path into a module
// CHECK-LABEL: circuit "Top" {
firrtl.circuit "Top" {
  // CHECK: hw.hierpath private @[[path:[a-zA-Z0-9_]+]] [@Top::@bar, @Bar::@barXMR, @XmrSrcMod::@[[xmrSym:[a-zA-Z0-9_]+]]]
  module @XmrSrcMod(out %_a: !firrtl.probe<uint<1>>) {
    // CHECK: module @XmrSrcMod() {
    // CHECK{LITERAL}:  verbatim.expr "internal.path" : () -> !firrtl.uint<1> {symbols = [@XmrSrcMod]}
    // CHECK:  = node sym @xmr_sym  %[[internal:.+]]  : !firrtl.uint<1>
    %z = verbatim.expr "internal.path" : () -> !firrtl.uint<1> {symbols = [@XmrSrcMod]}
    %1 = ref.send %z : !firrtl.uint<1>
    ref.define %_a, %1 : !firrtl.probe<uint<1>>
  }
  module @Bar(out %_a: !firrtl.probe<uint<1>>) {
    %xmr   = instance bar sym @barXMR @XmrSrcMod(out _a: !firrtl.probe<uint<1>>)
    // CHECK:  instance bar sym @barXMR  @XmrSrcMod()
    ref.define %_a, %xmr   : !firrtl.probe<uint<1>>
  }
  module @Top() {
    %bar_a = instance bar sym @bar  @Bar(out _a: !firrtl.probe<uint<1>>)
    // CHECK:  instance bar sym @bar  @Bar()
    %a = wire : !firrtl.uint<1>
    %0 = ref.resolve %bar_a : !firrtl.probe<uint<1>>
    // CHECK:      %[[#xmr:]] = sv.xmr.ref @[[path]]
    // CHECK-NEXT: %[[#cast:]] = builtin.unrealized_conversion_cast %[[#xmr]]
    strictconnect %a, %0 : !firrtl.uint<1>
    // CHECK-NEXT: strictconnect %a, %[[#cast]]
  }
}

// -----

// Test correct lowering of 0-width ports
firrtl.circuit "Top"  {
  module @XmrSrcMod(in %pa: !firrtl.uint<0>, out %_a: !firrtl.probe<uint<0>>) {
  // CHECK-LABEL: module @XmrSrcMod(in %pa: !firrtl.uint<0>)
    %0 = ref.send %pa : !firrtl.uint<0>
    ref.define %_a, %0 : !firrtl.probe<uint<0>>
  }
  module @Bar(out %_a: !firrtl.probe<uint<0>>) {
    %bar_pa, %bar__a = instance bar sym @barXMR  @XmrSrcMod(in pa: !firrtl.uint<0>, out _a: !firrtl.probe<uint<0>>)
    ref.define %_a, %bar__a : !firrtl.probe<uint<0>>
  }
  module @Top() {
    %bar__a = instance bar sym @bar  @Bar(out _a: !firrtl.probe<uint<0>>)
    %a = wire   : !firrtl.uint<0>
    %0 = ref.resolve %bar__a : !firrtl.probe<uint<0>>
    strictconnect %a, %0 : !firrtl.uint<0>
    // CHECK: %c0_ui0 = constant 0 : !firrtl.uint<0>
    // CHECK: strictconnect %a, %c0_ui0 : !firrtl.uint<0>
  }
}

// -----
// Test lowering of XMR to instance port (result).
// https://github.com/llvm/circt/issues/4559

// CHECK-LABEL: Issue4559
firrtl.circuit "Issue4559" {
// CHECK: hw.hierpath private @xmrPath [@Issue4559::@[[SYM:.+]]]
  extmodule @Source(out sourceport: !firrtl.uint<1>)
  module @Issue4559() {
    // CHECK: %[[PORT:.+]] = instance source @Source
    // CHECK-NEXT: %[[NODE:.+]] = node sym @[[SYM]] interesting_name %[[PORT]]
    // CHECK-NEXT: = sv.xmr.ref @xmrPath
    %port = instance source @Source(out sourceport: !firrtl.uint<1>)
    %port_ref = ref.send %port : !firrtl.uint<1>
    %port_val = ref.resolve %port_ref : !firrtl.probe<uint<1>>
  }
}

// -----
// Check read-only XMR of a rwprobe.

// CHECK-LABEL: circuit "ReadForceable"
firrtl.circuit "ReadForceable" {
  // CHECK: hw.hierpath private @xmrPath [@ReadForceable::@[[wSym:.+]]]
  // CHECK: module @ReadForceable(out %o: !firrtl.uint<2>)
  module @ReadForceable(out %o: !firrtl.uint<2>) {
    %w, %w_ref = wire forceable : !firrtl.uint<2>, !firrtl.rwprobe<uint<2>>
    %x = ref.resolve %w_ref : !firrtl.rwprobe<uint<2>>
    // CHECK-NOT: ref.resolve
    strictconnect %o, %x : !firrtl.uint<2>
    // CHECK:      %w, %w_ref = wire sym @[[wSym]] forceable : !firrtl.uint<2>, !firrtl.rwprobe<uint<2>>
    // CHECK-NEXT: %[[#xmr:]] = sv.xmr.ref @xmrPath : !hw.inout<i2>
    // CHECK-NEXT: %[[#cast:]] = builtin.unrealized_conversion_cast %[[#xmr]] : !hw.inout<i2> to !firrtl.uint<2>
    // CHECK:      strictconnect %o, %[[#cast]] : !firrtl.uint<2>
  }
}

// -----

// CHECK-LABEL: circuit "ForceRelease"
firrtl.circuit "ForceRelease" {
  // CHECK: hw.hierpath private @[[XMRPATH:.+]] [@ForceRelease::@[[INST_SYM:.+]], @RefMe::@[[TARGET_SYM:.+]]]
  // CHECK: module private @RefMe() {
  module private @RefMe(out %p: !firrtl.rwprobe<uint<4>>) {
    // CHECK-NEXT: %x, %x_ref = wire sym @[[TARGET_SYM]] forceable : !firrtl.uint<4>, !firrtl.rwprobe<uint<4>>
    %x, %x_ref = wire forceable : !firrtl.uint<4>, !firrtl.rwprobe<uint<4>>
    // CHECK-NEXT: }
    ref.define %p, %x_ref : !firrtl.rwprobe<uint<4>>
  }
  // CHECK-LABEL: module @ForceRelease
  module @ForceRelease(in %c: !firrtl.uint<1>, in %clock: !firrtl.clock, in %x: !firrtl.uint<4>) {
      // CHECK-NEXT: instance r sym @[[INST_SYM]] @RefMe()
      %r_p = instance r @RefMe(out p: !firrtl.rwprobe<uint<4>>)
      // CHECK-NEXT: %[[REF1:.+]] = sv.xmr.ref @[[XMRPATH]] : !hw.inout<i4>
      // CHECK-NEXT: %[[CAST1:.+]] = builtin.unrealized_conversion_cast %[[REF1]] : !hw.inout<i4> to !firrtl.rwprobe<uint<4>>
   
      // CHECK-NEXT: ref.force %clock, %c, %[[CAST1]], %x : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<4>

      ref.force %clock, %c, %r_p, %x : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<4>
      // CHECK-NEXT: %[[REF2:.+]] = sv.xmr.ref @[[XMRPATH]] : !hw.inout<i4>
      // CHECK-NEXT: %[[CAST2:.+]] = builtin.unrealized_conversion_cast %[[REF2]] : !hw.inout<i4> to !firrtl.rwprobe<uint<4>>

      // CHECK-NEXT: ref.force_initial %c, %[[CAST2]], %x : !firrtl.uint<1>, !firrtl.uint<4>
      ref.force_initial %c, %r_p, %x : !firrtl.uint<1>, !firrtl.uint<4>
      // CHECK-NEXT: %[[REF3:.+]] = sv.xmr.ref @[[XMRPATH]] : !hw.inout<i4>
      // CHECK-NEXT: %[[CAST3:.+]] = builtin.unrealized_conversion_cast %[[REF3]] : !hw.inout<i4> to !firrtl.rwprobe<uint<4>>
      // CHECK-NEXT: ref.release %clock, %c, %[[CAST3]] : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<4>>
      ref.release %clock, %c, %r_p : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<4>>
      // CHECK-NEXT: %[[REF4:.+]] = sv.xmr.ref @[[XMRPATH]] : !hw.inout<i4>
      // CHECK-NEXT: %[[CAST4:.+]] = builtin.unrealized_conversion_cast %[[REF4]] : !hw.inout<i4> to !firrtl.rwprobe<uint<4>>
      // CHECK-NEXT: ref.release_initial %c, %[[CAST4]] : !firrtl.uint<1>, !firrtl.rwprobe<uint<4>>
      ref.release_initial %c, %r_p : !firrtl.uint<1>, !firrtl.rwprobe<uint<4>>
    }
  }

// -----
// Check tracking of public output refs as sv.macro.decl and sv.macro.def

// CHECK-LABEL: circuit "Top"
firrtl.circuit "Top" {
  // CHECK: sv.macro.decl @ref_Top_Top_a
  // CHECK-NEXT{LITERAL}: sv.macro.def @ref_Top_Top_a "{{0}}"
  // CHECK-SAME:          ([@[[XMR1:.*]]]) {output_file = #hw.output_file<"ref_Top_Top.sv">}

  // CHECK-NEXT:  sv.macro.decl @ref_Top_Top_b
  // CHECK-NEXT{LITERAL}: sv.macro.def @ref_Top_Top_b "{{0}}"
  // CHECK-SAME:          ([@[[XMR2:.*]]]) {output_file = #hw.output_file<"ref_Top_Top.sv">}

  // CHECK-NEXT:  sv.macro.decl @ref_Top_Top_c
  // CHECK-NEXT{LITERAL}: sv.macro.def @ref_Top_Top_c "{{0}}.internal.path"
  // CHECK-SAME:          ([@[[XMR3:.*]]]) {output_file = #hw.output_file<"ref_Top_Top.sv">}

  // CHECK-NEXT:  sv.macro.decl @ref_Top_Top_d
  // CHECK-NEXT{LITERAL}: sv.macro.def @ref_Top_Top_d "{{0}}"
  // CHECK-SAME:          ([@[[XMR4:.+]]]) {output_file = #hw.output_file<"ref_Top_Top.sv">}

  // CHECK-NOT:   sv.macro.decl @ref_Top_Top_e
  // CHECK:  hw.hierpath private @[[XMR5:.+]] [@Foo::@x]
  // CHECK:  sv.macro.decl @ref_Top_Foo_x
  // CHECK-NEXT{LITERAL}: sv.macro.def @ref_Top_Foo_x "{{0}}"
  // CHECK-SAME:          ([@[[XMR5]]]) {output_file = #hw.output_file<"ref_Top_Foo.sv">}

  // CHECK-NEXT:  sv.macro.decl @ref_Top_Foo_y
  // CHECK-NEXT:          sv.macro.def @ref_Top_Foo_y "internal.path" 
  // CHECK-NOT:           ([
  // CHECK-SAME:          {output_file = #hw.output_file<"ref_Top_Foo.sv">}

  // CHECK:        hw.hierpath private @[[XMR1]] [@Top::@w]
  // CHECK:        hw.hierpath private @[[XMR2]] [@Top::@foo, @Foo::@x]
  // CHECK:        hw.hierpath private @[[XMR3]] [@Top::@foo]
  // CHECK:        hw.hierpath private @[[XMR4]] [@Top::@xmr_sym]
  
  // CHECK-LABEL: module @Top()
  module @Top(out %a: !firrtl.probe<uint<1>>, 
                     out %b: !firrtl.probe<uint<1>>, 
                     out %c: !firrtl.probe<uint<1>>, 
                     out %d: !firrtl.probe<uint<1>>,
                     in %e: !firrtl.probe<uint<1>>) {
    %w = wire sym @w : !firrtl.uint<1>
    %0 = ref.send %w : !firrtl.uint<1>
    ref.define %a, %0 : !firrtl.probe<uint<1>>
    
    %x, %y = instance foo sym @foo @Foo(out x: !firrtl.probe<uint<1>>, out y: !firrtl.probe<uint<1>>)
    ref.define %b, %x : !firrtl.probe<uint<1>>
    ref.define %c, %y : !firrtl.probe<uint<1>>
    
    %constant = constant 0 : !firrtl.uint<1>
    %1 = ref.send %constant : !firrtl.uint<1>
    ref.define %d, %1 : !firrtl.probe<uint<1>>
  }

  // CHECK-LABEL: module @Foo()
  module @Foo(out %x: !firrtl.probe<uint<1>>, out %y: !firrtl.probe<uint<1>>) {
    %w = wire sym @x : !firrtl.uint<1>
    %0 = ref.send %w : !firrtl.uint<1>
    ref.define %x, %0 : !firrtl.probe<uint<1>>

    %z = verbatim.expr "internal.path" : () -> !firrtl.uint<1>
    %1 = ref.send %z : !firrtl.uint<1>
    ref.define %y, %1 : !firrtl.probe<uint<1>>
  }
}

// -----
// Check resolving XMR's to internalPaths

// CHECK-LABEL: circuit "InternalPaths"
firrtl.circuit "InternalPaths" {
  extmodule private @RefExtMore(in in: !firrtl.uint<1>,
                                       out r: !firrtl.probe<uint<1>>,
                                       out data: !firrtl.uint<3>,
                                       out r2: !firrtl.probe<vector<bundle<a: uint<3>>, 3>>) attributes {convention = #firrtl<convention scalarized>, internalPaths = ["path.to.internal.signal", "in"]}
  // CHECK: hw.hierpath private @xmrPath [@InternalPaths::@xmr_sym] 
  // CHECK: module public @InternalPaths(
  module public @InternalPaths(in %in: !firrtl.uint<1>) {
    // CHECK: instance ext sym @[[EXT_SYM:.+]] @RefExtMore
    %ext_in, %ext_r, %ext_data, %ext_r2 =
      instance ext @RefExtMore(in in: !firrtl.uint<1>,
                                      out r: !firrtl.probe<uint<1>>,
                                      out data: !firrtl.uint<3>,
                                      out r2: !firrtl.probe<vector<bundle<a: uint<3>>, 3>>)
   strictconnect %ext_in, %in : !firrtl.uint<1>

   // CHECK: %[[XMR_R:.+]] = sv.xmr.ref @xmrPath ".path.to.internal.signal" : !hw.inout<i1>
   // CHECK: %[[XMR_R_CAST:.+]] = builtin.unrealized_conversion_cast %[[XMR_R]] : !hw.inout<i1> to !firrtl.uint<1>
   // CHECK: %node_r = node %[[XMR_R_CAST]]
   %read_r  = ref.resolve %ext_r : !firrtl.probe<uint<1>>
   %node_r = node %read_r : !firrtl.uint<1>
   // CHECK: %[[XMR_R2:.+]] = sv.xmr.ref @xmrPath ".in" : !hw.inout<array<3xstruct<a: i3>>>
   // CHECK: %[[XMR_R2_CAST:.+]] = builtin.unrealized_conversion_cast %[[XMR_R2]] : !hw.inout<array<3xstruct<a: i3>>> to !firrtl.vector<bundle<a: uint<3>>, 3>
   // CHECK: %node_r2 = node %[[XMR_R2_CAST]]
   %read_r2  = ref.resolve %ext_r2 : !firrtl.probe<vector<bundle<a: uint<3>>, 3>>
   %node_r2 = node %read_r2 : !firrtl.vector<bundle<a: uint<3>>, 3>
  }
}

// -----
// Check resolving XMR's to use macro ABI.

// CHECK-LABEL: circuit "RefABI"
firrtl.circuit "RefABI" {
  extmodule private @RefExtMore(in in: !firrtl.uint<1>,
                                       out r: !firrtl.probe<uint<1>>,
                                       out data: !firrtl.uint<3>,
                                       out r2: !firrtl.probe<vector<bundle<a: uint<3>>, 3>>) attributes {convention = #firrtl<convention scalarized>}
  // CHECK:  hw.hierpath private @xmrPath [@RefABI::@xmr_sym] 
  // CHECK: module public @RefABI(
  module public @RefABI(in %in: !firrtl.uint<1>) {
    %ext_in, %ext_r, %ext_data, %ext_r2 =
      instance ext @RefExtMore(in in: !firrtl.uint<1>,
                                      out r: !firrtl.probe<uint<1>>,
                                      out data: !firrtl.uint<3>,
                                      out r2: !firrtl.probe<vector<bundle<a: uint<3>>, 3>>)
   strictconnect %ext_in, %in : !firrtl.uint<1>

   // CHECK: %[[XMR_R:.+]] = sv.xmr.ref @xmrPath ".`ref_RefExtMore_RefExtMore_r" : !hw.inout<i1>
   // CHECK: %[[XMR_R_CAST:.+]] = builtin.unrealized_conversion_cast %[[XMR_R]] : !hw.inout<i1> to !firrtl.uint<1>
   // CHECK: %node_r = node %[[XMR_R_CAST]]
   %read_r  = ref.resolve %ext_r : !firrtl.probe<uint<1>>
   %node_r = node %read_r : !firrtl.uint<1>
   // CHECK: %[[XMR_R2:.+]] = sv.xmr.ref @xmrPath ".`ref_RefExtMore_RefExtMore_r2" : !hw.inout<array<3xstruct<a: i3>>>
   // CHECK: %[[XMR_R2_CAST:.+]] = builtin.unrealized_conversion_cast %[[XMR_R2]] : !hw.inout<array<3xstruct<a: i3>>> to !firrtl.vector<bundle<a: uint<3>>, 3>
   // CHECK: %node_r2 = node %[[XMR_R2_CAST]]
   %read_r2  = ref.resolve %ext_r2 : !firrtl.probe<vector<bundle<a: uint<3>>, 3>>
   %node_r2 = node %read_r2 : !firrtl.vector<bundle<a: uint<3>>, 3>
  }
}

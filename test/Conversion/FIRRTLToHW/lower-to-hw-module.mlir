// RUN: circt-opt -lower-firrtl-to-hw %s -verify-diagnostics | FileCheck %s

// The circuit should be removed.
// CHECK-NOT: circuit

// We should get a large header boilerplate.
// CHECK:   sv.ifdef "PRINTF_COND" {
// CHECK-NEXT:   sv.macro.def @PRINTF_COND_ "(`PRINTF_COND)"
// CHECK-NEXT:  } else  {
firrtl.circuit "Simple" {

   // CHECK-LABEL: hw.module.extern private @MyParameterizedExtModule
   // CHECK-SAME: <DEFAULT: i64, DEPTH: f64, FORMAT: none, WIDTH: i8>
   // CHECK-SAME: (%in: i1) -> (out: i8)
   // CHECK: attributes {verilogName = "name_thing"}
   extmodule private @MyParameterizedExtModule
     <DEFAULT: i64 = 0,
      DEPTH: f64 = 3.242000e+01,
      FORMAT: none = "xyz_timeout=%d\0A",
      WIDTH: i8 = 32>
    (in in: !firrtl.uint<1>, out out: !firrtl.uint<8>)
    attributes {defname = "name_thing"}

   // CHECK-LABEL: hw.module @Simple(%in1: i4, %in2: i2, %in3: i8) -> (out4: i4)
   module @Simple(in %in1: !firrtl.uint<4>,
                         in %in2: !firrtl.uint<2>,
                         in %in3: !firrtl.sint<8>,
                         out %out4: !firrtl.uint<4>) {

    %1 = asUInt %in1 : (!firrtl.uint<4>) -> !firrtl.uint<4>

    // CHECK: comb.concat %false, %in1
    // CHECK: comb.concat %false, %in1

    // CHECK: comb.sub
    %2 = sub %1, %1 : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<5>

    // CHECK: %3 = comb.concat %false, %in2 : i1, i2
    %3 = pad %in2, 3 : (!firrtl.uint<2>) -> !firrtl.uint<3>
    // CHECK: comb.concat %false, %3 : i1, i3
    %4 = pad %3, 4 : (!firrtl.uint<3>) -> !firrtl.uint<4>
    // CHECK: [[RESULT:%.+]] = comb.xor
    %5 = xor %in2, %4 : (!firrtl.uint<2>, !firrtl.uint<4>) -> !firrtl.uint<4>

    connect %out4, %5 : !firrtl.uint<4>, !firrtl.uint<4>
    // CHECK-NEXT: hw.output [[RESULT]] : i4
  }

  // CHECK-LABEL: hw.module private @TestInstance(
  module private @TestInstance(in %u2: !firrtl.uint<2>, in %s8: !firrtl.sint<8>,
                              in %clock: !firrtl.clock,
                              in %reset: !firrtl.uint<1>) {
    // CHECK-NEXT: %c0_i2 = hw.constant
    // CHECK-NEXT: %xyz.out4 = hw.instance "xyz" @Simple(in1: [[ARG1:%.+]]: i4, in2: %u2: i2, in3: %s8: i8) -> (out4: i4)
    %xyz:4 = instance xyz @Simple(in in1: !firrtl.uint<4>, in in2: !firrtl.uint<2>, in in3: !firrtl.sint<8>, out out4: !firrtl.uint<4>)

    // CHECK: [[ARG1]] = comb.concat %c0_i2, %u2 : i2, i2
    connect %xyz#0, %u2 : !firrtl.uint<4>, !firrtl.uint<2>

    // CHECK-NOT: hw.connect
    connect %xyz#1, %u2 : !firrtl.uint<2>, !firrtl.uint<2>

    connect %xyz#2, %s8 : !firrtl.sint<8>, !firrtl.sint<8>

    printf %clock, %reset, "%x"(%xyz#3) : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<4>

    // Parameterized module reference.
    // hw.instance carries the parameters, unlike at the FIRRTL layer.

    // CHECK: %myext.out = hw.instance "myext" @MyParameterizedExtModule<DEFAULT: i64 = 0, DEPTH: f64 = 3.242000e+01, FORMAT: none = "xyz_timeout=%d\0A", WIDTH: i8 = 32>(in: %reset: i1) -> (out: i8)
    %myext:2 = instance myext @MyParameterizedExtModule(in in: !firrtl.uint<1>, out out: !firrtl.uint<8>)

    // CHECK: [[FD:%.*]] = hw.constant -2147483646 : i32
    // CHECK: sv.fwrite [[FD]], "%x"(%xyz.out4) : i4
    // CHECK: sv.fwrite [[FD]], "Something interesting! %x"(%myext.out) : i8

    connect %myext#0, %reset : !firrtl.uint<1>, !firrtl.uint<1>

    printf %clock, %reset, "Something interesting! %x"(%myext#1) : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<8>
  }

  // CHECK-LABEL: hw.module private @OutputFirst(%in1: i1, %in4: i4) -> (out4: i4) {
  module private @OutputFirst(out %out4: !firrtl.uint<4>,
                             in %in1: !firrtl.uint<1>,
                             in %in4: !firrtl.uint<4>) {
    connect %out4, %in4 : !firrtl.uint<4>, !firrtl.uint<4>

    // CHECK-NEXT: hw.output %in4 : i4
  }

  // CHECK-LABEL: hw.module private @PortMadness(
  // CHECK: %inA: i4, %inB: i4, %inC: i4, %inE: i3)
  // CHECK: -> (outA: i4, outB: i4, outC: i4, outD: i4, outE: i4) {
  module private @PortMadness(in %inA: !firrtl.uint<4>,
                             in %inB: !firrtl.uint<4>,
                             in %inC: !firrtl.uint<4>,
                             out %outA: !firrtl.uint<4>,
                             out %outB: !firrtl.uint<4>,
                             out %outC: !firrtl.uint<4>,
                             out %outD: !firrtl.uint<4>,
                             in %inE: !firrtl.uint<3>,
                             out %outE: !firrtl.uint<4>) {
    // Normal
    connect %outA, %inA : !firrtl.uint<4>, !firrtl.uint<4>

    // Multi connect
    connect %outB, %inA : !firrtl.uint<4>, !firrtl.uint<4>
    connect %outB, %inB : !firrtl.uint<4>, !firrtl.uint<4>

    // Unconnected port outC reads as sv.constantZ.
    // CHECK:      [[OUTB:%.+]] = hw.wire %inB
    // CHECK-NEXT: [[OUTC:%.+]] = hw.wire %z_i4
    // CHECK-NEXT: [[OUTD:%.+]] = hw.wire %z_i4
    // CHECK-NEXT: [[T0:%.+]] = comb.concat %false, %inA
    // CHECK-NEXT: [[T1:%.+]] = comb.concat %false, [[OUTC]]
    // CHECK-NEXT: comb.sub bin [[T0]], [[T1]]
    %0 = sub %inA, %outC : (!firrtl.uint<4>, !firrtl.uint<4>) -> !firrtl.uint<5>

    // No connections to outD.

    connect %outE, %inE : !firrtl.uint<4>, !firrtl.uint<3>

    // Extension for outE
    // CHECK: [[OUTE:%.+]] = comb.concat %false, %inE : i1, i3
    // CHECK: hw.output %inA, [[OUTB]], [[OUTC]], [[OUTD]], [[OUTE]]
  }

  // CHECK-LABEL: hw.module private @Analog(%a1: !hw.inout<i1>) -> (outClock: i1) {
  // CHECK-NEXT:    %0 = sv.read_inout %a1 : !hw.inout<i1>
  // CHECK-NEXT:    hw.output %0 : i1
  module private @Analog(in %a1: !firrtl.analog<1>,
                        out %outClock: !firrtl.clock) {

    %clock = asClock %a1 : (!firrtl.analog<1>) -> !firrtl.clock
    connect %outClock, %clock : !firrtl.clock, !firrtl.clock
  }

  // Issue #373: https://github.com/llvm/circt/issues/373
  // CHECK-LABEL: hw.module private @instance_ooo
  module private @instance_ooo(in %arg0: !firrtl.uint<2>, in %arg1: !firrtl.uint<2>,
                              in %arg2: !firrtl.uint<3>,
                              out %out0: !firrtl.uint<8>) {
    // CHECK: %false = hw.constant false

    // CHECK-NEXT: hw.instance "myext" @MyParameterizedExtModule<DEFAULT: i64 = 0, DEPTH: f64 = 3.242000e+01, FORMAT: none = "xyz_timeout=%d\0A", WIDTH: i8 = 32>(in: [[ARG:%.+]]: i1) -> (out: i8)
    %myext:2 = instance myext @MyParameterizedExtModule(in in: !firrtl.uint<1>, out out: !firrtl.uint<8>)

    // CHECK: [[ADD:%.+]] = comb.add bin %0, %1

    // Calculation of input (the add + eq) happens after the
    // instance.
    %0 = add %arg0, %arg0 : (!firrtl.uint<2>, !firrtl.uint<2>) -> !firrtl.uint<3>

    // Multiple uses of the add.
    %a = eq %0, %arg2 : (!firrtl.uint<3>, !firrtl.uint<3>) -> !firrtl.uint<1>
    // CHECK-NEXT: [[ARG]] = comb.icmp bin eq [[ADD]], %arg2 : i3
    connect %myext#0, %a : !firrtl.uint<1>, !firrtl.uint<1>

    connect %out0, %myext#1 : !firrtl.uint<8>, !firrtl.uint<8>

    // CHECK-NEXT: hw.output %myext.out
  }

  // CHECK-LABEL: hw.module private @instance_cyclic
  module private @instance_cyclic(in %arg0: !firrtl.uint<2>, in %arg1: !firrtl.uint<2>) {
    // CHECK: %myext.out = hw.instance "myext" @MyParameterizedExtModule<DEFAULT: i64 = 0, DEPTH: f64 = 3.242000e+01, FORMAT: none = "xyz_timeout=%d\0A", WIDTH: i8 = 32>(in: %0: i1)
    %myext:2 = instance myext @MyParameterizedExtModule(in in: !firrtl.uint<1>, out out: !firrtl.uint<8>)

    // Output of the instance is fed into the input!
    %11 = bits %myext#1 2 to 2 : (!firrtl.uint<8>) -> !firrtl.uint<1>
    // CHECK: %0 = comb.extract %myext.out from 2 : (i8) -> i1

    connect %myext#0, %11 : !firrtl.uint<1>, !firrtl.uint<1>
  }

  // CHECK-LABEL: hw.module private @ZeroWidthPorts(%inA: i4) -> (outa: i4) {
  module private @ZeroWidthPorts(in %inA: !firrtl.uint<4>,
                                in %inB: !firrtl.uint<0>,
                                in %inC: !firrtl.analog<0>,
                                out %outa: !firrtl.uint<4>,
                                out %outb: !firrtl.uint<0>) {
     %0 = mul %inA, %inB : (!firrtl.uint<4>, !firrtl.uint<0>) -> !firrtl.uint<4>
    connect %outa, %0 : !firrtl.uint<4>, !firrtl.uint<4>

    %1 = mul %inB, %inB : (!firrtl.uint<0>, !firrtl.uint<0>) -> !firrtl.uint<0>
    connect %outb, %1 : !firrtl.uint<0>, !firrtl.uint<0>

    attach %inC, %inC : !firrtl.analog<0>, !firrtl.analog<0>

    // CHECK: [[OUTAC:%.+]] = hw.constant 0 : i4
    // CHECK-NEXT: hw.output [[OUTAC]] : i4
  }
  extmodule private @SameNamePorts(in inA: !firrtl.uint<4>,
                                in inA: !firrtl.uint<1>,
                                in inA: !firrtl.analog<1>,
                                out outa: !firrtl.uint<4>,
                                out outa: !firrtl.uint<1>)
  // CHECK-LABEL: hw.module private @ZeroWidthInstance
  module private @ZeroWidthInstance(in %iA: !firrtl.uint<4>,
                                   in %iB: !firrtl.uint<0>,
                                   in %iC: !firrtl.analog<0>,
                                   out %oA: !firrtl.uint<4>,
                                   out %oB: !firrtl.uint<0>) {

    // CHECK: %myinst.outa = hw.instance "myinst" @ZeroWidthPorts(inA: %iA: i4) -> (outa: i4)
    %myinst:5 = instance myinst @ZeroWidthPorts(
      in inA: !firrtl.uint<4>, in inB: !firrtl.uint<0>, in inC: !firrtl.analog<0>, out outa: !firrtl.uint<4>, out outb: !firrtl.uint<0>)
    // CHECK: = hw.instance "myinst" @SameNamePorts(inA: {{.+}}, inA: {{.+}}, inA: {{.+}}) -> (outa: i4, outa: i1)
    %myinst_sameName:5 = instance myinst @SameNamePorts(
      in inA: !firrtl.uint<4>, in inA: !firrtl.uint<1>, in inA: !firrtl.analog<1>, out outa: !firrtl.uint<4>, out outa: !firrtl.uint<1>)

    // Output of the instance is fed into the input!
    connect %myinst#0, %iA : !firrtl.uint<4>, !firrtl.uint<4>
    connect %myinst#1, %iB : !firrtl.uint<0>, !firrtl.uint<0>
    attach %myinst#2, %iC : !firrtl.analog<0>, !firrtl.analog<0>
    connect %oA, %myinst#3 : !firrtl.uint<4>, !firrtl.uint<4>
    connect %oB, %myinst#4 : !firrtl.uint<0>, !firrtl.uint<0>

    // CHECK: hw.output %myinst.outa
  }

  // CHECK-LABEL: hw.module private @SimpleStruct(%source: !hw.struct<valid: i1, ready: i1, data: i64>) -> (sink: !hw.struct<valid: i1, ready: i1, data: i64>) {
  // CHECK-NEXT:    hw.output %source : !hw.struct<valid: i1, ready: i1, data: i64>
  module private @SimpleStruct(in %source: !firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<64>>,
                              out %sink: !firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<64>>) {
    connect %sink, %source : !firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<64>>, !firrtl.bundle<valid: uint<1>, ready: uint<1>, data: uint<64>>
  }

  // https://github.com/llvm/circt/issues/690
  // CHECK-LABEL: hw.module private @bar690(%led_0: !hw.inout<i1>) {
  module private @bar690(in %led_0: !firrtl.analog<1>) {
  }
  // CHECK-LABEL: hw.module private @foo690()
  module private @foo690() {
    // CHECK: %.led_0.wire = sv.wire
    // CHECK: hw.instance "fpga" @bar690(led_0: %.led_0.wire: !hw.inout<i1>) -> ()
    %result = instance fpga @bar690(in led_0: !firrtl.analog<1>)
  }
  // CHECK-LABEL: hw.module private @foo690a(%a: !hw.inout<i1>) {
  module private @foo690a(in %a: !firrtl.analog<1>) {
    %result = instance fpga @bar690(in led_0: !firrtl.analog<1>)
    attach %result, %a: !firrtl.analog<1>, !firrtl.analog<1>
  }

  // https://github.com/llvm/circt/issues/740
  // CHECK-LABEL: hw.module private @foo740(%led_0: !hw.inout<i1>) {
  // CHECK-NEXT:  hw.instance "fpga" @bar740(led_0: %led_0: !hw.inout<i1>) -> ()
  extmodule private @bar740(in led_0: !firrtl.analog<1>)
  module private @foo740(in %led_0: !firrtl.analog<1>) {
    %result = instance fpga @bar740(in led_0: !firrtl.analog<1>)
    attach %result, %led_0 : !firrtl.analog<1>, !firrtl.analog<1>
  }

  extmodule private @UIntToAnalog_8(out a: !firrtl.analog<8>, out b: !firrtl.analog<8>)
  module @Example(out %port: !firrtl.analog<8>) {
    // CHECK-LABEL: hw.module @Example(%port: !hw.inout<i8>)
    // CHECK-NEXT: hw.instance "a2b" @UIntToAnalog_8(a: %port: !hw.inout<i8>, b: %port: !hw.inout<i8>)
    %a2b_a, %a2b_b = instance a2b  @UIntToAnalog_8(out a: !firrtl.analog<8>, out b: !firrtl.analog<8>)
    attach %port, %a2b_b, %a2b_a : !firrtl.analog<8>, !firrtl.analog<8>, !firrtl.analog<8>
  }

  // Memory modules are lowered to plain external modules.
  // CHECK: hw.module.extern @MRead_ext(%R0_addr: i4, %R0_en: i1, %R0_clk: i1) -> (R0_data: i42) attributes {verilogName = "MRead_ext"}
  memmodule @MRead_ext(in R0_addr: !firrtl.uint<4>, in R0_en: !firrtl.uint<1>, in R0_clk: !firrtl.uint<1>, out R0_data: !firrtl.uint<42>) attributes {dataWidth = 42 : ui32, depth = 12 : ui64, extraPorts = [], maskBits = 0 : ui32, numReadPorts = 1 : ui32, numReadWritePorts = 0 : ui32, numWritePorts = 0 : ui32, readLatency = 0 : ui32, writeLatency = 1 : ui32}
  
  // The following operations should be passed through without an error.
  // CHECK: sv.interface @SVInterface
  sv.interface @SVInterface { }

  // DontTouch on ports becomes symbol.
  // CHECK-LABEL: hw.module.extern private @PortDT
  // CHECK-SAME: (%a: i1 {hw.exportPort = #hw<innerSym@__PortDT__a>}, %hassym: i1 {hw.exportPort = #hw<innerSym@hassym>})
  // CHECK-SAME: -> (b: i2 {hw.exportPort = #hw<innerSym@__PortDT__b>})
  extmodule private @PortDT(
    in a: !firrtl.uint<1> [{class = "firrtl.transforms.DontTouchAnnotation"}],
    in hassym: !firrtl.uint<1> sym @hassym [{class = "firrtl.transforms.DontTouchAnnotation"}],
    out b: !firrtl.uint<2> [{class = "firrtl.transforms.DontTouchAnnotation"}]
  )
}

// RUN: circt-opt %s | FileCheck %s

firrtl.circuit "MyModule" {

firrtl.module @mod() { }
firrtl.extmodule @extmod()
firrtl.memmodule @memmod () attributes {
  depth = 16 : ui64, dataWidth = 1 : ui32, extraPorts = [],
  maskBits = 0 : ui32, numReadPorts = 0 : ui32, numWritePorts = 0 : ui32,
  numReadWritePorts = 0 : ui32, readLatency = 0 : ui32,
  writeLatency = 1 : ui32}

// Constant op supports different return types.
firrtl.module @Constants() {
  // CHECK: %c0_ui0 = constant 0 : !firrtl.uint<0>
  constant 0 : !firrtl.uint<0>
  // CHECK: %c0_si0 = constant 0 : !firrtl.sint<0>
  constant 0 : !firrtl.sint<0>
  // CHECK: %c4_ui8 = constant 4 : !firrtl.uint<8>
  constant 4 : !firrtl.uint<8>
  // CHECK: %c-4_si16 = constant -4 : !firrtl.sint<16>
  constant -4 : !firrtl.sint<16>
  // CHECK: %c1_clock = specialconstant 1 : !firrtl.clock
  specialconstant 1 : !firrtl.clock
  // CHECK: %c1_reset = specialconstant 1 : !firrtl.reset
  specialconstant 1 : !firrtl.reset
  // CHECK: %c1_asyncreset = specialconstant 1 : !firrtl.asyncreset
  specialconstant 1 : !firrtl.asyncreset
  // CHECK: constant 4 : !firrtl.uint<8> {name = "test"}
  constant 4 : !firrtl.uint<8> {name = "test"}

  aggregateconstant [1, 2, 3] : !firrtl.bundle<a: uint<8>, b: uint<5>, c: uint<4>>
  aggregateconstant [1, 2, 3] : !firrtl.vector<uint<8>, 3>
  aggregateconstant [[1, 2], [3, 4]] : !firrtl.vector<bundle<a: uint<8>, b: uint<5>>, 2>

}

//module MyModule :
//  input in: UInt<8>
//  output out: UInt<8>
//  out <= in
firrtl.module @MyModule(in %in : !firrtl.uint<8>,
                        out %out : !firrtl.uint<8>) {
  connect %out, %in : !firrtl.uint<8>, !firrtl.uint<8>
}

// CHECK-LABEL: module @MyModule(in %in: !firrtl.uint<8>, out %out: !firrtl.uint<8>)
// CHECK-NEXT:    connect %out, %in : !firrtl.uint<8>, !firrtl.uint<8>
// CHECK-NEXT:  }


//circuit Top :
//  module Top :
//    output out:UInt
//    input b:UInt<32>
//    input c:Analog<13>
//    input d:UInt<16>
//    out <= add(b,d)

firrtl.circuit "Top" {
  module @Top(out %out: !firrtl.uint,
                     in %b: !firrtl.uint<32>,
                     in %c: !firrtl.analog<13>,
                     in %d: !firrtl.uint<16>) {
    %3 = add %b, %d : (!firrtl.uint<32>, !firrtl.uint<16>) -> !firrtl.uint<33>

    %4 = invalidvalue : !firrtl.analog<13>
    attach %c, %4 : !firrtl.analog<13>, !firrtl.analog<13>
    %5 = add %3, %d : (!firrtl.uint<33>, !firrtl.uint<16>) -> !firrtl.uint<34>

    connect %out, %5 : !firrtl.uint, !firrtl.uint<34>
  }
}

// CHECK-LABEL: circuit "Top" {
// CHECK-NEXT:    module @Top(out %out: !firrtl.uint,
// CHECK:                            in %b: !firrtl.uint<32>, in %c: !firrtl.analog<13>, in %d: !firrtl.uint<16>) {
// CHECK-NEXT:      %0 = add %b, %d : (!firrtl.uint<32>, !firrtl.uint<16>) -> !firrtl.uint<33>
// CHECK-NEXT:      %invalid_analog13 = invalidvalue : !firrtl.analog<13>
// CHECK-NEXT:      attach %c, %invalid_analog13 : !firrtl.analog<13>, !firrtl.analog<13>
// CHECK-NEXT:      %1 = add %0, %d : (!firrtl.uint<33>, !firrtl.uint<16>) -> !firrtl.uint<34>
// CHECK-NEXT:      connect %out, %1 : !firrtl.uint, !firrtl.uint<34>
// CHECK-NEXT:    }
// CHECK-NEXT:  }


// Test some hard cases of name handling.
firrtl.module @Mod2(in %in : !firrtl.uint<8>,
                    out %out : !firrtl.uint<8>) attributes {portNames = ["some_name", "out"]}{
  connect %out, %in : !firrtl.uint<8>, !firrtl.uint<8>
}

// CHECK-LABEL: module @Mod2(in %some_name: !firrtl.uint<8>,
// CHECK:                           out %out: !firrtl.uint<8>)
// CHECK-NEXT:    connect %out, %some_name : !firrtl.uint<8>, !firrtl.uint<8>
// CHECK-NEXT:  }

// Check that quotes port names are paresable and printed with quote only if needed.
// CHECK: extmodule @TrickyNames(in "777": !firrtl.uint, in abc: !firrtl.uint)
firrtl.extmodule @TrickyNames(in "777": !firrtl.uint, in "abc": !firrtl.uint)

// Modules may be completely empty.
// CHECK-LABEL: module @no_ports() {
firrtl.module @no_ports() {
}

// stdIntCast can work with clock inputs/outputs too.
// CHECK-LABEL: @ClockCast
firrtl.module @ClockCast(in %clock: !firrtl.clock) {
  // CHECK: %0 = builtin.unrealized_conversion_cast %clock : !firrtl.clock to i1
  %0 = builtin.unrealized_conversion_cast %clock : !firrtl.clock to i1

  // CHECK: %1 = builtin.unrealized_conversion_cast %0 : i1 to !firrtl.clock
  %1 = builtin.unrealized_conversion_cast %0 : i1 to !firrtl.clock
}


// CHECK-LABEL: @TestDshRL
firrtl.module @TestDshRL(in %in1 : !firrtl.uint<2>, in %in2: !firrtl.uint<3>) {
  // CHECK: %0 = dshl %in1, %in2 : (!firrtl.uint<2>, !firrtl.uint<3>) -> !firrtl.uint<9>
  %0 = dshl %in1, %in2 : (!firrtl.uint<2>, !firrtl.uint<3>) -> !firrtl.uint<9>

  // CHECK: %1 = dshr %in1, %in2 : (!firrtl.uint<2>, !firrtl.uint<3>) -> !firrtl.uint<2>
  %1 = dshr %in1, %in2 : (!firrtl.uint<2>, !firrtl.uint<3>) -> !firrtl.uint<2>

  // CHECK: %2 = dshlw %in1, %in2 : (!firrtl.uint<2>, !firrtl.uint<3>) -> !firrtl.uint<2>
  %2 = dshlw %in1, %in2 : (!firrtl.uint<2>, !firrtl.uint<3>) -> !firrtl.uint<2>
}

// We allow implicit truncation of a register's reset value.
// CHECK-LABEL: @RegResetTruncation
firrtl.module @RegResetTruncation(in %clock: !firrtl.clock, in %reset: !firrtl.uint<1>, in %value: !firrtl.bundle<a: uint<2>>, out %out: !firrtl.bundle<a: uint<1>>) {
  %r2 = regreset %clock, %reset, %value  : !firrtl.clock, !firrtl.uint<1>, !firrtl.bundle<a: uint<2>>, !firrtl.bundle<a: uint<1>>
  connect %out, %r2 : !firrtl.bundle<a: uint<1>>, !firrtl.bundle<a: uint<1>>
}

// CHECK-LABEL: @TestNodeName
firrtl.module @TestNodeName(in %in1 : !firrtl.uint<8>) {
  // CHECK: %n1 = node %in1 : !firrtl.uint<8>
  %n1 = node %in1 : !firrtl.uint<8>

  // CHECK: %n1_0 = node %in1 {name = "n1"} : !firrtl.uint<8>
  %n2 = node %in1 {name = "n1"} : !firrtl.uint<8>
}

// Basic test for NLA operations.
// CHECK: hw.hierpath private @nla [@Parent::@child, @Child]
hw.hierpath private @nla [@Parent::@child, @Child]
firrtl.module @Child() {
  %w = wire sym @w : !firrtl.uint<1>
}
firrtl.module @Parent() {
  instance child sym @child @Child()
}

// CHECK-LABEL: @VerbatimExpr
firrtl.module @VerbatimExpr() {
  // CHECK: %[[TMP:.+]] = verbatim.expr "FOO" : () -> !firrtl.uint<42>
  // CHECK: %[[TMP2:.+]] = verbatim.expr "$bits({{[{][{]0[}][}]}})"(%[[TMP]]) : (!firrtl.uint<42>) -> !firrtl.uint<32>
  // CHECK: add %[[TMP]], %[[TMP2]] : (!firrtl.uint<42>, !firrtl.uint<32>) -> !firrtl.uint<43>
  %0 = verbatim.expr "FOO" : () -> !firrtl.uint<42>
  %1 = verbatim.expr "$bits({{0}})"(%0) : (!firrtl.uint<42>) -> !firrtl.uint<32>
  %2 = add %0, %1 : (!firrtl.uint<42>, !firrtl.uint<32>) -> !firrtl.uint<43>
}

// CHECK-LABL: @LowerToBind
// CHECK: instance foo sym @s1 {lowerToBind} @InstanceLowerToBind()
firrtl.module @InstanceLowerToBind() {}
firrtl.module @LowerToBind() {
  instance foo sym @s1 {lowerToBind} @InstanceLowerToBind()
}

// CHECK-LABEL: @ProbeTest
firrtl.module @ProbeTest(in %in1 : !firrtl.uint<2>, in %in2 : !firrtl.uint<3>, out %out3: !firrtl.uint<3>) {
  %w1 = wire  : !firrtl.uint<4>
  // CHECK: %[[TMP3:.+]] = cat
  %w2 = cat %in1, %in1 : (!firrtl.uint<2>, !firrtl.uint<2>) -> !firrtl.uint<4>
  connect %w1, %w2 : !firrtl.uint<4>, !firrtl.uint<4>
  connect %out3, %in2 : !firrtl.uint<3>, !firrtl.uint<3>
  %someNode = node %in1 : !firrtl.uint<2>
  // CHECK: probe @foobar, %in1, %in2, %out3, %w1, %[[TMP3]], %someNode : !firrtl.uint<2>, !firrtl.uint<3>, !firrtl.uint<3>, !firrtl.uint<4>, !firrtl.uint<4>, !firrtl.uint<2>
  probe @foobar, %in1, %in2, %out3, %w1, %w2, %someNode : !firrtl.uint<2>, !firrtl.uint<3>, !firrtl.uint<3>, !firrtl.uint<4>, !firrtl.uint<4>, !firrtl.uint<2>
}

// CHECK-LABEL: module @InnerSymAttr
firrtl.module @InnerSymAttr() {
  %w = wire sym [<@w,2,public>,<@x,1,private>,<@syh,4,public>] : !firrtl.bundle<a: uint<1>, b: uint<1>, c: uint<1>, d: uint<1>>
  // CHECK: %w = wire sym [<@x,1,private>, <@w,2,public>, <@syh,4,public>]
  %w1 = wire sym [<@w1,0,public>] : !firrtl.bundle<a: uint<1>, b: uint<1>, c: uint<1>, d: uint<1>>
  // CHECK: %w1 = wire sym @w1
  %w2 = wire sym [<@w2,0,private>] : !firrtl.bundle<a: uint<1>, b: uint<1>, c: uint<1>, d: uint<1>>
  // CHECK: %w2 = wire sym [<@w2,0,private>]
  %w3, %w3_ref = wire sym [<@w3,2,public>,<@x2,1,private>,<@syh2,0,public>] forceable : !firrtl.bundle<a: uint<1>, b: uint<1>, c: uint<1>, d: uint<1>>, !firrtl.rwprobe<bundle<a: uint<1>, b: uint<1>, c: uint<1>, d: uint<1>>>
  // CHECK: %w3, %w3_ref = wire sym [<@syh2,0,public>, <@x2,1,private>, <@w3,2,public>]
}

// CHECK-LABEL: module @EnumTest
firrtl.module @EnumTest(in %in : !firrtl.enum<a: uint<1>, b: uint<2>>,
                        out %out : !firrtl.uint<2>, out %tag : !firrtl.uint<1>) {
  %v = subtag %in[b] : !firrtl.enum<a: uint<1>, b: uint<2>>
  // CHECK: = subtag %in[b] : !firrtl.enum<a: uint<1>, b: uint<2>>

  %t = tagextract %in : !firrtl.enum<a: uint<1>, b: uint<2>>
  // CHECK: = tagextract %in : !firrtl.enum<a: uint<1>, b: uint<2>>

  strictconnect %out, %v : !firrtl.uint<2>
  strictconnect %tag, %t : !firrtl.uint<1>

  %p = istag %in a : !firrtl.enum<a: uint<1>, b: uint<2>>
  // CHECK: = istag %in a : !firrtl.enum<a: uint<1>, b: uint<2>>

  %c1_ui8 = constant 1 : !firrtl.uint<8>
  %some = enumcreate Some(%c1_ui8) : (!firrtl.uint<8>) -> !firrtl.enum<None: uint<0>, Some: uint<8>>
  // CHECK: = enumcreate Some(%c1_ui8) : (!firrtl.uint<8>) -> !firrtl.enum<None: uint<0>, Some: uint<8>>

  match %in : !firrtl.enum<a: uint<1>, b: uint<2>> {
    case a(%arg0) {
      %w = wire : !firrtl.uint<1>
    }
    case b(%arg0) {
      %x = wire : !firrtl.uint<1>
    }
  }
  // CHECK: match %in : !firrtl.enum<a: uint<1>, b: uint<2>> {
  // CHECK:   case a(%arg0) {
  // CHECK:     %w = wire : !firrtl.uint<1>
  // CHECK:   }
  // CHECK:   case b(%arg0) {
  // CHECK:     %x = wire : !firrtl.uint<1>
  // CHECK:   }
  // CHECK: }

}

// CHECK-LABEL: OpenAggTest
// CHECK-SAME: !firrtl.openbundle<a: bundle<data: uint<1>>, b: openvector<openbundle<x: uint<2>, y: probe<uint<2>>>, 2>>
firrtl.module @OpenAggTest(in %in: !firrtl.openbundle<a: bundle<data: uint<1>>, b: openvector<openbundle<x: uint<2>, y: probe<uint<2>>>, 2>>) {
  %a = opensubfield %in[a] : !firrtl.openbundle<a: bundle<data: uint<1>>, b: openvector<openbundle<x: uint<2>, y: probe<uint<2>>>, 2>>
  %data = subfield %a[data] : !firrtl.bundle<data: uint<1>>
  %b = opensubfield %in[b] : !firrtl.openbundle<a: bundle<data: uint<1>>, b: openvector<openbundle<x: uint<2>, y: probe<uint<2>>>, 2>>
  %b_0 = opensubindex %b[0] : !firrtl.openvector<openbundle<x: uint<2>, y: probe<uint<2>>>, 2>
  %b_1 = opensubindex %b[1] : !firrtl.openvector<openbundle<x: uint<2>, y: probe<uint<2>>>, 2>
  %b_0_y = opensubfield %b_0[y] : !firrtl.openbundle<x : uint<2>, y: probe<uint<2>>>
}

// CHECK-LABEL: StringTest
// CHECK-SAME:  (in %in: !firrtl.string, out %out: !firrtl.string)
firrtl.module @StringTest(in %in: !firrtl.string, out %out: !firrtl.string) {
  propassign %out, %in : !firrtl.string
  // CHECK: %0 = string "hello"
  %0 = string "hello"
}
}

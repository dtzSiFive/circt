; RUN: circt-translate -import-firrtl -verify-diagnostics -split-input-file %s | circt-opt | FileCheck %s

FIRRTL version 2.1.0
circuit MyModule :     ; CHECK: circuit "MyModule" {

  ; CHECK-LABEL: module @MyModule(in %in: !firrtl.uint, out %out: !firrtl.uint<8>) {
  module MyModule :   @[FooBar.scala 369:27]
    input in: UInt
    output out: UInt<8>

    ; CHECK: %0 = widthCast %in : (!firrtl.uint) -> !firrtl.uint<8>
    ; CHECK: strictconnect %out, %0 : !firrtl.uint<8>
    out <= in

  ; CHECK: }


  ; CHECK-LABEL: module private @circuit(in %in: !firrtl.uint<80>) {
  module circuit :    ; Module with a keyword id.
    input in: UInt<80>

  ; CHECK: }

  ; CHECK-LABEL: extmodule private @MyExtModule(in in: !firrtl.uint, out out: !firrtl.uint<8>)
  ; CHECK: attributes {defname = "myextmodule"}
  ; CHECK-NOT: {
  extmodule MyExtModule :
    input in: UInt
    output ,,, out ,,: ,, UInt,<,8,>  ; Commas are whitespace
    defname = myextmodule

  ; CHECK-LABEL: extmodule private @MyParameterizedExtModule
  ; CHECK-SAME:    <FORMAT: none = "xyz_timeout=%d\0A",
  ; CHECK-SAME:     DEFAULT: ui32 = 0,
  ; CHECK-SAME:     WIDTH: ui32 = 32,
  ; CHECK-SAME:     DEPTH: f64 = 3.242000e+01>
  ; CHECK-SAME:    (in in: !firrtl.uint,
  ; CHECK-SAME:     out out: !firrtl.uint<8>)
  ; CHECK-SAME:    attributes {defname = "name_thing"}
  ; CHECK-NOT: {
  extmodule MyParameterizedExtModule :
    input in: UInt
    output out: UInt<8>
    defname = name_thing
    parameter FORMAT = "xyz_timeout=%d\n"
    parameter DEFAULT = 0
    parameter WIDTH = 32
    parameter DEPTH = 32.42


  ; CHECK-LABEL: intmodule private @MyIntModule(in in: !firrtl.uint, out out: !firrtl.uint<8>)
  ; CHECK: attributes {intrinsic = "testIntrinsic1"}
  ; CHECK-NOT: {
  intmodule MyIntModule :
    input in: UInt
    output out: UInt<8>
    intrinsic = testIntrinsic1


  ; CHECK-LABEL: intmodule private @MyParameterizedIntModule
  ; CHECK-SAME:    <FORMAT: none = "xyz_timeout=%d\0A",
  ; CHECK-SAME:     DEFAULT: ui32 = 0,
  ; CHECK-SAME:     WIDTH: ui32 = 32,
  ; CHECK-SAME:     DEPTH: f64 = 3.242000e+01>
  ; CHECK-SAME:    (in in: !firrtl.uint,
  ; CHECK-SAME:     out out: !firrtl.uint<8>)
  ; CHECK-SAME:    attributes {intrinsic = "testIntrinsic2"}
  ; CHECK-NOT: {
  intmodule MyParameterizedIntModule :
    input in: UInt
    output out: UInt<8>
    intrinsic = testIntrinsic2
    parameter FORMAT = "xyz_timeout=%d\n"
    parameter DEFAULT = 0
    parameter WIDTH = 32
    parameter DEPTH = 32.42

  ; Check that integers are extended to 32 bits if they are smaller.
  ; CHECK-LABEL: extmodule private @IntegerParamsModule
  extmodule IntegerParamsModule :
    ; CHECK-SAME: a: ui32 = 1
    parameter a = 1
    ; CHECK-SAME: b: ui40 = 4294967296
    parameter b = 4294967296

  ; Module to test type parsing.

  ; CHECK-LABEL: module private @types(
  module types :
    input c: Clock         ; CHECK: %c: !firrtl.clock,
    input r: Reset         ; CHECK: %r: !firrtl.reset,
    input ar: AsyncReset   ; CHECK: %ar: !firrtl.asyncreset,
    input a: Analog        ; CHECK: %a: !firrtl.analog,
    input a8: Analog<8>    ; CHECK: %a8: !firrtl.analog<8>,
    input s: SInt          ; CHECK: %s: !firrtl.sint,
    input s4: SInt<4>      ; CHECK: %s4: !firrtl.sint<4>,
    input u: UInt          ; CHECK: %u: !firrtl.uint,
    input bf: { flip int_1 : UInt<1>, int_out : UInt<2>}
    ; CHECK: %bf: !firrtl.bundle<int_1 flip: uint<1>, int_out: uint<2>>

    input vec: UInt<1>[4] ; CHECK: %vec: !firrtl.vector<uint<1>, 4>) {


  ; CHECK-LABEL: module private @stmts(
  module stmts :
    input reset : UInt<1>         ; CHECK: in %reset: !firrtl.uint<1>,
    input reset_async: AsyncReset ; CHECK: in %reset_async: !firrtl.asyncreset,
    input reset_abstract: Reset   ; CHECK: in %reset_abstract: !firrtl.reset,
    input clock : Clock           ; CHECK: in %clock: !firrtl.clock,
    output auto : UInt<1>         ; CHECK: out %auto: !firrtl.uint<1>,
    output auto11 : UInt<11>      ; CHECK: out %auto11: !firrtl.uint<11>,
    output sauto : SInt<9>        ; CHECK: out %sauto: !firrtl.sint<9>,
    input i8 : UInt<8>            ; CHECK: in %i8: !firrtl.uint<8>,
    input s1 : SInt<1>            ; CHECK: in %s1: !firrtl.sint<1>,
    input s8 : SInt<8>            ; CHECK: in %s8: !firrtl.sint<8>,
    input a1 : Analog<1>          ; CHECK: in %a1: !firrtl.analog<1>,
    input a8 : Analog<8>          ; CHECK: in %a8: !firrtl.analog<8>,
    input ab : {x : Analog<1>}    ; CHECK: in %ab: !firrtl.bundle<x: analog<1>>)

    ; CHECK: %_t = wire interesting_name : !firrtl.vector<uint<1>, 12>
    wire _t : UInt<1>[12] @[Nodes.scala 370:76]

    ; CHECK: %_t_2 = wire interesting_name : !firrtl.vector<uint<1>, 12>
    wire _t_2 : UInt<1>[12]

    ; CHECK: strictconnect %_t, %_t_2 : !firrtl.vector<uint<1>, 12>
    _t <= _t_2

    ; CHECK: [[INV:%.+]]  = invalidvalue : !firrtl.uint<1>
    ; CHECK-NEXT: strictconnect %auto, [[INV]] : !firrtl.uint<1>
    auto is invalid

    ; CHECK-NOT: attach %a1
    a1 is invalid

    ; CHECK-NOT: attach %ab
    ab is invalid

    ; CHECK: skip
    skip  @[SKipLoc.scala 42:24]

    ; CHECK: [[INV:%.+]] = invalidvalue : !firrtl.uint<1>
    ; CHECK-NEXT: strictconnect %auto, [[INV]] : !firrtl.uint<1>
    auto is invalid

    ; CHECK-NOT: connect %reset
    ; CHECK-NOT: strictconnect %reset
    reset is invalid

    ; CHECK: %out_0 = wire interesting_name : !firrtl.bundle<member: bundle<0: bundle<clock: clock, reset: uint<1>>>>
    wire out_0 : { member : { 0 : { clock : Clock, reset : UInt<1>}}}

    ; CHECK: %_t_3 = wire interesting_name : !firrtl.vector<uint<1>, 12>
    ; CHECK: [[A:%.+]] = subindex %_t_3[0] : !firrtl.vector<uint<1>, 12>
    ; CHECK: %_t_4 = wire interesting_name : !firrtl.vector<uint<1>, 12>
    ; CHECK: [[B:%.+]] = subindex %_t_4[0] : !firrtl.vector<uint<1>, 12>
    ; CHECK: strictconnect [[A]], [[B]]
    wire _t_3 : UInt<1>[12] @[Nodes.scala 370:76]
    wire _t_4 : UInt<1>[12]
    _t_3[0] <= _t_4[0] @[Xbar.scala 21:44]

    ; CHECK: %n1 = node interesting_name %i8 : !firrtl.uint<8>
    node n1 = i8

    ; CHECK: add %reset, %reset : (!firrtl.uint<1>, !firrtl.uint<1>) -> !firrtl.uint<2>
    node n2 = add(reset, reset)

    ; CHECK: asClock %reset : (!firrtl.uint<1>) -> !firrtl.clock
    node n3 = asClock(reset)

    ; CHECK: asUInt %clock : (!firrtl.clock) -> !firrtl.uint<1>
    node check_u0 = asUInt(clock)
    ; CHECK: asUInt %i8 : (!firrtl.uint<8>) -> !firrtl.uint<8>
    node check_u1 = asUInt(i8)
    ; CHECK: asUInt %s8 : (!firrtl.sint<8>) -> !firrtl.uint<8>
    node check_u2 = asUInt(s8)
    ; CHECK: asUInt %a8 : (!firrtl.analog<8>) -> !firrtl.uint<8>
    node check_u3 = asUInt(a8)
    ; CHECK: asUInt %reset_abstract : (!firrtl.reset) -> !firrtl.uint<1>
    node check_u5 = asUInt(reset_abstract)
    ; CHECK: asUInt %reset_async : (!firrtl.asyncreset) -> !firrtl.uint<1>
    node check_u6 = asUInt(reset_async)

    ; CHECK: asSInt %clock : (!firrtl.clock) -> !firrtl.sint<1>
    node check_s0 = asSInt(clock)
    ; CHECK: asSInt %i8 : (!firrtl.uint<8>) -> !firrtl.sint<8>
    node check_s1 = asSInt(i8)
    ; CHECK: asSInt %s8 : (!firrtl.sint<8>) -> !firrtl.sint<8>
    node check_s2 = asSInt(s8)
    ; CHECK: asSInt %a8 : (!firrtl.analog<8>) -> !firrtl.sint<8>
    node check_s3 = asSInt(a8)
    ; CHECK: asSInt %reset_abstract : (!firrtl.reset) -> !firrtl.sint<1>
    node check_s5 = asSInt(reset_abstract)
    ; CHECK: asSInt %reset_async : (!firrtl.asyncreset) -> !firrtl.sint<1>
    node check_s6 = asSInt(reset_async)

    ; CHECK: asAsyncReset %clock : (!firrtl.clock) -> !firrtl.asyncreset
    node check_ar0 = asAsyncReset(clock)
    ; CHECK: asAsyncReset %reset : (!firrtl.uint<1>) -> !firrtl.asyncreset
    node check_ar1 = asAsyncReset(reset)
    ; CHECK: asAsyncReset %s1 : (!firrtl.sint<1>) -> !firrtl.asyncreset
    node check_ar2 = asAsyncReset(s1)
    ; CHECK: asAsyncReset %a1 : (!firrtl.analog<1>) -> !firrtl.asyncreset
    node check_ar3 = asAsyncReset(a1)
    ; CHECK: asAsyncReset %reset_abstract : (!firrtl.reset) -> !firrtl.asyncreset
    node check_ar4 = asAsyncReset(reset_abstract)
    ; CHECK: asAsyncReset %reset_async : (!firrtl.asyncreset) -> !firrtl.asyncreset
    node check_ar5 = asAsyncReset(reset_async)

    ; CHECK: asClock %clock : (!firrtl.clock) -> !firrtl.clock
    node check_c0 = asClock(clock)
    ; CHECK: asClock %reset : (!firrtl.uint<1>) -> !firrtl.clock
    node check_c1 = asClock(reset)
    ; CHECK: asClock %s1 : (!firrtl.sint<1>) -> !firrtl.clock
    node check_c2 = asClock(s1)
    ; CHECK: asClock %a1 : (!firrtl.analog<1>) -> !firrtl.clock
    node check_c3 = asClock(a1)
    ; CHECK: asClock %reset_abstract : (!firrtl.reset) -> !firrtl.clock
    node check_c4 = asClock(reset_abstract)
    ; CHECK: asClock %reset_async : (!firrtl.asyncreset) -> !firrtl.clock
    node check_c5 = asClock(reset_async)

    ; CHECK: node interesting_name %auto : !firrtl.uint<1>
    node check_output = auto

    ; CHECK: %c42_ui10 = constant 42 : !firrtl.const.uint<10>
    ; CHECK: %c171_ui8 = constant 171 : !firrtl.const.uint<8>
    ; CHECK: add %c42_ui10, %c171_ui8
    ; CHECK: constCast
    ; CHECK: strictconnect %auto
    auto11 <= add(UInt<10>(42), UInt<8>("hAB"))

    ; CHECK: %c-85_si8 = constant -85 : !firrtl.const.sint<8>
    sauto <= add(s8, SInt<8>(-85))

    ; CHECK: when %reset : !firrtl.uint<1> {
    ; CHECK:   strictconnect %_t, %_t_2
    ; CHECK: } else {
    ; CHECK:   strictconnect %_t, %_t_2
    ; CHECK: }
    when reset : _t <= _t_2 else : _t <= _t_2

    ; CHECK: when %reset : !firrtl.uint<1> {
    ; CHECK:   [[N4A:%.+]] = node interesting_name %_t_2
    ; CHECK:   strictconnect %_t, [[N4A]]
    ; CHECK: } else {
    ; CHECK:   [[N4B:%.+]] = node interesting_name %_t_2
    ; CHECK:   strictconnect %_t, [[N4B]]
    ; CHECK: }
    when reset :
      node n4 = _t_2
      _t <= n4
    else :
      node n4 = _t_2   ; 'n4' name is in unique scopes.
      _t <= n4

    ; CHECK: [[TMP:%.+]] = constant 4
    ; CHECK: [[COND:%.+]] = lt %reset, [[TMP]]
    ; CHECK: when [[COND]] : !firrtl.uint<1> {
    ; CHECK:   strictconnect %_t, %_t_2
    ; CHECK: }
    ; CHECK-NOT: else
    when lt(reset, UInt(4)) :   ;; When with no else.
      _t <= _t_2

    ; CHECK: when %reset : !firrtl.uint<1> {
    ; CHECK:   strictconnect %_t, %_t_2
    ; CHECK: } else  {
    ; CHECK:   [[COND:%.+]] = not %reset
    ; CHECK:   when [[COND]] : !firrtl.uint<1> {
    ; CHECK:     strictconnect %_t, %_t_2
    ; CHECK:   }
    ; CHECK: }
    when reset :
      _t <= _t_2
    else when not(reset) :
      _t <= _t_2

    ; CHECK: when %reset : !firrtl.uint<1> {
    ; CHECK:   strictconnect %_t, %_t
    ; CHECK: } else  {
    ; CHECK:   [[COND:%.+]] = not %reset
    ; CHECK:   when [[COND]] : !firrtl.uint<1> {
    ; CHECK:     strictconnect %_t, %_t_2
    ; CHECK:   } else  {
    ; CHECK:     strictconnect %_t, %_t_2
    ; CHECK:   }
    ; CHECK: }
    when reset:
      _t <= _t_2
    else when not(reset) :
      _t <= _t_2
    else :
      _t <= _t_2

    ; CHECK: printf %clock, %reset, "Something interesting!\0A %x %x" (%_t, %_t_2) : !firrtl.clock, !firrtl.uint<1>, !firrtl.vector<uint<1>, 12>, !firrtl.vector<uint<1>, 12>
    printf(clock, reset, "Something interesting!\n %x %x", _t, _t_2)

    ; CHECK: printf %clock, %reset, "Something interesting!\0A %x %x" {name = "printf_0"} (%_t, %_t_2) : !firrtl.clock, !firrtl.uint<1>, !firrtl.vector<uint<1>, 12>, !firrtl.vector<uint<1>, 12>
    printf(clock, reset, "Something interesting!\n %x %x", _t, _t_2) : printf_0

    ; CHECK: stop %clock, %reset, 42 : !firrtl.clock, !firrtl.uint<1>
    stop(clock, reset, 42)

    ; CHECK: stop %clock, %reset, 42 {name = "stop_0"} : !firrtl.clock, !firrtl.uint<1>
    stop(clock, reset, 42) : stop_0

    ; CHECK: bits %i8 4 to 2 : (!firrtl.uint<8>) -> !firrtl.uint<3>
    node n4 = bits(i8, 4, 2)

    ; CHECK: shl %i8, 4 : (!firrtl.uint<8>) -> !firrtl.uint<12>
    ; CHECK: shr %i8, 8 : (!firrtl.uint<8>) -> !firrtl.uint<1>
    node n5 = or(shl(i8, 4), shr(i8, 8))

    ; CHECK: dshl %i8, %{{.*}} : (!firrtl.uint<8>, !firrtl.const.uint<4>) -> !firrtl.uint<23>
    node n6 = dshl(i8, UInt<4>(7))
    ; CHECK: dshlw %i8, %{{.*}} : (!firrtl.uint<8>, !firrtl.const.uint<4>) -> !firrtl.uint<8>
    node n6s = dshlw(i8, UInt<4>(7))

    ; CHECK: cat %{{.*}}, %{{.*}} : (!firrtl.uint<12>, !firrtl.uint<23>) -> !firrtl.uint<35>
    node n7 = cat(n5, n6)

    ; CHECK: mux(%reset, %i8, %{{.*}}) : (!firrtl.uint<1>, !firrtl.uint<8>, !firrtl.const.uint) -> !firrtl.uint
    node n8 = mux(reset, i8, UInt(4))

    ; CHECK: %_t_2621 = regreset interesting_name %clock, %reset, %{{.*}} : !firrtl.clock, !firrtl.uint<1>, !firrtl.const.uint<4>, !firrtl.uint<4>
    reg _t_2621 : UInt<4>, clock with :
      reset => (reset, UInt<4>("h0")) @[Edges.scala 230:27]

    ; CHECK: %_t_1601 = regreset interesting_name %clock, %reset, %{{.*}} : !firrtl.clock, !firrtl.uint<1>, !firrtl.const.uint<2>, !firrtl.uint<2>
    reg _t_1601 : UInt<2>, clock with :
      (reset => (reset, UInt<2>("h00"))) @[Edges.scala 230:27]

    ; CHECK: div %i8, %{{.*}} : (!firrtl.uint<8>, !firrtl.const.uint<4>) -> !firrtl.uint<8>
    node n9 = div(i8, UInt<4>(4))

    ; CHECK: tail %i8, 7 : (!firrtl.uint<8>) -> !firrtl.uint<1>
    ; CHECK: tail %i8, 0 : (!firrtl.uint<8>) -> !firrtl.uint<8>
    ; CHECK: head %i8, 4 : (!firrtl.uint<8>) -> !firrtl.uint<4>
    node n10 = add(add(tail(i8, 7), tail(i8, 0)), head(i8, 4))

    ; CHECK: tail %{{.*}}, 3 : (!firrtl.sint<8>) -> !firrtl.uint<5>
    node n10s = tail(asSInt(i8), 3)

    ; The Scala implementation of FIRRTL prints registers without a reset value
    ; using the register name as the reset.  Make sure we handle this for
    ; compatibility.
    ; CHECK: %_t_2622 = reg interesting_name %clock : !firrtl.clock, !firrtl.uint<4>
    reg _t_2622 : UInt<4>, clock with :
      reset => (UInt<1>("h0"), _t_2622)

    ; CHECK: %xyz_in = instance xyz interesting_name @circuit(in in: !firrtl.uint<80>)
    inst xyz of circuit
    ; CHECK: [[PAD:%.*]] = pad %i8, 80 : (!firrtl.uint<8>) -> !firrtl.uint<80>
    ; CHECK: strictconnect %xyz_in, [[PAD]] : !firrtl.uint<80>
    xyz.in <= i8

    ; CHECK: %myext_in, %myext_out = instance myext interesting_name @MyExtModule(in in: !firrtl.uint, out out: !firrtl.uint<8>)
    inst myext of MyExtModule
    myext.in <= i8
    printf(clock, reset, "Something interesting! %x", myext.out)

    ; CHECK: when %reset : !firrtl.uint<1> {
    when reset :
      ; CHECK: %reset_myext_in, %reset_myext_out = instance reset_myext interesting_name @MyExtModule(in in: !firrtl.uint, out out: !firrtl.uint<8>)
      inst reset_myext of MyExtModule
      reset_myext.in <= i8
    ; CHECK: }

    ; CHECK: subaccess %_t[%i8] : !firrtl.vector<uint<1>, 12>, !firrtl.uint<8>
    auto <= _t[i8]

    ; CHECK: subaccess %_t[%auto] : !firrtl.vector<uint<1>, 12>, !firrtl.uint<1>
    auto <= _t[auto]

    ; CHECK: %myMem = chirrtl.combmem interesting_name : !chirrtl.cmemory<bundle<id: uint<4>, resp: uint<2>>, 8>
    cmem myMem : { id : UInt<4>, resp : UInt<2>} [8] @[Decoupled.scala 209:24]

    ; CHECK: %memValue_data, %memValue_port = chirrtl.memoryport Infer %myMem {name = "memValue"} : (!chirrtl.cmemory<bundle<id: uint<4>, resp: uint<2>>, 8>) -> (!firrtl.bundle<id: uint<4>, resp: uint<2>>, !chirrtl.cmemoryport)
    ; CHECK: chirrtl.memoryport.access %memValue_port[%i8], %clock : !chirrtl.cmemoryport, !firrtl.uint<8>, !firrtl.clock
    infer mport memValue = myMem[i8], clock
    auto11 <= memValue.id

    ; CHECK: %base_table_0 = chirrtl.seqmem interesting_name Undefined : !chirrtl.cmemory<vector<uint<1>, 9>, 256>
    smem base_table_0 : UInt<1>[9] [256]
    ; CHECK: %base_table_1 = chirrtl.seqmem interesting_name Old : !chirrtl.cmemory<vector<uint<1>, 9>, 256>
    smem base_table_1 : UInt<1>[9] [256] old

    ; CHECK: %tableValue_data, %tableValue_port = chirrtl.memoryport Read %base_table_1 {name = "tableValue"} : (!chirrtl.cmemory<vector<uint<1>, 9>, 256>) -> (!firrtl.vector<uint<1>, 9>, !chirrtl.cmemoryport)
    ; CHECK: chirrtl.memoryport.access %tableValue_port[%i8], %clock : !chirrtl.cmemoryport, !firrtl.uint<8>, !firrtl.clock
    read mport tableValue = base_table_1[i8], clock

    ; Check that we can handle large memory sizes.
    ; CHECK: %testharness = chirrtl.seqmem interesting_name Undefined : !chirrtl.cmemory<vector<uint<8>, 16>, 2147483648>
    smem testharness : UInt<8>[16][2147483648], undefined

    ; CHECK: pad %i8, 10 : (!firrtl.uint<8>) -> !firrtl.uint<10>
    node n11 = pad(i8, 10)

    ; CHECK: andr %n11 : (!firrtl.uint<10>) -> !firrtl.uint<1>
    node n12 = andr(n11)

    ; CHECK: = not %auto : (!firrtl.uint<1>) -> !firrtl.uint<1>
    node n13 = not(auto)


    ; CHECK: %_M__T_10, %_M__T_11, %_M__T_18 = mem interesting_name Undefined {depth = 8 : i64, name = "_M", portNames = ["_T_10", "_T_11", "_T_18"]
    ; CHECK-SAME: readLatency = 0 : i32, writeLatency = 1 : i32} :
    ; CHECK-SAME: !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data: bundle<id: uint<4>>, mask: bundle<id: uint<1>>>,
    ; CHECK-SAME: !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data: bundle<id: uint<4>>, mask: bundle<id: uint<1>>>,
    ; CHECK-SAME: !firrtl.bundle<addr: uint<3>, en: uint<1>, clk: clock, data flip: bundle<id: uint<4>>
    mem _M : @[Decoupled.scala 209:24]
        data-type => { id : UInt<4> }
        depth => 8
        read-latency => 0
        write-latency => 1
        reader => _T_18
        writer => _T_10 _T_11
        read-under-write => undefined
    _M._T_18.addr is invalid @[Decoupled.scala 209:24]
    _M._T_18.clk is invalid @[Decoupled.scala 209:24]
    _M._T_18.en <= UInt<1>("h0") @[Decoupled.scala 209:24]
    _M._T_10.addr is invalid @[Decoupled.scala 209:24]
    _M._T_10.clk is invalid @[Decoupled.scala 209:24]
    _M._T_10.en <= UInt<1>("h0") @[Decoupled.scala 209:24]
    _M._T_10.data is invalid @[Decoupled.scala 209:24]
    _M._T_10.mask is invalid @[Decoupled.scala 209:24]

    ; CHECK: attach %a8, %a8, %a8 :
    attach (a8, a8, a8)

    wire pred: UInt <1>
    wire en: UInt <1>
    pred <= eq(i8, i8)
    en <= not(reset)
    ; CHECK: assert %clock, %pred, %en, "X equals Y when Z is valid" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1> {eventControl = 0 : i32, isConcurrent = false}
    assert(clock, pred, en, "X equals Y when Z is valid")
    ; CHECK: assert %clock, %pred, %en, "X equals Y when Z is valid" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1> {eventControl = 0 : i32, isConcurrent = false, name = "assert_0"}
    assert(clock, pred, en, "X equals Y when Z is valid") : assert_0
    ; CHECK: assume %clock, %pred, %en, "X equals Y when Z is valid" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1> {eventControl = 0 : i32, isConcurrent = false}
    assume(clock, pred, en, "X equals Y when Z is valid")
    ; CHECK: assume %clock, %pred, %en, "X equals Y when Z is valid" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1> {eventControl = 0 : i32, isConcurrent = false, name = "assume_0"}
    assume(clock, pred, en, "X equals Y when Z is valid") : assume_0
    ; CHECK: cover %clock, %pred, %en, "X equals Y when Z is valid" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1> {eventControl = 0 : i32, isConcurrent = false}
    cover(clock, pred, en, "X equals Y when Z is valid")
    ; CHECK: cover %clock, %pred, %en, "X equals Y when Z is valid" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1> {eventControl = 0 : i32, isConcurrent = false, name = "cover_0"}
    cover(clock, pred, en, "X equals Y when Z is valid") : cover_0

  ; CHECK-LABEL: module private @type_handling(
  module type_handling :
    wire _t_6 : { flip b : { bits : { source : UInt<7> } } }
    node _t_8 = bits(_t_6.b.bits.source, 5, 0)

    ; CHECK: %flip1 = wire interesting_name : !firrtl.bundle<x flip: bundle<a flip: uint>>
    wire flip1 : { flip x : { flip a : UInt } }
    ; CHECK: %flip2 = wire interesting_name : !firrtl.bundle<x flip: bundle<a flip: uint, b: analog>>
    wire flip2 : { flip x : { flip a : UInt, b: Analog } }
    ; CHECK: %flip3 = wire interesting_name : !firrtl.bundle<x flip: bundle<a flip: uint, b flip: analog>>
    wire flip3 : { flip x : { flip a : UInt, flip b: Analog } }
    ; CHECK: %flip4 = wire interesting_name : !firrtl.bundle<x flip: vector<bundle<a flip: uint>, 4>>
    wire flip4 : { flip x : { flip a : UInt }[4] }


  ; CHECK-LABEL: module private @expr_stmt_ambiguity(
  module expr_stmt_ambiguity :
    ; CHECK: %reg = wire interesting_name : !firrtl.uint
    wire reg : UInt
    ; CHECK: strictconnect %reg,
    reg <= UInt(42)

    ; CHECK: %write = wire
    wire write : { id : UInt<4>, resp : UInt<2>}

    ; CHECK: subfield %write[id]
    write.id <= UInt(1)

  ; CHECK-LABEL: module private @expr_stmt_ambiguity2(
  module expr_stmt_ambiguity2 :
    ; CHECK: instance write interesting_name @circuit
    inst write of circuit
    ; CHECK: strictconnect %write_in
    write.in <= UInt(1)

  ; CHECK-LABEL: module private @oversize_shift(
  module oversize_shift :
    wire value : UInt<2>
    ; CHECK: shr %value, 5 : (!firrtl.uint<2>) -> !firrtl.uint<1>
    node n = shr(value, 5)

  ; CHECK-LABEL: module private @when_else_ambiguity(
  module when_else_ambiguity :
    output out : UInt
    input in : UInt
    wire reset : UInt<1>

  ; CHECK: when {{.*}} : !firrtl.uint<1> {
    when reset : @[Debug.scala 1176:37]
    ; CHECK: when {{.*}} : !firrtl.uint<1> {
      when reset :
        out <= in
    ; CHECK: }
    ; CHECK: } else {
    else :
        ; CHECK: when {{.*}} : !firrtl.uint<1> {
      when reset : @[Debug.scala 1180:39]
        out <= in
    ; CHECK: }
    ; CHECK: }


  ; CHECK-LABEL: module private @chisel_when_mport_bug(
  module chisel_when_mport_bug :
    input cond : UInt<1>
    input addr : UInt
    input clock : Clock

    ; Memory ports should be declared in the scope of the cmemory, but should
    ; be enabled at the location of the mport.

    ; CHECK: %memory = chirrtl.seqmem interesting_name Undefined  : !chirrtl.cmemory<vector<uint<1>, 9>, 256>
    smem memory : UInt<1>[9] [256]

    ; CHECK: %xyz0_data, %xyz0_port = chirrtl.memoryport Read %memory {name = "xyz0"} : (!chirrtl.cmemory<vector<uint<1>, 9>, 256>) -> (!firrtl.vector<uint<1>, 9>, !chirrtl.cmemoryport)
    ; CHECK: when %cond : !firrtl.uint<1> {
    ; CHECK:    chirrtl.memoryport.access %xyz0_port[%addr], %clock : !chirrtl.cmemoryport, !firrtl.uint, !firrtl.clock
    ; CHECK: }
    when cond :
      read mport xyz0 = memory[addr], clock

    ; CHECK: when %cond : !firrtl.uint<1> {
    ; CHECK:    %n0 = node interesting_name %xyz0_data  : !firrtl.vector<uint<1>, 9>
    ; CHECK: }
    when cond :
      node n0 = xyz0

    ; CHECK: %n1 = node interesting_name %xyz0_data  : !firrtl.vector<uint<1>, 9>
    node n1 = xyz0


  ; CHECK-LABEL: module private @constant_implicit_cse(
  module constant_implicit_cse :
    input cond : UInt<1>

    ; CHECK: [[CST15:%.+]] = constant 15 : !firrtl.const.uint<4>
    ; CHECK: %a = node interesting_name [[CST15]]
    node a = UInt<4>(15)
    ; CHECK: %b = node interesting_name [[CST15]]
    node b = UInt<4>(15)

    ;; Constants always get emitted to the top level.
    ; CHECK: [[CST7:%.+]] = constant 7 : !firrtl.const.uint<4>
    ; CHECK: when %cond : !firrtl.uint<1> {
    when cond :
      ; CHECK: %c = node interesting_name [[CST15]]
      node c = UInt<4>(15)
      ; CHECK: %d = node interesting_name [[CST7]]
      node d = UInt<4>(7)
      ; CHECK: when %cond : !firrtl.uint<1> {
      when cond :
        ; CHECK:  %e = node interesting_name [[CST7]]
        node e = UInt<4>(7)
    ; CHECK: }
    ; CHECK: }

    ; CHECK:  %f = node interesting_name [[CST15]]
    node f = UInt<4>(15)
    node g = UInt<4>(7)

  ; CHECK-LABEL: module private @subfield_implicit_cse
  module subfield_implicit_cse :
    input i: {x: UInt<1>}
    input cond: UInt<1>
    output o: UInt<1>

    ; Subfields always get emitted by their declarations.
    ; CHECK: [[SUB:%.+]] = subfield %i[x]

    ; CHECK: %n3 = node interesting_name [[SUB]]
    node n3 = i.x

    ; CHECK: when %cond : !firrtl.uint<1> {
    when cond:
      ; CHECK: %n4 = node interesting_name [[SUB]]
      node n4 = i.x
    ; CHECK: }

    ; Check that invalidation reuses subfields
    wire w: {a: UInt<1>}[1]
    ; CHECK: %invalid = invalidvalue : !firrtl.vector<bundle<a: uint<1>>, 1>
    ; CHECK: strictconnect %w, %invalid
    w is invalid
    ; CHECK: %invalid_0 = invalidvalue : !firrtl.vector<bundle<a: uint<1>>, 1>
    ; CHECK: strictconnect %w, %invalid_0
    w is invalid

  ; CHECK-LABEL: module private @flip_one
  module flip_one :
    input bf: { flip int_1 : UInt<1>, int_out : UInt<2>}
    ; CHECK: %0 = subfield %bf[int_1]
    ; CHECK: %_T = node interesting_name %0
    node _T = bf.int_1
    ; CHECK: when %_T : !firrtl.uint<1> {
    when _T :
      skip

  ; CHECK-LABEL: module private @mem_depth_1
  module mem_depth_1 :
    input clock : Clock
    input reset : UInt<1>

    mem bar : @[Decoupled.scala 218:16]
      data-type => UInt<3>
      depth => 1
      read-latency => 0
      write-latency => 1
      reader => io_deq_bits_MPORT
      writer => MPORT
      read-under-write => undefined
      ; CHECK: %bar_MPORT, %bar_io_deq_bits_MPORT = mem interesting_name Undefined {depth = 1 : i64, name = "bar", portNames = ["MPORT", "io_deq_bits_MPORT"], readLatency = 0 : i32, writeLatency = 1 : i32} :
      ; CHECK: !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data: uint<3>, mask: uint<1>>,
      ; CHECK: !firrtl.bundle<addr: uint<1>, en: uint<1>, clk: clock, data flip: uint<3>>

  ; CHECK-LABEL: module private @mem_no_ports() {
  ; CHECK-NEXT: }
  ; https://github.com/llvm/circt/issues/531
  module mem_no_ports :
    mem bar : @[Decoupled.scala 218:16]
      data-type => UInt<3>
      depth => 1
      read-latency => 0
      write-latency => 1
      read-under-write => undefined

  ; CHECK-LABEL: module private @issue354(out %tmp5: !firrtl.sint<19>) {
  module issue354 :
    output tmp5: SInt<19>
    tmp5 <= SInt<19>(8)
     ; CHECK: %c8_si19 = constant 8 : !firrtl.const.sint<19>
     ; CHECK: [[VAL:%.*]] = constCast %c8_si19 : (!firrtl.const.sint<19>) -> !firrtl.sint<19>
     ; CHECK: strictconnect %tmp5, [[VAL]] : !firrtl.sint<19>

   ; CHECK-LABEL: module private @issue347
  module issue347 :
    output tmp12: SInt<4>
    tmp12 <= SInt<4>(-4)
    ; CHECK: %c-4_si4 = constant -4 : !firrtl.const.sint<4>

  ; CHECK-LABEL: extmodule private @issue183<A: si32 = -1>()
  extmodule issue183:
     parameter A = -1

  ; The Scala FIRRTL Compiler allows this for an aggregate node with an internal
  ; analog.
  ; CHECK-LABEL: module private @analog_in_aggregate_node
  module analog_in_aggregate_node:
    input a: { a: UInt<1>, b: Analog<1>}
    ; CHECK: %b = node interesting_name %a : !firrtl.bundle<a: uint<1>, b: analog<1>>
    node b = a

  ; Check that a register clock sink is converted to passive
  ; CHECK-LABEL: module private @register_clock_passive
  module register_clock_passive:
    input clkIn: Clock
    output clkOut: Clock
    clkOut <= clkIn
    ; CHECK: reg interesting_name %clkOut
    reg r: UInt<1>, clkOut

  ; Check that a register reset sink is converted to passive
  ; CHECK-LABEL: module private @register_reset_passive
  module register_reset_passive:
    input clk: Clock
    output rst: UInt<1>
    rst is invalid
    ; CHECK: regreset interesting_name %clk, %rst
    reg r: UInt<1>, clk with : (reset => (rst, UInt<1>(0)))

  ; Check that a register init sink is converted to passive
  ; CHECK-LABEL: module private @register_init_passive
  module register_init_passive:
    input clk: Clock
    input rst: UInt<1>
    output init: UInt<1>
    init is invalid
    ; CHECK: regreset interesting_name %clk, %rst, %init
    reg r: UInt<1>, clk with : (reset => (rst, init))

  ; https://github.com/llvm/circt/issues/492
  ; CHECK-LABEL: module private @WriteOnlyMemIssue492
  module WriteOnlyMemIssue492 :
    input clock: Clock
    input wAddr: UInt<4>
    input wEn: UInt<1>
    input wMask: UInt<1>
    input wData: UInt<8>

    mem memory:
      data-type => UInt<8>
      depth => 16
      writer => w
      read-latency => 0
      write-latency => 1
      read-under-write => undefined

    memory.w.clk <= clock
    memory.w.en <= wEn
    memory.w.addr <= wAddr
    memory.w.mask <= wMask
    memory.w.data <= wData

  ; https://github.com/llvm/circt/issues/559
  ; CHECK-LABEL: module private @TrickyIssue559
  module TrickyIssue559:
    input input: UInt<1>
    output output: UInt<1>
    ; CHECK: strictconnect %output, %input
    output <= input

  ; CHECK-LABEL: module private @CheckInvalids
  module CheckInvalids_in0 :
    input in0 : UInt<1>
    ; CHECK-NOT: connect %in0
    ; CHECK-NOT: strictconnect %in0
    in0 is invalid

  module CheckInvalids_in1 :
    input in1 : { a : UInt<1>, b : UInt<1> }
    ; CHECK-NOT: connect %in1
    ; CHECK-NOT: strictconnect %in1
    in1 is invalid

  module CheckInvalids_in2 :
    input in2 : { a : UInt<1>, flip b : UInt<1>}
    ; CHECK: [[IN2_B:%.+]] = subfield %in2[b]
    ; CHECK: [[INV:%.+]] = invalidvalue
    ; CHECK: strictconnect [[IN2_B]], [[INV]]
    in2 is invalid

  module CheckInvalids_in3 :
    input in3 : {a : { b : UInt<1>, flip c : UInt<1>}}
    ; CHECK: [[IN3_A:%.+]] = subfield %in3[a]
    ; CHECK: [[IN3_A_C:%.+]] = subfield [[IN3_A]][c]
    ; CHECK: [[INV:%.+]] = invalidvalue
    ; CHECK: strictconnect [[IN3_A_C]], [[INV]]
    in3 is invalid

  module CheckInvalids_out0 :
    output out0 : UInt<1>
    ; CHECK: [[INV:%.+]] = invalidvalue
    ; CHECK: strictconnect %out0, [[INV]]
    out0 is invalid

  module CheckInvalids_out1 :
    output out1 : { a : UInt<1>, b : UInt<1> }
    ; CHECK: [[INV:%.+]] = invalidvalue : !firrtl.bundle<a: uint<1>, b: uint<1>>
    ; CHECK: strictconnect %out1, [[INV]]
    out1 is invalid

  module CheckInvalids_out2 :
    output out2 : { a : UInt<1>, flip b : UInt<1>}
    ; CHECK: [[OUT2_A:%.+]] = subfield %out2[a]
    ; CHECK: [[INV:%.+]] = invalidvalue
    ; CHECK: strictconnect [[OUT2_A]], [[INV]]
    out2 is invalid

  module CheckInvalids_out3 :
    output out3 : {a : { b : UInt<1>, flip c : UInt<1>}}
    ; CHECK: [[OUT3_A:%.+]] = subfield %out3[a]
    ; CHECK: [[OUT3_A_B:%.+]] = subfield [[OUT3_A]][b]
    ; CHECK: [[INV:%.+]] = invalidvalue
    ; CHECK: strictconnect [[OUT3_A_B]], [[INV]]
    out3 is invalid

  module CheckInvalids_wires :
    ; CHECK: %wire0 = wire
    ; CHECK: [[INV:%.+]] = invalidvalue
    ; CHECK: strictconnect %wire0, [[INV]]
    wire wire0 : UInt<1>
    wire0 is invalid

    ; CHECK: %wire1 = wire
    ; CHECK: [[WIRE1_B:%.+]] = subfield %wire1[b]
    ; CHECK: [[WIRE1_A:%.+]] = subfield %wire1[a]
    ; CHECK: [[INV:%.+]] = invalidvalue
    ; CHECK: strictconnect [[WIRE1_A]], [[INV]]
    ; CHECK: [[INV:%.+]] = invalidvalue
    ; CHECK: strictconnect [[WIRE1_B]], [[INV]]
    wire wire1 : {a : UInt<1>, flip b : UInt<1> }
    wire1 is invalid

    ; An analog in the leaf of a wire should be attached not connected.
    ; CHECK: %wire2 = wire
    ; CHECK: [[WIRE2_X:%.+]] = subfield %wire2[x]
    ; CHECK: [[WIRE2_X_B:%.+]] = subfield [[WIRE2_X]][b]
    ; CHECK: [[WIRE2_X_A:%.+]] = subfield [[WIRE2_X]][a]
    ; CHECK: [[INV:%.+]] = invalidvalue
    ; CHECK: strictconnect [[WIRE2_X_A]], [[INV]]
    ; CHECK-NOT: attach [[WIRE2_X_B]], [[INV]]
    wire wire2 : {x : {flip a : UInt<1>, flip b: Analog<1> } }
    wire2 is invalid

    ; https://github.com/llvm/circt/issues/563
    ; CHECK: %U0_in0, %U0_in1, %U0_out0, %U0_out1 = instance U0 interesting_name @mod_0_563
    inst U0 of mod_0_563

    ; CHECK: [[INV:%.+]] = invalidvalue
    ; CHECK: strictconnect %U0_in0, [[INV]]
    ; CHECK: [[INV:%.+]] = invalidvalue : !firrtl.bundle<a: uint<5>>
    ; CHECK: strictconnect %U0_in1, [[INV]]
    U0 is invalid

  ; This reference is declared after its first use.
  ; https://github.com/llvm/circt/issues/163
  module mod_0_563 :
    input in0: UInt<5>
    input in1: { a : UInt<5> }
    output out0: UInt<5>
    output out1: { a : UInt<5> }
    out0 <= in0
    out1 <= in1

  ; https://github.com/llvm/circt/issues/606
  ; CHECK-LABEL: module private @mutableSubIndex606
  module mutableSubIndex606 :
    output io : UInt<1>[8]
    ; CHECK:  %0 = subindex %io[0] : !firrtl.vector<uint<1>, 8>
    ; CHECK: [[VAL:%.*]] = constCast %c0_ui1 : (!firrtl.const.uint<1>) -> !firrtl.uint<1>
    ; CHECK: strictconnect %0, [[VAL]] : !firrtl.uint<1>
    io[0] <= UInt<1>("h00")


  ; https://github.com/llvm/circt/issues/782
  ; CHECK-LABEL: mem_madness782
  module mem_madness782:
    input clock: Clock
    input rAddr: UInt<4>
    input rEn: UInt<1>
    output rData: UInt<8>

    ; CHECK: %mem_r = mem interesting_name Undefined {depth = 16 : i64, name = "mem", portNames = ["r"], readLatency = 2 : i32, writeLatency = 1 : i32} : !firrtl.bundle<addr: uint<4>, en: uint<1>, clk: clock, data flip: uint<8>>
    mem mem:
      data-type => UInt<8>
      depth => 16
      reader => r
      read-latency => 2
      write-latency => 1
      read-under-write => undefined

    mem.r.clk <= clock
    mem.r.en <= rEn
    mem.r.addr <= rAddr
    rData <= mem.r.data

  ; Test that behavioral memory reads and writes both work and that flow checks
  ; don't fail here.  (A memory port should have duplex flow.)
  ; See: https://github.com/llvm/circt/issues/1058
  ; CHECK-LABEL: module private @BehavioralMemory
  module BehavioralMemory:
    input clock: Clock
    input rAddr: UInt<3>
    output rData: UInt<1>
    input wAddr: UInt<3>
    input wData: UInt<1>

    cmem a: UInt<1>[8]

    ; CHECK: strictconnect %rData, %r
    infer mport r = a[rAddr], clock
    rData <= r

    ; CHECK: strictconnect %w_data, %wData
    infer mport w = a[wAddr], clock
    w <= wData

  ; Test that a mux with an unknown width select line parses.  This is a check
  ; of the predicate enforced on UInt1Type.
  ; CHECK-LABEL: module private @MuxUnknownWidthSelect_Issue1108
  module MuxUnknownWidthSelect_Issue1108:
    input a: UInt<1>
    input b: UInt<1>
    input sel: UInt
    output c: UInt<8>
    c <= mux(sel, a, b)

  ; Test that a mux with aggregate type is still compatible even if the leaf
  ; types disagree in their width.
  ; CHECK-LABEL: module private @MuxAggregateWidthMismatch_Issue2806
  module MuxAggregateWidthMismatch_Issue2806:
    input a: UInt<1>[1]
    input b: UInt<32>[1]
    input x: {u: UInt<1>, v: UInt<2>}
    input y: {u: UInt<32>, v: UInt<2>}
    input sel: UInt<1>
    output c: UInt[1]
    output z: {u: UInt, v: UInt}
    ; CHECK: mux(%sel, %a, %b)
    ; CHECK-SAME: -> !firrtl.vector<uint<32>, 1>
    ; CHECK: mux(%sel, %x, %y)
    ; CHECK-SAME: -> !firrtl.bundle<u: uint<32>, v: uint<2>>
    c <= mux(sel, a, b)
    z <= mux(sel, x, y)

  ; CHECK-LABEL: extmodule private @RawStringParam
  ; CHECK-SAME:    <TYPE: none = "bit",
  ; CHECK-SAME:     FORMAT: none = "xyz_timeout=%d\\n",
  ; CHECK-SAME:     MIXED_QUOTES: none = "\22'\\\22">
  extmodule RawStringParam :
    parameter TYPE = 'bit'
    parameter FORMAT = 'xyz_timeout=%d\n'
    parameter MIXED_QUOTES = '"\'\"'
  ; "

  ; CHECK-LABEL: module private @issue1303
  module issue1303:
    output out: Reset
    out <= UInt(1)
    ; CHECK: %[[c1:.*]] = constant 1 : !firrtl.const.uint
    ; CHECK-NEXT: %[[c2:.*]] = resetCast %[[c1]]
    ; CHECK-NEXT: %[[c3:.*]] = constCast %[[c2]]
    ; CHECK-NEXT: strictconnect %out, %[[c3]] : !firrtl.reset


  ; CHECK-LABEL: @resetBundle
  module resetBundle:
    input a: {a: UInt<1>, b: AsyncReset}
    output b: {a: Reset, b: Reset}

    b <= a
    ; CHECK: %1 = subfield %a[a] : !firrtl.bundle<a: uint<1>, b: asyncreset>
    ; CHECK: %[[r1:.*]] = resetCast %1
    ; CHECK: strictconnect %0, %[[r1]] : !firrtl.reset
    ; CHECK: %3 = subfield %b[b] : !firrtl.bundle<a: reset, b: reset>
    ; CHECK: %4 = subfield %a[b] : !firrtl.bundle<a: uint<1>, b: asyncreset>
    ; CHECK: %[[r4:.*]] = resetCast %4
    ; CHECK: strictconnect %3, %[[r4]] : !firrtl.reset

  ; CHECK-LABEL: @WhenEncodedVerification
  module WhenEncodedVerification:
    input clock: Clock
    input cond: UInt<1>
    input enable: UInt<1>
    input not_reset: UInt<1>
    input value: UInt<42>

    ; rocket-chip properties
    when cond:
      printf(clock, enable, "assert:foo 0", value)
    ; CHECK: [[TMP:%.+]] = not %cond
    ; CHECK-NEXT: assert %clock, [[TMP]], %enable, "foo 0" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1> {eventControl = 0 : i32, isConcurrent = true}

    when cond:
      printf(clock, enable, "assume:foo 1", value)
    ; CHECK: [[TMP:%.+]] = not %cond
    ; CHECK-NEXT: assume %clock, [[TMP]], %enable, "foo 1" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1> {eventControl = 0 : i32, isConcurrent = true}

    when cond:
      printf(clock, enable, "cover:foo 2", value)
    ; CHECK-NEXT: cover %clock, %cond, %enable, "foo 2" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1> {eventControl = 0 : i32, isConcurrent = true}

    when cond:
      printf(clock, enable, "assert:foo_0:", value)
    ; CHECK: [[TMP:%.+]] = not %cond
    ; CHECK-NEXT: assert %clock, [[TMP]], %enable, "" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1> {eventControl = 0 : i32, isConcurrent = true, name = "foo_0"}

    when cond:
      printf(clock, enable, "assume:foo_1:", value)
    ; CHECK: [[TMP:%.+]] = not %cond
    ; CHECK-NEXT: assume %clock, [[TMP]], %enable, "" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1> {eventControl = 0 : i32, isConcurrent = true, name = "foo_1"}

    when cond:
      printf(clock, enable, "cover:foo_2:", value)
    ; CHECK-NEXT: cover %clock, %cond, %enable, "" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1> {eventControl = 0 : i32, isConcurrent = true, name = "foo_2"}

    when cond:
      printf(clock, enable, "assert:custom label 0:foo 3", value)
    ; CHECK: [[TMP:%.+]] = not %cond
    ; CHECK-NEXT: assert %clock, [[TMP]], %enable, "foo 3" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1> {eventControl = 0 : i32, isConcurrent = true, name = "custom label 0"}

    when cond:
      printf(clock, enable, "assume:custom label 1:foo 4", value)
    ; CHECK: [[TMP:%.+]] = not %cond
    ; CHECK-NEXT: assume %clock, [[TMP]], %enable, "foo 4" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1> {eventControl = 0 : i32, isConcurrent = true, name = "custom label 1"}

    when cond:
      printf(clock, enable, "cover:custom label 2:foo 5", value)
    ; CHECK-NEXT: cover %clock, %cond, %enable, "foo 5" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1> {eventControl = 0 : i32, isConcurrent = true, name = "custom label 2"}

    ; Optional `stop` with same clock and condition should be removed.
    when cond:
      printf(clock, enable, "assert:without_stop")
      stop(clock, enable, 1)
    ; CHECK: assert %clock, {{%.+}}, %enable, "without_stop" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>
    ; CHECK-NOT: stop

    when cond:
      ; expected-warning @+1 {{printf-encoded assertion has format string arguments which may cause lint warnings}}
      printf(clock, enable, "assert:foo 6, %d", value, value)
    ; CHECK: assert {{.+}} "foo 6, %d"(%value) : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<42>

    ; AssertNotX -- usually `cond` only checks for not-in-reset, and `enable` is
    ; just set to 1; the actual check `^value !== 'x` is implicit.
    when cond:
      printf(clock, enable, "assertNotX:%d:value must not be X!", value)
    ; CHECK: [[TMP1:%.+]] = not %cond
    ; CHECK: [[TMP2:%.+]] = xorr %value
    ; CHECK: [[TMP3:%.+]] = int.isX
    ; CHECK: [[TMP4:%.+]] = not
    ; CHECK: [[TMP5:%.+]] = or [[TMP1]], [[TMP4]]
    ; CHECK: assert %clock, [[TMP5]], %enable, "value must not be X!" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>
    ; CHECK-SAME: name = "notX"

    ; Chisel built-in assertions
    when cond:
      printf(clock, enable, "Assertion failed with value %d", value)
      stop(clock, enable, 1)
    ; CHECK: [[TMP:%.+]] = not %cond
    ; CHECK-NEXT: assert %clock, [[TMP]], %enable, "Assertion failed with value %d"(%value) : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<42>
    ; CHECK-SAME: format = "ifElseFatal"
    ; CHECK-SAME: isConcurrent = true
    ; CHECK-SAME: name = "chisel3_builtin"

    when cond:
      printf(clock, enable, "Assertion failed: some message with value %d", value)
      stop(clock, enable, 1)
    ; CHECK: [[TMP:%.+]] = not %cond
    ; CHECK-NEXT: assert %clock, [[TMP]], %enable, "Assertion failed: some message with value %d"(%value) : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<42>
    ; CHECK-SAME: format = "ifElseFatal"
    ; CHECK-SAME: isConcurrent = true
    ; CHECK-SAME: name = "chisel3_builtin"

    ; Verification Library Assertions

    ; Predicate modifier `noMod`
    when cond:
      printf(clock, enable, "Assertion failed: [verif-library-assert]<extraction-summary>{\"predicateModifier\":{\"type\":\"noMod\"},\"conditionalCompileToggles\":[{\"type\":\"unrOnly\"},{\"type\":\"formalOnly\"}],\"labelExts\":[\"label\",\"magic\"],\"format\":{\"type\":\"sva\"},\"baseMsg\":\"Hello Assert\"}", value)
      stop(clock, enable, 1)
    ; CHECK: [[TMP:%.+]] = not %cond
    ; CHECK-NEXT: assert %clock, [[TMP]], %enable, "Hello Assert"(%value) : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<42>
    ; CHECK-SAME: format = "sva"
    ; CHECK-SAME: guards = ["USE_UNR_ONLY_CONSTRAINTS", "USE_FORMAL_ONLY_CONSTRAINTS"]
    ; CHECK-SAME: name = "verif_library_label_magic"

    ; Predicate modifier `trueOrIsX`
    when cond:
      printf(clock, enable, "Assertion failed: [verif-library-assert]<extraction-summary>{\"predicateModifier\":{\"type\":\"trueOrIsX\"},\"conditionalCompileToggles\":[{\"type\":\"unrOnly\"},{\"type\":\"formalOnly\"}],\"labelExts\":[\"label\",\"magic\"],\"format\":{\"type\":\"sva\"},\"baseMsg\":\"Hello Assert\"}", value)
      stop(clock, enable, 1)
    ; CHECK: [[CONDINV:%.+]] = not %cond
    ; CHECK: [[TMP1:%.+]] = xorr [[CONDINV]]
    ; CHECK-NEXT: [[TMP2:%.+]] = verbatim.expr "{{[{][{]0[}][}]}} === 1'bx"([[TMP1]])
    ; CHECK-NEXT: [[TMP:%.+]] = or [[CONDINV]], [[TMP2]]
    ; CHECK-NEXT: assert %clock, [[TMP]], %enable, "Hello Assert"(%value) : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<42>
    ; CHECK-SAME: format = "sva"
    ; CHECK-SAME: guards = ["USE_UNR_ONLY_CONSTRAINTS", "USE_FORMAL_ONLY_CONSTRAINTS"]
    ; CHECK-SAME: name = "verif_library_label_magic"

    ; Verification Library Assumptions

    ; Predicate modifier `noMod`
    when cond:
      printf(clock, enable, "Assumption failed: [verif-library-assume]<extraction-summary>{\"predicateModifier\":{\"type\":\"noMod\"},\"conditionalCompileToggles\":[{\"type\":\"unrOnly\"},{\"type\":\"formalOnly\"}],\"labelExts\":[\"label\",\"voodoo\"],\"baseMsg\":\"Hello Assume\"}", value)
      stop(clock, enable, 1)
    ; CHECK: [[TMP:%.+]] = not %cond
    ; CHECK-NEXT: assume %clock, [[TMP]], %enable, "Hello Assume"(%value) : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<42>
    ; CHECK-SAME: guards = ["USE_UNR_ONLY_CONSTRAINTS", "USE_FORMAL_ONLY_CONSTRAINTS"]
    ; CHECK-SAME: name = "verif_library_label_voodoo"

    ; Predicate modifier `trueOrIsX`
    when cond:
      printf(clock, enable, "Assumption failed: [verif-library-assume]<extraction-summary>{\"predicateModifier\":{\"type\":\"trueOrIsX\"},\"conditionalCompileToggles\":[{\"type\":\"unrOnly\"},{\"type\":\"formalOnly\"}],\"labelExts\":[\"label\",\"voodoo\"],\"baseMsg\":\"Hello Assume\"}", value)
      stop(clock, enable, 1)
    ; CHECK: [[CONDINV:%.+]] = not %cond
    ; CHECK: [[TMP1:%.+]] = xorr [[CONDINV]]
    ; CHECK-NEXT: [[TMP2:%.+]] = verbatim.expr "{{[{][{]0[}][}]}} === 1'bx"([[TMP1]])
    ; CHECK-NEXT: [[TMP:%.+]] = or [[CONDINV]], [[TMP2]]
    ; CHECK-NEXT: assume %clock, [[TMP]], %enable, "Hello Assume"(%value) : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<42>
    ; CHECK-SAME: guards = ["USE_UNR_ONLY_CONSTRAINTS", "USE_FORMAL_ONLY_CONSTRAINTS"]
    ; CHECK-SAME: name = "verif_library_label_voodoo"

    ; New flavor of when-encoded verification that also includes an assert
    assert(clock, cond, enable, "hello")
    node not_cond = eq(cond, UInt<1>(0))
    when not_cond:
      printf(clock, enable, "Assertion failed: hello")
    ; CHECK-NOT: assert %clock, %cond, %enable, "hello" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>
    ; CHECK: [[TMP:%.+]] = not %not_cond
    ; CHECK-NEXT: assert %clock, [[TMP]], %enable, "Assertion failed: hello" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>
    ; CHECK-SAME: format = "ifElseFatal"
    ; CHECK-SAME: name = "chisel3_builtin"

    when not_reset:
      assert(clock, cond, enable, "hello outside reset")
      node not_cond2 = eq(cond, UInt<1>(0))
      when not_cond2:
        printf(clock, enable, "Assertion failed: hello outside reset")
    ; CHECK: when %not_reset : !firrtl.uint<1> {
    ; CHECK-NOT: assert %clock, %cond, %enable, "hello outside reset" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>
    ; CHECK: [[TMP:%.+]] = not %not_cond2
    ; CHECK-NEXT: assert %clock, [[TMP]], %enable, "Assertion failed: hello outside reset" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>
    ; CHECK-SAME: format = "ifElseFatal"
    ; CHECK-SAME: name = "chisel3_builtin"
    ; CHECK: }

    ; Check that the above doesn't error if the assert is a double user of the
    ; condition.
    when not_reset:
      assert(clock, UInt<1>(1), UInt<1>(1), "double user assert")
      node not_cond3 = eq(UInt<1>(1), UInt<1>(0))
      when not_cond3:
        printf(clock, UInt<1>(1), "Assertion failed: double user assert")
    ; CHECK-NOT: assert %clock, {{.+}} "double user assert" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>

    when cond :
        printf(clock, not(enable), "assert: bar")
    ; CHECK: [[NOT_ENABLE:%.+]] = not %enable
    ; CHECK-NEXT: [[NOT_COND:%.+]] = not %cond
    ; CHECK-NEXT: assert %clock, [[NOT_COND]], [[NOT_ENABLE]], " bar" : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1> {eventControl = 0 : i32, isConcurrent = true}

    ; Verification Library Covers

    ; Predicate modifier `noMod`
    when not(cond):
      printf(clock, enable, "Assertion failed: [verif-library-cover]<extraction-summary>{\"predicateModifier\":{\"type\":\"noMod\"},\"conditionalCompileToggles\":[{\"type\":\"unrOnly\"},{\"type\":\"formalOnly\"}],\"labelExts\":[\"cover\",\"label\"],\"format\":{\"type\":\"sva\"},\"baseMsg\":\"cover hello world\"}", value)
    assert(clock, cond, enable, "")
    ; CHECK: [[TMP_INV:%.+]] = not %cond
    ; CHECK: [[TMP:%.+]] = not [[TMP_INV]]
    ; CHECK-NOT: assert %clock, [[TMP]], %enable, "cover hello world"(%value) : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<42>
    ; CHECK: cover %clock, [[TMP]], %enable, "cover hello world"(%value) : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>, !firrtl.uint<42>
    ; CHECK-SAME: guards = ["USE_UNR_ONLY_CONSTRAINTS", "USE_FORMAL_ONLY_CONSTRAINTS"]
    ; CHECK-SAME: name = "verif_library_cover_label"

  module LargeMem :
    input clock : Clock
    input reset : Reset
    ; CHECK: !chirrtl.cmemory<vector<uint<8>, 16>, 34359738368>
    smem testharness : UInt<8>[16] [34359738368]
    node w_addr = UInt<36>(42) @[Cat.scala 31:58]
    write mport MPORT = testharness[w_addr], clock

  ; Module as identifier
  ; Test parsing of "module" as an identifier for instance and module names
  extmodule SomeModule:
    input in: UInt

  module ModuleAsIdentifier:
    inst module of SomeModule
    ; CHECK: instance module interesting_name @SomeModule
    module.in <= UInt(1)

  ; CHECK-LABEL: module private @EnumTypes
circuit EnumTypes:
  module EnumTypes:
    ; CHECK-SAME: in %i: !firrtl.enum<Some: uint<8>, None: uint<0>>
    input i : {| Some : UInt<8>, None |}
    output o : UInt<8>

    ; CHECK: %c0_ui8 = constant 0 : !firrtl.const.uint<8>
    ; CHECK: %0 = enumcreate Some(%c0_ui8) : (!firrtl.const.uint<8>) -> !firrtl.enum<Some: uint<8>, None: uint<0>>
    ; CHECK: %n = node interesting_name %0 : !firrtl.enum<Some: uint<8>, None: uint<0>>
    node n = {|Some : UInt<8>, None|}(Some, UInt<8>(0))

    ; CHECK: match %i : !firrtl.enum<Some: uint<8>, None: uint<0>> {
    match i:
      ; CHECK: case Some(%arg0) {
      ; CHECK:   strictconnect %o, %arg0 : !firrtl.uint<8>
      ; CHECK: }
      Some(x):
        o <= x
      ; CHECK: case None(%arg0) {
      ; CHECK:   %invalid_ui8 = invalidvalue : !firrtl.uint<8>
      ; CHECK:   strictconnect %o, %invalid_ui8 : !firrtl.uint<8>
      ; CHECK: }
      None:
        o is invalid
        
  ; CHECK-LABEL: module private @Properties
  module Properties :
    ; CHECK-SAME: out %a: !firrtl.string
    output a : String
    ; CHECK: %0 = string "hello"
    ; CHECK: propassign %a, %0 : !firrtl.string
    propassign a, String("hello")

  ; CHECK-LABEL: module private @RefsChild(
  ; CHECK-SAME: out %r: !firrtl.probe<uint<1>>
  ; CHECK-SAME: out %rw: !firrtl.rwprobe<uint<1>>
  module RefsChild :
    input in : UInt<1>
    output r : Probe<UInt<1>>
    output rw : RWProbe<UInt<1>>

    ; CHECK-NEXT: %[[NODE:.+]], %[[NODE_RWREF:.+]] = node
    ; CHECK-SAME: forceable
    node n = in
    ; CHECK-NEXT: %[[REF:.+]] = ref.send %[[NODE]]
    ; CHECK-NEXT: ref.define %r, %[[REF]]
    define r = probe(n)
    ; CHECK-NOT: ref.send
    ; CHECK-NEXT: ref.define %rw, %[[NODE_RWREF]]
    define rw = rwprobe(n)

  ; CHECK-LABEL: module private @Refs(
  module Refs :
    input in : UInt<1>
    output r : Probe<UInt<1>>
    output rw : RWProbe<UInt<1>>
    output out : UInt<1>
    output out2 : UInt<1>
    output out3 : UInt<3>
    ; CHECK-SAME: out %agg_out: !firrtl.probe<bundle<a: uint<1>, b: uint>>
    output agg_out : Probe<{a: UInt<1>, b: UInt}>

    ; CHECK-NEXT: %[[RC_IN:.+]], %[[RC_R:.+]], %[[RC_RW:.+]] = instance rc
    inst rc of RefsChild
    rc.in <= in
    ; CHECK: %[[OUTREF:.+]] = ref.send %out
    ; CHECK-NEXT: ref.define %r, %[[OUTREF]]
    define r = probe(out)
    ; CHECK-NEXT: ref.define %rw, %[[RC_RW]]
    ; CHECK-SAME: rwprobe
    define rw = rc.rw

    ; CHECK-NEXT: %[[READ_RC_R:.+]] = ref.resolve %[[RC_R]]
    ; CHECK-NEXT: connect %out, %[[READ_RC_R]]
    out <= read(rc.r)
    ; CHECK-NEXT: %[[READ_RC_RW:.+]] = ref.resolve %[[RC_RW]]
    ; CHECK-NEXT: connect %out, %[[READ_RC_RW]]
    out <= read(rc.rw)

    ; ref.sub parsing
    ; CHECK-DAG: %[[AGG:.+]] = wire interesting_name : !firrtl.bundle<a flip: uint<1>, b: uint>
    ; CHECK-DAG: %[[AGG2:.+]] = wire interesting_name : !firrtl.bundle<a: uint, b flip: uint<1>>
    wire agg : { flip a : UInt<1>, b : UInt }
    wire agg2 : { a : UInt, flip b : UInt<1> }
    ; CHECK-DAG: %[[AGG_B:.+]] = subfield %[[AGG]][b]
    ; CHECK-DAG: %[[AGG_B_PROBE:.+]] = ref.send %[[AGG_B]]
    ; CHECK-DAG: %[[READ_AGG_B_PROBE:.+]] = ref.resolve %[[AGG_B_PROBE]]
    ; CHECK-NEXT: %[[wfix:.*]] = widthCast %[[READ_AGG_B_PROBE]]
    ; CHECK-DAG: strictconnect %out2, %[[wfix]]
    out2 <= read(probe(agg.b))
    ; CHECK-DAG: %[[AGG2_PROBE:.+]] = ref.send %[[AGG2]]
    ; CHECK-DAG: %[[READ_AGG2_PROBE:.+]] = ref.resolve %[[AGG2_PROBE]]
    ; CHECK-DAG: %[[READ_AGG2_PROBE__B:.+]] = subfield %[[READ_AGG2_PROBE]][b]
    ; CHECK-DAG: strictconnect %out2, %[[READ_AGG2_PROBE__B]]
    out2 <= read(probe(agg2)).b

    ; CHECK: %[[AGG3:.+]] = wire
    wire agg3 : { a : UInt<1>, b : UInt }
    ; CHECK-NEXT: %[[AGG3_PROBE:.+]] = ref.send %[[AGG3]]
    ; CHECK-NEXT: ref.define %agg_out, %[[AGG3_PROBE]]
    define agg_out = probe(agg3)

    ; CHECK: %{{.+}}, %[[REM_R:.+]], %{{.+}}, %[[REM_R2:.+]] = instance rem
    ; CHECK-NEXT: %[[REM_R2_1:.+]] = ref.sub %[[REM_R2]][1]
    ; CHECK-NEXT: %[[REM_R2_1_A:.+]] = ref.sub %[[REM_R2_1]][0]
    ; CHECK-NEXT: %[[READ_REM_R2_1_A:.+]] = ref.resolve %[[REM_R2_1_A]]
    ; CHECK-NEXT: strictconnect %out3, %[[READ_REM_R2_1_A]]
    inst rem of RefExtMore
    out3 <= read(rem.r2[1].a)

    ; CHECK: %[[PROBE_IN:.+]] = ref.send %in
    ; CHECK-DAG: %[[READ_PROBE_IN:.+]] = ref.resolve %[[PROBE_IN]]
    ; CHECK-DAG: %[[SUM:.+]] = and %[[READ_PROBE_IN]],
    out <= and(read(probe(in)), UInt(1))

  ; CHECK-LABEL: module private @ForceRelease(
  module ForceRelease :
    input in : UInt<1>
    input clock : Clock
    input cond : UInt<1>

    ; CHECK-NEXT: %{{.+}}, %{{.+}}, %[[RC_RW:.+]] = instance rc
    inst rc of RefsChild
    rc.in <= in

    ; CHECK: ref.force_initial %[[TRUE:.+]], %[[RC_RW]], %{{.+}} : !firrtl.uint<1>, !firrtl.uint<1>
    force_initial(rc.rw, UInt<1>(0))

    ; CHECK: ref.force %clock, %cond, %[[RC_RW]], %{{.+}} : !firrtl.clock, !firrtl.uint<1>, !firrtl.uint<1>
    force(clock, cond, rc.rw, UInt<1>(1))
    ; CHECK: %[[NOT_COND:.+]] = not %cond
    ; CHECK: ref.release %clock, %[[NOT_COND]], %[[RC_RW]] : !firrtl.clock, !firrtl.uint<1>, !firrtl.rwprobe<uint<1>>
    release(clock, not(cond), rc.rw)

    ; CHECK-NEXT: %{{.+}}, %{{.+}}, %[[RC2_RW:.+]] = instance rc2
    inst rc2 of RefsChild
    rc2.in <= in
    ; CHECK: ref.release_initial %[[TRUE]], %[[RC2_RW]] : !firrtl.uint<1>, !firrtl.rwprobe<uint<1>>
    release_initial(rc2.rw)

  ; CHECK-LABEL: extmodule private @RefExt(
  ; CHECK-SAME:  in in: !firrtl.uint<1>, out r: !firrtl.probe<uint<1>>
  ; CHECK-SAME: internalPaths = ["in"]
  extmodule RefExt :
    input in : UInt<1>
    output r : Probe<UInt<1>>
    ref r is "in"

  ; CHECK-LABEL: extmodule private @RefExtMore(
  ; CHECK-SAME:  in in: !firrtl.uint<1>
  ; CHECK-SAME: out r: !firrtl.probe<uint<1>>
  ; CHECK-SAME: out data: !firrtl.uint<3>
  ; CHECK-SAME: out r2: !firrtl.probe<vector<bundle<a: uint<3>>, 3>>
  ; CHECK-SAME: internalPaths = ["path.to.internal.signal", "in"]
  extmodule RefExtMore :
    input in : UInt<1>
    output r : Probe<UInt<1>>
    output data : UInt<3>
    output r2 : Probe<{a : UInt<3>}[3]>
    ref r2 is "in"
    ref r is "path.to.internal.signal"


  ; CHECK-LABEL: extmodule private @RefExtABI
  ; CHECK-NOT: internalPaths
  extmodule RefExtABI :
    input in : UInt<1>
    output r : Probe<UInt<1>>
    output data : UInt<3>
    output r2 : Probe<{a : UInt<3>}[3]>


  ; CHECK-LABEL: module private @ProbeInvalidate
  ; CHECK-NEXT: }
  module ProbeInvalidate:
    output p : Probe<UInt<1>>
    p is invalid

  ; CHECK-LABEL: module private @NumericFields
  ; See: https://github.com/llvm/circt/issues/5110
  module NumericFields:
    input a: {0: {0: {bar: UInt<1>}}}
    output b: UInt<1>
    input c: {0: {0: {0: {bar: UInt<1>}}}}
    output d: UInt<1>

    ; CHECK:      %0 = subfield %c["0"]
    ; CHECK-NEXT: %1 = subfield %0["0"]
    ; CHECK-NEXT: %2 = subfield %1["0"]
    ; CHECK-NEXT: %3 = subfield %2[bar]
    ; CHECK-NEXT: %4 = subfield %a["0"]
    ; CHECK-NEXT: %5 = subfield %4["0"]
    ; CHECK-NEXT: %6 = subfield %5[bar]
    b <= a.0.0.bar
    d <= c.0.0.0.bar
    ; CHECK-NEXT: strictconnect %b, %6
    ; CHECK-NEXT: strictconnect %d, %3

  ; CHECK-LABEL: module private @ConstTypes(
  module ConstTypes:
    input c: const Clock         ; CHECK: %c: !firrtl.const.clock,
    input r: const Reset         ; CHECK: %r: !firrtl.const.reset,
    input ar: const AsyncReset   ; CHECK: %ar: !firrtl.const.asyncreset,
    input a: const Analog        ; CHECK: %a: !firrtl.const.analog,
    input a8: const Analog<8>    ; CHECK: %a8: !firrtl.const.analog<8>,
    input s: const SInt          ; CHECK: %s: !firrtl.const.sint,
    input s4: const SInt<4>      ; CHECK: %s4: !firrtl.const.sint<4>,
    input u: const UInt          ; CHECK: %u: !firrtl.const.uint,

    ; CHECK: %b: !firrtl.const.bundle<int_1 flip: uint<1>, int_out: uint<2>>
    input b: const {flip int_1 : UInt<1>, int_out : UInt<2>}
    ; CHECK: %b_constfields: !firrtl.bundle<int_1 flip: const.uint<1>, int_out: const.uint<2>>
    input b_constfields: {flip int_1 : const UInt<1>, int_out : const UInt<2>}
    ; CHECK: %mixedb: !firrtl.bundle<a: sint<1>, b: const.uint<2>>
    input mixedb: {a: SInt<1>, b: const UInt<2>}
    ; CHECK: %vec: !firrtl.const.vector<uint<1>, 4>) {
    input vec: const UInt<1>[4]

    ; CHECK-NEXT: %w = wire interesting_name : !firrtl.const.sint<4>
    wire w: const SInt<4>
    ; CHECK-NEXT: strictconnect %w, %s4 : !firrtl.const.sint<4>
    w <= s4
    ; CHECK-NEXT: %nonconst_w = wire interesting_name : !firrtl.sint<4>
    wire nonconst_w: SInt<4>
    ; CHECK-NEXT: [[CAST:%.+]] = constCast %s4 : (!firrtl.const.sint<4>) -> !firrtl.sint<4>
    ; CHECK-NEXT: strictconnect %nonconst_w, [[CAST]] : !firrtl.sint<4>
    nonconst_w <= s4

;// -----

; CHECK-LABEL: circuit "Foo_v3p0p0"
FIRRTL version 3.0.0
circuit Foo_v3p0p0:
  module Foo_v3p0p0:
    input clock: Clock
    input reset: UInt<1>
    input a: UInt<1>
    output b: UInt<1>

    ; CHECK: strictconnect %b, %a
    connect b, a

    ; CHECK: [[INV:%.+]]  = invalidvalue : !firrtl.uint<1>
    ; CHECK-NEXT: strictconnect %b, [[INV]] : !firrtl.uint<1>
    invalidate b

    ; CHECK:      %[[zero:[0-9A-Za-z_]+]] = constant 0
    ; CHECK-NEXT: %r = regreset interesting_name %clock, %reset, %[[zero]]
    regreset r: UInt<1>, clock, reset, UInt<1>(0)

  ; CHECK: module private @LiteralIdentifiers
  ; CHECK-SAME: in %_0: !firrtl.bundle<1: uint<1>>
  ; CHECK-SAME: out %_2: !firrtl.bundle<3: uint<1>>
  ; CHECK-SAME: portNames = ["0", "2"]
  module LiteralIdentifiers:
    input `0`: {`1`: UInt<1>}
    output `2`: {`3`: UInt<1>}

    ; CHECK-NEXT: %0 = subfield %_0["1"] : !firrtl.bundle<1: uint<1>>
    ; CHECK-NEXT: %1 = subfield %_2["3"] : !firrtl.bundle<3: uint<1>>
    ; CHECK-NEXT: strictconnect %1, %0 : !firrtl.uint<1>
    `2`.`3` <= `0`.`1`

    ; Ensure that `a` is equivalent to a.
    ; CHECK-NEXT: %a = node {{.+}}%_0
    node `a` = `0`
    ; CHECK-NEXT %b = node {{.+}}%a
    node b = a

;// -----

FIRRTL version 1.2.0
circuit Foo_v1p9p9:
  module Foo_v1p9p9:
    input a: UInt<1>
    output b: UInt<1>
    output auto : UInt<1>

    ; CHECK: strictconnect %b, %a
    b <- a

    ; CHECK: [[a0:%.+]] = subfield %bundleWithAnalog[a]
    ; CHECK: [[a1:%.+]] = subfield %bundleWithAnalog[a]
    ; CHECK: attach [[a1]], [[a0]]
    wire bundleWithAnalog : {a: Analog<1>}
    bundleWithAnalog <- bundleWithAnalog


    ; CHECK: [[A:%.+]] = subfield %out_0[member] : !firrtl.bundle<member: bundle<0: bundle<clock: clock, reset: uint<1>>>>
    ; CHECK: [[B:%.+]] = subfield [[A]]["0"] : !firrtl.bundle<0: bundle<clock: clock, reset: uint<1>>>
    ; CHECK: [[C:%.+]] = subfield [[B]][reset] : !firrtl.bundle<clock: clock, reset: uint<1>>
    ; CHECK: strictconnect %auto, [[C]] : !firrtl.uint<1>
    wire out_0 : { member : { 0 : { clock : Clock, reset : UInt<1>}}}
    auto <- out_0.member.0.reset @[Field 173:49]

;// -----
; Check reference expressions using literal identifiers or keywords.

; CHECK-LABEL: circuit "Probes_refexprs"
FIRRTL version 3.0.0
circuit Probes_refexprs:
  extmodule refs:
    output `0`: {`1`: Probe<{`2`: UInt<1>}>, rwprobe: RWProbe<UInt<1>>}
    output ref: {module: Probe<{when: UInt<1>}>}

  module Probes_refexprs:
    output `0`: {`1`: Probe<{`2`: UInt<1>}>}
    output out: UInt<1>

    inst `9` of refs
    inst ref of refs

    ; CHECK: ref.define
    define `0`.`1` = `9`.`0`.`1`

    ; CHECK-COUNT-4: ref.resolve
    node a = read(`9`.`0`.`1`).`2`
    node b = read(`9`.`0`.`1`.`2`)
    node c = read(`9`.ref.module).when
    ; Keyword as leading part of static ref expression:
    node d = read(ref.`0`.`1`.`2`)

    out <= and(and(a, b), and(c, d))

    ; CHECK: %[[TEST:.+]], %[[TEST_REF:.+]] = wire
    wire `test`: {`0`: UInt<1>, `b`: UInt<1>}
    `test`.`0` <= a
    `test`.`b` <= b
     ; CHECK: force_initial %{{.+}}, %[[TEST_REF]], %[[TEST]]
    force_initial(rwprobe(`test`), `test`)
    ; CHECK: force_initial
    force_initial(`9`.`0`.rwprobe, `test`.`0`)

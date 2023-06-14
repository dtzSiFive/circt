// RUN: circt-opt -firrtl-add-seqmem-ports %s | FileCheck %s

// Should create the output file even if there are no seqmems.
// CHECK-LABEL: circuit "NoMems" {
// CHECK-NOT: class = "sifive.enterprise.firrtl.AddSeqMemPortsFileAnnotation"
firrtl.circuit "NoMems" attributes {annotations = [
  {
    class = "sifive.enterprise.firrtl.AddSeqMemPortsFileAnnotation",
    filename = "sram.txt"
  }]} {
  module @NoMems() {}
  // CHECK: sv.verbatim "" {output_file = #hw.output_file<"metadata{{/|\\\\}}sram.txt", excludeFromFileList>}
}

// Test for when there is a memory but no ports are added. The output file
// should be empty.
// CHECK-LABEL: circuit "NoAddedPorts"  {
firrtl.circuit "NoAddedPorts" attributes {annotations = [
  {
    class = "sifive.enterprise.firrtl.AddSeqMemPortsFileAnnotation",
    filename = "sram.txt"
  }]} {
  // CHECK: memmodule @MWrite_ext(in W0_addr: !firrtl.uint<4>, in W0_en: !firrtl.uint<1>, in W0_clk: !firrtl.clock, in W0_data: !firrtl.uint<42>)
  memmodule @MWrite_ext(in W0_addr: !firrtl.uint<4>, in W0_en: !firrtl.uint<1>, in W0_clk: !firrtl.clock, in W0_data: !firrtl.uint<42>) attributes {dataWidth = 42 : ui32, depth = 12 : ui64, extraPorts = [], maskBits = 1 : ui32, numReadPorts = 0 : ui32, numReadWritePorts = 0 : ui32, numWritePorts = 1 : ui32, readLatency = 1 : ui32, writeLatency = 1 : ui32}
  module @NoAddedPorts() {
    %0:4 = instance MWrite_ext  @MWrite_ext(in W0_addr: !firrtl.uint<4>, in W0_en: !firrtl.uint<1>, in W0_clk: !firrtl.clock, in W0_data: !firrtl.uint<42>)
  }
  // CHECK: sv.verbatim "" {output_file = #hw.output_file<"metadata{{/|\\\\}}sram.txt", excludeFromFileList>}
}

// Test for when there is no memory and we try to add ports. The output file
// should be empty.
// CHECK-LABEL: circuit "NoMemory"  {
// CHECK-NOT: class = "sifive.enterprise.firrtl.AddSeqMemPortAnnotation"
firrtl.circuit "NoMemory" attributes {annotations = [
  {
    class = "sifive.enterprise.firrtl.AddSeqMemPortsFileAnnotation",
    filename = "sram.txt"
  },
  {
    class = "sifive.enterprise.firrtl.AddSeqMemPortAnnotation",
    name = "user_input",
    input = true,
    width = 5
  }]} {
  module @NoMemory() {
  }
  // CHECK: sv.verbatim "" {output_file = #hw.output_file<"metadata{{/|\\\\}}sram.txt", excludeFromFileList>}
}

// Test for a single added port.
// CHECK-LABEL: circuit "Single"  {
firrtl.circuit "Single" attributes {annotations = [
  {
    class = "sifive.enterprise.firrtl.AddSeqMemPortAnnotation",
    name = "user_input",
    input = true,
    width = 3
  }]} {
  // CHECK:      memmodule @MWrite_ext
  // CHECK-SAME:    in user_input: !firrtl.uint<3>
  // CHECK-SAME:    extraPorts = [{direction = "input", name = "user_input", width = 3 : ui32}]
  memmodule @MWrite_ext(in W0_addr: !firrtl.uint<4>, in W0_en: !firrtl.uint<1>, in W0_clk: !firrtl.clock, in W0_data: !firrtl.uint<42>) attributes {dataWidth = 42 : ui32, depth = 12 : ui64, extraPorts = [], maskBits = 1 : ui32, numReadPorts = 0 : ui32, numReadWritePorts = 0 : ui32, numWritePorts = 1 : ui32, readLatency = 1 : ui32, writeLatency = 1 : ui32}
  module @Single() {
    %0:4 = instance MWrite_ext  @MWrite_ext(in W0_addr: !firrtl.uint<4>, in W0_en: !firrtl.uint<1>, in W0_clk: !firrtl.clock, in W0_data: !firrtl.uint<42>)
  }
}

// Test for two ports added.
// CHECK-LABEL: circuit "Two"  {
firrtl.circuit "Two" attributes {annotations = [
  {
    class = "sifive.enterprise.firrtl.AddSeqMemPortAnnotation",
    name = "user_input",
    input = true,
    width = 3
  },
  {
    class = "sifive.enterprise.firrtl.AddSeqMemPortAnnotation",
    name = "user_output",
    input = false,
    width = 4
  }]} {
  memmodule @MWrite_ext(in W0_addr: !firrtl.uint<4>, in W0_en: !firrtl.uint<1>, in W0_clk: !firrtl.clock, in W0_data: !firrtl.uint<42>) attributes {dataWidth = 42 : ui32, depth = 12 : ui64, extraPorts = [], maskBits = 1 : ui32, numReadPorts = 0 : ui32, numReadWritePorts = 0 : ui32, numWritePorts = 1 : ui32, readLatency = 1 : ui32, writeLatency = 1 : ui32}
  // The ports should be attached in the opposite order of the annotations.
  // CHECK: module @Child(out %sram_0_user_output: !firrtl.uint<4>, in %sram_0_user_input: !firrtl.uint<3>)
  module @Child() {
    %0:4 = instance MWrite_ext  @MWrite_ext(in W0_addr: !firrtl.uint<4>, in W0_en: !firrtl.uint<1>, in W0_clk: !firrtl.clock, in W0_data: !firrtl.uint<42>)
    // CHECK: strictconnect %sram_0_user_output, %MWrite_ext_user_output : !firrtl.uint<4>
    // CHECK: strictconnect %MWrite_ext_user_input, %sram_0_user_input : !firrtl.uint<3>
  }
  // CHECK: module @Two(out %sram_0_user_output: !firrtl.uint<4> [{class = "firrtl.transforms.DontTouchAnnotation"}], in %sram_0_user_input: !firrtl.uint<3> [{class = "firrtl.transforms.DontTouchAnnotation"}], out %sram_1_user_output: !firrtl.uint<4> [{class = "firrtl.transforms.DontTouchAnnotation"}], in %sram_1_user_input: !firrtl.uint<3> [{class = "firrtl.transforms.DontTouchAnnotation"}])
  module @Two() {
    instance child0 @Child()
    instance child1 @Child()
    // CHECK: %child0_sram_0_user_output, %child0_sram_0_user_input = instance child0  @Child
    // CHECK: %child1_sram_0_user_output, %child1_sram_0_user_input = instance child1  @Child
    // CHECK: strictconnect %sram_0_user_output, %child0_sram_0_user_output : !firrtl.uint<4>
    // CHECK: strictconnect %child0_sram_0_user_input, %sram_0_user_input : !firrtl.uint<3>
    // CHECK: strictconnect %sram_1_user_output, %child1_sram_0_user_output : !firrtl.uint<4>
    // CHECK: strictconnect %child1_sram_0_user_input, %sram_1_user_input : !firrtl.uint<3>

  }
}

// Test for a ports added port with a DUT. The input ports should be wired to
// zero, and not wired up through the test harness.
// CHECK-LABEL: circuit "TestHarness"  {
firrtl.circuit "TestHarness" attributes {annotations = [
  {
    class = "sifive.enterprise.firrtl.AddSeqMemPortAnnotation",
    name = "user_input",
    input = true,
    width = 3
  },
  {
    class = "sifive.enterprise.firrtl.AddSeqMemPortAnnotation",
    name = "user_output",
    input = false,
    width = 4
  }]} {
  memmodule @MWrite_ext(in W0_addr: !firrtl.uint<4>, in W0_en: !firrtl.uint<1>, in W0_clk: !firrtl.clock, in W0_data: !firrtl.uint<42>) attributes {dataWidth = 42 : ui32, depth = 12 : ui64, extraPorts = [], maskBits = 1 : ui32, numReadPorts = 0 : ui32, numReadWritePorts = 0 : ui32, numWritePorts = 1 : ui32, readLatency = 1 : ui32, writeLatency = 1 : ui32}
  // CHECK: module @DUT(out %sram_0_user_output: !firrtl.uint<4> [{class = "firrtl.transforms.DontTouchAnnotation"}], in %sram_0_user_input: !firrtl.uint<3> [{class = "firrtl.transforms.DontTouchAnnotation"}])
  module @DUT() attributes {annotations = [{class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}]} {
    %0:4 = instance MWrite_ext  @MWrite_ext(in W0_addr: !firrtl.uint<4>, in W0_en: !firrtl.uint<1>, in W0_clk: !firrtl.clock, in W0_data: !firrtl.uint<42>)
  }
  module @TestHarness() {
    instance dut @DUT()
    // CHECK: %dut_sram_0_user_output, %dut_sram_0_user_input = instance dut @DUT(out sram_0_user_output: !firrtl.uint<4> [{class = "firrtl.transforms.DontTouchAnnotation"}], in sram_0_user_input: !firrtl.uint<3> [{class = "firrtl.transforms.DontTouchAnnotation"}])
    // CHECK: %c0_ui3 = constant 0 : !firrtl.uint<3>
    // CHECK: strictconnect %dut_sram_0_user_input, %c0_ui3 : !firrtl.uint<3>
  }
}

// Slightly more complicated test.
firrtl.circuit "Complex" attributes {annotations = [
  {
    class = "sifive.enterprise.firrtl.AddSeqMemPortsFileAnnotation",
    filename = "sram.txt"
  },
  {
    class = "sifive.enterprise.firrtl.AddSeqMemPortAnnotation",
    name = "user_input",
    input = true,
    width = 3
  },
  {
    class = "sifive.enterprise.firrtl.AddSeqMemPortAnnotation",
    name = "user_output",
    input = false,
    width = 4
  }]} {
  memmodule @MWrite_ext(in W0_addr: !firrtl.uint<4>, in W0_en: !firrtl.uint<1>, in W0_clk: !firrtl.clock, in W0_data: !firrtl.uint<42>) attributes {dataWidth = 42 : ui32, depth = 12 : ui64, extraPorts = [], maskBits = 1 : ui32, numReadPorts = 0 : ui32, numReadWritePorts = 0 : ui32, numWritePorts = 1 : ui32, readLatency = 1 : ui32, writeLatency = 1 : ui32}
  module @Child() {
    %0:4 = instance MWrite_ext  @MWrite_ext(in W0_addr: !firrtl.uint<4>, in W0_en: !firrtl.uint<1>, in W0_clk: !firrtl.clock, in W0_data: !firrtl.uint<42>)
  }
  module @DUT() attributes {annotations = [{class = "sifive.enterprise.firrtl.MarkDUTAnnotation"}]} {
    // Double check that these instances now have symbols on them:
    // CHECK: instance MWrite_ext sym @MWrite_ext
    %0:4 = instance MWrite_ext  @MWrite_ext(in W0_addr: !firrtl.uint<4>, in W0_en: !firrtl.uint<1>, in W0_clk: !firrtl.clock, in W0_data: !firrtl.uint<42>)
    // CHECK: instance child sym @child  @Child
    instance child @Child()
    // CHECK: instance MWrite_ext sym @MWrite_ext_0  @MWrite_ext
    %1:4 = instance MWrite_ext  @MWrite_ext(in W0_addr: !firrtl.uint<4>, in W0_en: !firrtl.uint<1>, in W0_clk: !firrtl.clock, in W0_data: !firrtl.uint<42>)
  }
  module @Complex() {
    instance dut @DUT()
  }
  // CHECK: sv.verbatim
  // CHECK-SAME{LITERAL}: 0 -> {{0}}.{{1}}
  // CHECK-SAME{LITERAL}: 1 -> {{0}}.{{2}}.{{3}}
  // CHECK-SAME{LITERAL}: 2 -> {{0}}.{{4}}
  // CHECK-SAME: {output_file = #hw.output_file<"metadata{{/|\\\\}}sram.txt", excludeFromFileList>,
  // CHECK-SAME: symbols = [@DUT, #hw.innerNameRef<@DUT::@MWrite_ext>, #hw.innerNameRef<@DUT::@child>, #hw.innerNameRef<@Child::@MWrite_ext>, #hw.innerNameRef<@DUT::@MWrite_ext_0>]
}
